"""阶段1 · 数据接入智能体

职责：
① 三家云 API / CSV 接入组串级 5 分钟数据（历史回补 + 每日增量），断点续拉、配额先算后拉
② 数据清洗：电流<0.5A 剔除、缺失值标记"缺测"而非零发、时钟漂移对齐 5 分钟格、半成品文件校验
③ 建库：全量 5 分钟明细（Parquet 分区）+ 日汇总特征库（每串每日：中午中位数、曲线签名向量、天气标记）
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..adapters import build_adapter
from ..features import (compute_inverter_daily, compute_station_daily, compute_string_daily, station_noon_cv)
from .base import Agent, AgentContext, AgentResult


class IngestAgent(Agent):
    name = "ingest"
    stage = "阶段1 数据接入与治理"

    def __init__(self, backfill_days: int = 0, platform_overrides: Optional[Dict[str, Dict]] = None):
        self.backfill_days = backfill_days
        self.platform_overrides = platform_overrides or {}

    # ------------------------------------------------------------------ 入口
    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        stations = ctx.store.stations()
        if stations.empty:
            res.status, res.summary = "skipped", "无电站档案，先导入 stations/strings"
            return res
        total_rows, fetched_days, healed_days = 0, 0, 0
        icfg = ctx.settings.ingest
        for _, st in stations.iterrows():
            sid = st["station_id"]
            if ctx.station_ids and sid not in ctx.station_ids:
                continue
            extra = json.loads(st.get("extra_json") or "{}")
            pconf = {**extra.get("platform_config", {}), **self.platform_overrides.get(st["platform"], {})}
            if st["platform"] != "csv" and not pconf:
                res.alerts.append(f"{sid} 平台 {st['platform']} 未配置凭证，跳过拉取（仅用已落库数据）")
                self._refresh_features(ctx, sid, [ctx.run_date])
                continue
            try:
                adapter = build_adapter(st["platform"], sid, pconf, ctx.store)
            except Exception as e:
                res.alerts.append(f"{sid} 适配器初始化失败: {e}")
                self._record_failure(ctx, sid, f"适配器初始化失败: {e}", res)
                continue
            # 断点续拉 + 自愈补缺：backfill 窗口内缺的日期 + 回看 heal_lookback_days 内的缺口（当天总是重拉以补齐增量）
            have = set(ctx.store.available_dates("string", sid))
            start = ctx.run_date - timedelta(days=self.backfill_days)
            window = [start + timedelta(days=k) for k in range((ctx.run_date - start).days + 1)]
            missing = [d for d in window if d.isoformat() not in have or d == ctx.run_date]
            heal_start = ctx.run_date - timedelta(days=icfg.heal_lookback_days)
            gaps = [heal_start + timedelta(days=k) for k in range((ctx.run_date - heal_start).days)
                    if (heal_start + timedelta(days=k)).isoformat() not in have and (heal_start + timedelta(days=k)) < start]
            gaps = [g for g in gaps if g >= self._earliest_data(ctx, sid)]          # 比首个数据日更早的不算缺口
            gaps = gaps[-icfg.max_heal_days_per_run:]
            if gaps:
                res.alerts.append(f"{sid} 发现 {len(gaps)} 个历史缺口日（{gaps[0]}~{gaps[-1]}），本次补拉")
            if not missing and not gaps:
                self._refresh_features(ctx, sid, [ctx.run_date])
                continue
            ok_any, failed_ranges = False, []
            for lo, hi in self._ranges(sorted(set(missing) | set(gaps))):
                fr, err = self._fetch_with_retry(ctx, adapter, lo, hi, pconf)
                if fr is None:
                    failed_ranges.append(f"{lo}~{hi}: {err}")
                    continue
                ok_any = True
                for n in fr.notes:
                    res.alerts.append(f"{sid}: {n}")
                created = self._ensure_profile(ctx, sid, adapter, fr, pconf)
                if created:
                    res.alerts.append(f"{sid} 自动建立档案骨架：{created['inverters']} 台逆变器、{created['strings']} 路（n_modules/朝向待补；"
                                      f"华为站可用 pvdiag platform hints 由 MPPT 容量反推）")
                total_rows += self.ingest_frames(ctx, sid, fr.strings, fr.inverters, fr.irradiance)
                n_days = (hi - lo).days + 1
                fetched_days += n_days
                healed_days += sum(1 for g in gaps if lo <= g <= hi)
            if failed_ranges:                 # 一次运行只计一次失败（不按区间数累加）
                self._record_failure(ctx, sid, "拉取失败 " + "；".join(failed_ranges)[:300], res)
            if ok_any:
                ctx.store.set_state(f"ingest:{sid}:last_ok", {"date": ctx.run_date.isoformat(), "at": datetime.now().isoformat()})
                if not failed_ranges:
                    ctx.store.set_state(f"ingest:{sid}:failures", 0)
        res.metrics = {"rows_written": total_rows, "days": fetched_days, "healed_gap_days": healed_days}
        res.summary = f"写入 {total_rows} 行明细，覆盖 {fetched_days} 站·日" + (f"，补齐历史缺口 {healed_days} 天" if healed_days else "")
        return res

    # ------------------------------------------------------------------ 档案骨架
    @staticmethod
    def _ensure_profile(ctx: AgentContext, station_id: str, adapter, fr, pconf: Dict) -> Optional[Dict[str, int]]:
        """云平台新站第一次拉到数据而档案为空 → 用设备清单 + 样本里出现过的路建立骨架（只填 string_id/inverter_id/pv_index/mppt），
        流水线即可跑同机互比；组件数/朝向/并联数等由人工或 platform hints 补。已有档案的站不动。"""
        from .profile import ProfileAgent
        have_inv = ctx.store.inverters(station_id)
        have_str = ctx.store.strings(station_id)
        if (not have_inv.empty and not have_str.empty) or fr.strings is None or fr.strings.empty:
            return None
        created = {"inverters": 0, "strings": 0}
        if have_inv.empty:
            try:
                devs = adapter.list_devices()
            except Exception:
                devs = pd.DataFrame()
            if devs.empty:
                ids = sorted(fr.inverters["inverter_id"].unique()) if fr.inverters is not None and not fr.inverters.empty else \
                    sorted(fr.strings["string_id"].str.replace(r"-pv\d+$", "", regex=True).unique())
                devs = pd.DataFrame({"inverter_id": ids, "name": ids, "model": "", "rated_ac_kw": None})
            inv_df = pd.DataFrame({"inverter_id": devs["inverter_id"], "station_id": station_id, "name": devs.get("name", devs["inverter_id"]),
                                   "model": devs.get("model", ""), "rated_ac_kw": devs.get("rated_ac_kw"),
                                   "extra_json": [json.dumps({k: (None if pd.isna(v) else v) for k, v in r.items()
                                                              if k in ("esn", "dev_id", "ps_key", "sn")}, ensure_ascii=False) for r in devs.to_dict("records")]})
            ProfileAgent.import_inverters(ctx.store, inv_df)
            created["inverters"] = int(len(inv_df))
        if have_str.empty:
            ipm = int(pconf.get("inputs_per_mppt", 2))
            channel_is_mppt = bool(pconf.get("channel_is_mppt", getattr(adapter, "platform", "") == "sungrow"))
            sid_series = pd.Series(sorted(fr.strings["string_id"].unique()))
            pv = sid_series.str.extract(r"pv(\d+)$")[0].astype(float)
            inv = sid_series.str.replace(r"-pv\d+$", "", regex=True)
            st_df = pd.DataFrame({"string_id": sid_series, "station_id": station_id, "inverter_id": inv, "pv_index": pv,
                                  "mppt": pv if channel_is_mppt else ((pv - 1) // ipm + 1),
                                  "n_modules": None, "azimuth": None, "tilt": None, "notes": "auto:skeleton"})
            ProfileAgent.import_strings(ctx.store, st_df)
            created["strings"] = int(len(st_df))
        return created

    # ------------------------------------------------------------------ 自愈工具
    @staticmethod
    def _ranges(days: List[date]) -> List[tuple]:
        """连续日期合并成区间，减少 API 调用"""
        out: List[tuple] = []
        for d in days:
            if out and (d - out[-1][1]).days == 1:
                out[-1] = (out[-1][0], d)
            else:
                out.append((d, d))
        return out

    @staticmethod
    def _earliest_data(ctx: AgentContext, station_id: str) -> date:
        ds = ctx.store.available_dates("string", station_id)
        return date.fromisoformat(ds[0]) if ds else ctx.run_date

    def _fetch_with_retry(self, ctx: AgentContext, adapter, lo: date, hi: date, pconf: Dict):
        """指数退避重试；全部失败且配置了 fallback_csv_root → 降级读本地文件（已有拉数脚本的落地目录）"""
        import time
        icfg = ctx.settings.ingest
        err = None
        for attempt in range(icfg.retries):
            try:
                return adapter.fetch(lo, hi), None
            except Exception as e:
                err = e
                ctx.logger.warning("%s 拉取 %s~%s 第 %d 次失败: %s", adapter.station_id, lo, hi, attempt + 1, e)
                if attempt < icfg.retries - 1:
                    time.sleep(min(icfg.backoff_seconds * (2 ** attempt), 60))
        fb = pconf.get("fallback_csv_root")
        if fb:
            try:
                fr = build_adapter("csv", adapter.station_id, {"root": fb}, ctx.store).fetch(lo, hi)
                fr.notes.append(f"平台拉取失败（{err}），已降级读取本地文件 {fb}")
                return fr, None
            except Exception as e2:
                err = f"{err}；降级读本地文件也失败: {e2}"
        return None, err

    def _record_failure(self, ctx: AgentContext, station_id: str, msg: str, res: AgentResult) -> None:
        n = int(ctx.store.get_state(f"ingest:{station_id}:failures", 0) or 0) + 1
        ctx.store.set_state(f"ingest:{station_id}:failures", n)
        level = "error" if n >= ctx.settings.ingest.escalate_after_failures else "warn"
        text = f"{station_id} {msg}（连续第 {n} 次）" + ("，已升级告警：检查授权/配额/网络" if level == "error" else "")
        if level == "error":
            ctx.store.alert(ctx.run_date, "error", self.name, text)   # error 直接落库（基类只会把 res.alerts 记成 warn）
            res.summary = (res.summary + "；" if res.summary else "") + text
        else:
            res.alerts.append(text)

    # ------------------------------------------------------------------ 直接灌入 DataFrame（模拟/测试/旧文件）
    def ingest_frames(self, ctx: AgentContext, station_id: str, strings: pd.DataFrame,
                      inverters: pd.DataFrame, irradiance: Optional[pd.DataFrame] = None) -> int:
        n = 0
        days: List[date] = []
        if strings is not None and not strings.empty:
            strings = strings.copy()
            strings["ts"] = pd.to_datetime(strings["ts"])
            n += ctx.store.write_samples("string", station_id, strings)
            days = sorted(set(strings["ts"].dt.date))
        if inverters is not None and not inverters.empty:
            inverters = inverters.copy()
            inverters["ts"] = pd.to_datetime(inverters["ts"])
            n += ctx.store.write_samples("inverter", station_id, inverters)
            days = sorted(set(days) | set(inverters["ts"].dt.date))
        irr_by_day: Dict[date, float] = {}
        if irradiance is not None and not irradiance.empty:
            irr = irradiance.copy()
            irr["ts"] = pd.to_datetime(irr["ts"])
            irr_by_day = (irr.groupby(irr["ts"].dt.date)["ghi_w_m2"].sum() * (5 / 60) / 1000.0).to_dict()
        self._refresh_features(ctx, station_id, days, irr_by_day)
        return n

    # ------------------------------------------------------------------ 特征重算
    def _refresh_features(self, ctx: AgentContext, station_id: str, days: List[date],
                          irr_by_day: Optional[Dict[date, float]] = None) -> None:
        cfg, th = ctx.settings.baseline, ctx.settings.thresholds
        st = ctx.store.query("SELECT capacity_kwp FROM stations WHERE station_id=?", (station_id,))
        capacity = float(st["capacity_kwp"].iloc[0]) if not st.empty and pd.notna(st["capacity_kwp"].iloc[0]) else None
        inv = ctx.store.inverters(station_id)
        rated = {r.inverter_id: r.rated_ac_kw for r in inv.itertuples() if pd.notna(r.rated_ac_kw)} if not inv.empty else {}
        from ..irradiance import resolve_daily_irradiance
        irr_resolved = resolve_daily_irradiance(ctx.settings, ctx.store, station_id, sorted(days), irr_by_day)
        for d in sorted(days):
            s = ctx.store.read_samples("string", station_id, d)
            i = ctx.store.read_samples("inverter", station_id, d)
            if s.empty and i.empty:
                continue
            sf = compute_string_daily(s, d, cfg, th)
            inv_daily = compute_inverter_daily(i, d, cfg, rated)
            if inv_daily.empty and not sf.empty:
                # 平台没给逆变器功率 → 用组串 I×V 合成逆变器功率曲线做天气判定
                i = self._synth_inverter_power(s, ctx.store.strings(station_id))
                inv_daily = compute_inverter_daily(i, d, cfg, rated)
            inv_daily = self._unflag_frozen_limited(sf, inv_daily, ctx.store.strings(station_id), th.stale_run_min)
            hist = ctx.store.station_daily(station_id, d - timedelta(days=cfg.peak_lookback_days + 5), d)
            irr_val, irr_src = irr_resolved.get(d, (None, "none"))
            row = compute_station_daily(station_id, d, inv_daily, station_noon_cv(i, d, cfg), hist, capacity, cfg, irr_val, irr_src)
            ctx.store.upsert_df("daily_features", sf)
            ctx.store.upsert_df("inverter_daily", inv_daily)
            ctx.store.upsert_df("station_daily", pd.DataFrame([row]))

    @staticmethod
    def _unflag_frozen_limited(sf: pd.DataFrame, inv_daily: pd.DataFrame, profiles: pd.DataFrame, stale_min: int = 90) -> pd.DataFrame:
        """数据冻结 ≠ 限发：功率"平顶"若伴随该机大多数组串电流钉死 ≥90 分钟，是采集冻结（走数据问题），不标限发。
        （NREL 标注片段 inv_2173 的教训：钉死日全部被误标成限发）"""
        if sf.empty or inv_daily.empty or "stale_run_min" not in sf.columns or profiles.empty:
            return inv_daily
        m = profiles.set_index("string_id")["inverter_id"]
        x = sf.assign(inverter_id=sf["string_id"].map(m)).dropna(subset=["inverter_id"])
        frozen = x.groupby("inverter_id")["stale_run_min"].apply(lambda v: float((v.fillna(0) >= stale_min).mean()) >= 0.5)
        bad = set(frozen[frozen].index)
        if bad:
            inv_daily = inv_daily.copy()
            mask = inv_daily["inverter_id"].isin(bad) & (inv_daily["limited_flag"] == 1)
            inv_daily.loc[mask, "limited_flag"] = 0
            inv_daily.loc[mask, "limited_slots_json"] = "[]"
        return inv_daily

    @staticmethod
    def _synth_inverter_power(strings: pd.DataFrame, profiles: pd.DataFrame) -> pd.DataFrame:
        m = profiles.set_index("string_id")["inverter_id"].to_dict() if not profiles.empty else {}
        df = strings.copy()
        df["inverter_id"] = df["string_id"].map(m).fillna(df["string_id"].str.rsplit("-", n=1).str[0])
        df["p"] = pd.to_numeric(df["current_a"], errors="coerce").clip(lower=0) * pd.to_numeric(df["voltage_v"], errors="coerce").fillna(0) / 1000.0
        out = df.groupby(["inverter_id", "ts"], as_index=False)["p"].sum().rename(columns={"p": "pac_kw"})
        out["status"] = None
        return out
