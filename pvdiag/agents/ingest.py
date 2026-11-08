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
        total_rows, fetched_days = 0, 0
        for _, st in stations.iterrows():
            if ctx.station_ids and st["station_id"] not in ctx.station_ids:
                continue
            extra = json.loads(st.get("extra_json") or "{}")
            pconf = {**extra.get("platform_config", {}), **self.platform_overrides.get(st["platform"], {})}
            if st["platform"] != "csv" and not pconf:
                res.alerts.append(f"{st['station_id']} 平台 {st['platform']} 未配置凭证，跳过拉取（仅用已落库数据）")
                self._refresh_features(ctx, st["station_id"], [ctx.run_date])
                continue
            try:
                adapter = build_adapter(st["platform"], st["station_id"], pconf, ctx.store)
            except Exception as e:
                res.alerts.append(f"{st['station_id']} 适配器初始化失败: {e}")
                continue
            start = ctx.run_date - timedelta(days=self.backfill_days)
            have = set(ctx.store.available_dates("string", st["station_id"]))
            # 断点续拉：只拉缺的日期（当天总是重拉以补齐增量）
            missing = [start + timedelta(days=k) for k in range((ctx.run_date - start).days + 1)
                       if (start + timedelta(days=k)).isoformat() not in have or (start + timedelta(days=k)) == ctx.run_date]
            if not missing:
                self._refresh_features(ctx, st["station_id"], [ctx.run_date])
                continue
            try:
                fr = adapter.fetch(min(missing), max(missing))
            except Exception as e:
                res.alerts.append(f"{st['station_id']} 拉取失败: {e}")
                ctx.store.alert(ctx.run_date, "error", self.name, f"{st['station_id']} 拉取失败: {e}")
                continue
            for n in fr.notes:
                res.alerts.append(f"{st['station_id']}: {n}")
            total_rows += self.ingest_frames(ctx, st["station_id"], fr.strings, fr.inverters, fr.irradiance)
            fetched_days += len(missing)
            ctx.store.set_state(f"ingest:{st['station_id']}:last_ok", {"date": ctx.run_date.isoformat(),
                                                                       "rows": int(len(fr.strings)), "calls": fr.calls_used})
        res.metrics = {"rows_written": total_rows, "days": fetched_days}
        res.summary = f"写入 {total_rows} 行明细，覆盖 {fetched_days} 站·日"
        return res

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
            row = compute_station_daily(station_id, d, inv_daily, station_noon_cv(i, d, cfg), hist, capacity, cfg,
                                        (irr_by_day or {}).get(d))
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
