"""阶段3 · 基线智能体（确定"标准"）

① 晴好天筛选器：全站中午功率 ≥ 晴天峰值 35%（且窗口变异系数 ≤0.15 抗云影）
② 横向基线【主力】：同机同配置各串中午窗口电流中位数互比，滚动 30 个晴好天
   ratio_{s,d} = I_s,d / median_group,d ；dev_today = ratio − 1 ；baseline_ratio = 窗口内 ratio 中位数
③ 竖向基线【辅助】：有辐照数据时 Y=E/H 滚动分布；无辐照时用"机间互比"（逆变器单位装机中午功率）兜底整机问题
④ 基线分段管理：清洗/维修/换串事件后自动重置（事件前数据作废）
冷启动：窗口内晴好天 < min_clear_days → 只做同日互比，不做"对自身历史"的判定
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import Agent, AgentContext, AgentResult
from .profile import ProfileAgent


def _robust_std(x: pd.Series) -> float:
    """1.4826 × 中位绝对偏差（排除 0 电流串），样本 <3 时返回 NaN"""
    v = x[x > -0.9].to_numpy(dtype=float)
    if len(v) < 3:
        return np.nan
    return float(1.4826 * np.median(np.abs(v - np.median(v))))


class BaselineAgent(Agent):
    name = "baseline"
    stage = "阶段3 基线建模"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        n_rows, n_cold, n_inv = 0, 0, 0
        from ..parallel import map_stations
        sids = ctx.stations()
        for sid, (bl, ibl) in zip(sids, map_stations(ctx.settings, ctx, self.compute_for_date, sids, ctx.run_date)):
            if not bl.empty:
                ctx.store.upsert_df("baselines", bl)
                n_rows += len(bl)
                n_cold += int(bl["cold_start"].sum())
            if not ibl.empty:
                ctx.store.upsert_df("inverter_baselines", ibl)
                n_inv += len(ibl)
        res.metrics = {"strings": n_rows, "cold_start": n_cold, "inverters": n_inv}
        res.summary = f"{n_rows} 串基线（冷启动 {n_cold}），{n_inv} 台逆变器机间互比"
        if n_rows and n_cold / n_rows > 0.5:
            res.alerts.append(f"超过一半组串处于冷启动（晴好天样本不足 {ctx.settings.baseline.min_clear_days}），仅做同日互比")
        return res

    # ------------------------------------------------------------------ 核心
    def compute_for_date(self, ctx: AgentContext, station_id: str, d: date) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cfg, th = ctx.settings.baseline, ctx.settings.thresholds
        lookback = d - timedelta(days=max(120, cfg.rolling_clear_days * 3))
        strings = ctx.store.strings(station_id)
        if strings.empty:
            return pd.DataFrame(), pd.DataFrame()
        feats = ctx.store.daily_features(station_id, lookback, d)
        if feats.empty or (feats["date"] == d.isoformat()).sum() == 0:
            return pd.DataFrame(), pd.DataFrame()
        weather = ctx.store.station_daily(station_id, lookback, d).set_index("date")
        events = ctx.store.events()
        inv_daily = ctx.store.query(
            "SELECT i.* FROM inverter_daily i JOIN inverters v ON v.inverter_id=i.inverter_id "
            "WHERE v.station_id=? AND i.date BETWEEN ? AND ?", (station_id, lookback.isoformat(), d.isoformat()))

        keep = ["string_id", "inverter_id", "group_key", "azimuth", "tilt", "known_shading"] + [c for c in ("module_type", "device_type") if c in strings.columns]
        f = feats.merge(strings[keep], on="string_id", how="left")
        f["weather_tag"] = f["date"].map(weather["weather_tag"]) if not weather.empty else "nodata"
        stale = f["stale_run_min"].fillna(0) >= th.stale_run_min if "stale_run_min" in f.columns else pd.Series(False, index=f.index)
        f["valid"] = (f["noon_median_i"] >= th.zero_current_a) & (f["stuck"] == 0) & (f["zero_all_day"] == 0) & f["noon_median_i"].notna() & (~stale)
        # 限发标记（逆变器当天限发 → 该日不参与竖向/机间比，互比仍有效）
        if not inv_daily.empty:
            lim = inv_daily[["inverter_id", "date", "limited_flag"]].rename(columns={"limited_flag": "limited"})
            f = f.merge(lim, on=["inverter_id", "date"], how="left")
            f["limited"] = f["limited"].fillna(0).astype(int)
        else:
            f["limited"] = 0

        # ---- 朝向物理归一（季节性）：同机各串朝向/倾角不同时，按晴空倾斜面辐照把中午电流折算到"参考朝向"再互比
        #      （冬天东向比南向低 25%，夏天只低 6%——这是物理，不是故障；档案错了才是异常）
        f["orient_exp_ratio"] = self._orientation_expectation(f, strings, ctx, station_id)
        # 双面组件混装：相对同机多数组件类型按背面增益归一（双面串天然高 ~8%，不是别人低）
        if "module_type" in f.columns and f["module_type"].notna().any():
            mt = f["module_type"].fillna("mono")
            ref_bif = strings.groupby("inverter_id")["module_type"].agg(lambda s: (s.fillna("mono") == "bifacial").mean() > 0.5) if "module_type" in strings.columns else None
            is_ref_bif = f["inverter_id"].map(ref_bif).fillna(False).astype(bool) if ref_bif is not None else pd.Series(False, index=f.index)
            g = th.bifacial_gain
            factor = np.where((mt == "bifacial") & ~is_ref_bif, g, np.where((mt != "bifacial") & is_ref_bif, 1.0 / g, 1.0))
            f["orient_exp_ratio"] = f["orient_exp_ratio"] * factor
        f["noon_norm"] = f["noon_median_i"] / f["orient_exp_ratio"]

        # ---- 群组中位数（三级回退：配置群组 → 整机 → 无）
        v = f[f["valid"]]
        g1 = v.groupby(["date", "group_key"])["noon_norm"].agg(["median", "count"]).rename(
            columns={"median": "g_med", "count": "g_n"})
        g2 = v.groupby(["date", "inverter_id"])["noon_norm"].agg(["median", "count"]).rename(
            columns={"median": "i_med", "count": "i_n"})
        f = f.merge(g1, left_on=["date", "group_key"], right_index=True, how="left")
        f = f.merge(g2, left_on=["date", "inverter_id"], right_index=True, how="left")
        use_group = f["g_n"] >= cfg.min_group_size
        use_inv = (~use_group) & (f["i_n"] >= cfg.min_group_size)
        f["group_level"] = np.where(use_group, "group", np.where(use_inv, "inverter", "none"))
        f["group_median_i"] = np.where(use_group, f["g_med"], np.where(use_inv, f["i_med"], np.nan))
        f["group_n"] = np.where(use_group, f["g_n"], np.where(use_inv, f["i_n"], 0)).astype(int)
        f["ratio"] = f["noon_norm"] / f["group_median_i"]
        f.loc[~f["valid"], "ratio"] = np.where(f.loc[~f["valid"], "zero_all_day"] == 1, 0.0, np.nan)  # 0 电流串 ratio=0
        f["dev"] = f["ratio"] - 1
        # ---- 群组当日噪声：有效串偏差的稳健标准差（1.4826×MAD）。云影在大电站各逆变器间不同步，
        #      "晴好天"也可能噪声 3%~7%；此时 -5% 只是噪声，不是低发（统计显著性门控用）
        f["group_level_key"] = np.where(use_group, f["group_key"].astype(str), f["inverter_id"].astype(str))
        vv = f[f["valid"] & f["dev"].notna() & (f["dev"] > -0.9)]
        keys = ["date", "group_level_key"]
        med = vv.groupby(keys)["dev"].transform("median")
        mad = (vv["dev"] - med).abs().groupby([vv["date"], vv["group_level_key"]]).agg(["median", "size"])
        noise = (1.4826 * mad["median"]).where(mad["size"] >= 3).rename("group_noise")      # 与 _robust_std 同口径，向量化
        f = f.merge(noise, left_on=keys, right_index=True, how="left")

        # ---- 每串滚动统计（numpy 向量化：按串取下标一次，不在循环里切 DataFrame；万串级每日秒级）
        today = d.isoformat()
        f = f.sort_values(["string_id", "date"]).reset_index(drop=True)
        seg_map = self._segment_starts(strings, events, d)
        irr_series = weather["irradiance_kwh_m2"] if (not weather.empty and "irradiance_kwh_m2" in weather.columns
                                                     and weather["irradiance_kwh_m2"].notna().any()) else None
        col = {c: f[c].to_numpy() for c in ("date", "weather_tag", "ratio", "dev", "valid", "group_noise", "zero_all_day", "limited")}
        nz_all = f["noon_zero_frac"].to_numpy(dtype=float) if "noon_zero_frac" in f.columns else np.full(len(f), np.nan)
        irr_all = f["date"].map(irr_series).to_numpy(dtype=float) if irr_series is not None else None
        energy_all = f["day_energy_kwh"].to_numpy(dtype=float) if irr_series is not None else None
        today_rows = {sid: i for sid, i in zip(f.loc[f["date"] == today, "string_id"], f.index[f["date"] == today])}
        recent_cut = (d - timedelta(days=th.rainy_streak_caution_days)).isoformat()
        rows: List[Dict] = []
        for sid, idx in f.groupby("string_id", sort=False).indices.items():
            ti = today_rows.get(sid)
            if ti is None:
                continue
            seg_start = seg_map.get(sid)
            if seg_start:
                idx = idx[col["date"][idx] >= seg_start]
            dates = col["date"][idx]; wtag = col["weather_tag"][idx]
            ratio = col["ratio"][idx].astype(float); dev = col["dev"][idx].astype(float)
            valid = col["valid"][idx].astype(bool); noise = col["group_noise"][idx].astype(float)
            zad = col["zero_all_day"][idx]
            before = dates < today
            clear = wtag == "clear"
            hist_mask = clear & before & ~np.isnan(ratio) & valid
            hist = ratio[hist_mask][-cfg.rolling_clear_days:]
            baseline_n = int(len(hist))
            baseline_ratio = float(np.median(hist)) if baseline_n else np.nan
            cold = int(baseline_n < cfg.min_clear_days)
            cur = f.iloc[ti]
            ratio_today, dev_today = cur["ratio"], cur["dev"]
            dev_vs_own = (ratio_today / baseline_ratio - 1) if (not cold and pd.notna(ratio_today) and baseline_ratio) else np.nan
            # 晴好天序列（含今天）用于持续度；群组噪声过大的日子跳过；中午整段 0 输出（停机）的日子跳过
            seq = clear & ~np.isnan(dev) & (dates <= today)
            devs = dev[seq]
            noises = np.nan_to_num(noise[seq], nan=0.0)
            skip_day = (np.nan_to_num(nz_all[idx][seq]) >= th.outage_noon_skip_frac) & (zad[seq] == 0)
            noises = np.where(skip_day, np.inf, noises)
            seq_dates = dates[seq]
            p5, idx5 = self._streak_gated(devs, noises, th.fault_floor, th.noise_k, th.noise_skip)
            p3, idx3 = self._streak_gated(devs, noises, th.normal_floor, th.noise_k, th.noise_skip)
            first_breach = str(seq_dates[idx3]) if p3 > 0 and idx3 is not None else None
            roll7 = float(np.median(devs[-7:])) if len(devs) else np.nan
            oc = dev[(wtag == "overcast") & ~np.isnan(dev) & valid][-5:]
            overcast_dev = float(np.median(oc)) if len(oc) >= 2 else np.nan
            n_clear_recent = int(np.sum(clear & (dates > recent_cut)))
            v_dev = np.nan
            if irr_all is not None:
                h = irr_all[idx]; y = energy_all[idx] / np.where(h == 0, np.nan, h)
                yh = y[before & clear & (col["limited"][idx] == 0) & ~np.isnan(y)][-30:]
                yt = y[dates == today]
                if len(yh) >= 5 and len(yt) and not np.isnan(yt[0]):
                    v_dev = float(yt[0] / np.median(yh) - 1)
            rows.append({
                "string_id": sid, "date": today, "segment_start": seg_start, "group_key": cur["group_key"],
                "group_n": int(cur["group_n"]), "group_level": cur["group_level"],
                "group_median_i": cur["group_median_i"], "ratio_today": ratio_today,
                "baseline_ratio": baseline_ratio, "baseline_n": baseline_n,
                "dev_today": dev_today, "dev_vs_own": dev_vs_own, "dev_roll7": roll7,
                "persist_days_5": int(p5), "persist_days_3": int(p3), "first_breach_date": first_breach,
                "v_dev": v_dev, "overcast_dev": overcast_dev,
                "n_clear_recent": n_clear_recent, "cold_start": cold,
                "weather_tag": cur["weather_tag"], "limited": int(cur["limited"]),
                "group_noise": None if pd.isna(cur.get("group_noise")) else float(cur["group_noise"]),
                "orient_exp_ratio": float(cur["orient_exp_ratio"]) if pd.notna(cur.get("orient_exp_ratio")) else 1.0,
            })
        bl = pd.DataFrame(rows)

        # ---- 机间互比（整机问题兜底）
        ibl = self._inverter_baselines(ctx, station_id, d, f, inv_daily, strings)
        return bl, ibl

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _orientation_expectation(f: pd.DataFrame, strings: pd.DataFrame, ctx: AgentContext, station_id: str) -> pd.Series:
        """每行（串·日）的期望电流比 = 该串朝向 vs 同机参考朝向（众数）在当日中午窗口的晴空 POA 比；朝向缺失/相同 → 1.0"""
        from ..solar import expected_noon_ratio, lon_offset_minutes
        out = pd.Series(1.0, index=f.index)
        if "azimuth" not in f.columns or f["azimuth"].isna().all() or f["tilt"].isna().all():
            return out
        st = ctx.store.query("SELECT lat, lon FROM stations WHERE station_id=?", (station_id,))
        lat = float(st["lat"].iloc[0]) if not st.empty and pd.notna(st["lat"].iloc[0]) else 30.0
        lon_off = lon_offset_minutes(float(st["lon"].iloc[0]) if not st.empty and pd.notna(st["lon"].iloc[0]) else None,
                                     ctx.settings.baseline.tz_meridian_deg)
        # 同机参考朝向 = 该逆变器各串 (azimuth, tilt) 的众数
        ref = {}
        for inv, g in strings.dropna(subset=["azimuth", "tilt"]).groupby("inverter_id"):
            pair = g.groupby(["azimuth", "tilt"]).size().idxmax()
            ref[inv] = (float(pair[0]), float(pair[1]))
        if not ref:
            return out
        az = f["azimuth"].to_numpy(dtype=float); tilt = f["tilt"].to_numpy(dtype=float)
        inv_ids = f["inverter_id"].to_numpy(); dates = f["date"].to_numpy()
        cache: Dict[tuple, float] = {}
        vals = np.ones(len(f))
        for i in range(len(f)):
            r = ref.get(inv_ids[i])
            if r is None or np.isnan(az[i]) or np.isnan(tilt[i]) or (az[i] == r[0] and tilt[i] == r[1]):
                continue
            key = (dates[i], az[i], tilt[i], r)
            if key not in cache:
                cache[key] = expected_noon_ratio(lat, date.fromisoformat(dates[i]), az[i], tilt[i], r[0], r[1],
                                                 (ctx.settings.baseline.noon_start, ctx.settings.baseline.noon_end), lon_off)
            vals[i] = max(cache[key], 0.2)
        return pd.Series(vals, index=f.index)

    @staticmethod
    def _segment_starts(strings: pd.DataFrame, events: pd.DataFrame, d: date) -> Dict[str, Optional[str]]:
        """每串的基线分段起点（最近一次清洗/维修/换串等事件日），一次算完"""
        if events.empty:
            return {}
        return {r["string_id"]: ProfileAgent.segment_start_for(r, events, d) for r in strings.to_dict("records")}

    @staticmethod
    def _streak(devs: np.ndarray, floor: float) -> int:
        n = 0
        for x in devs[::-1]:
            if np.isnan(x) or x > floor:
                break
            n += 1
        return n

    @staticmethod
    def _streak_gated(devs: np.ndarray, noises: np.ndarray, floor: float, k: float, skip: float) -> Tuple[int, Optional[int]]:
        """带噪声门控的连续低发天数（从最近一天往前数）：
        - 群组噪声 > skip：云影日，跳过（既不计入也不打断）
        - 偏差 ≤ floor 且 偏差 ≤ -k×噪声：计 1 天
        - 否则打断
        返回 (天数, 连续段最早一天在序列中的下标)"""
        n, first = 0, None
        for i in range(len(devs) - 1, -1, -1):
            x, s = devs[i], noises[i] if i < len(noises) else np.nan
            if np.isnan(x):
                break
            s = 0.0 if (s is None or np.isnan(s)) else float(s)
            if s > skip:
                continue
            if x <= floor and x <= -k * s:
                n += 1
                first = i
            else:
                break
        return n, first

    @staticmethod
    def _vertical_dev(g: pd.DataFrame, weather: pd.DataFrame, today: str) -> float:
        """竖向：单位辐照产出 Y=E/H（需 station_daily.irradiance_kwh_m2）；无辐照返回 NaN"""
        if weather.empty or "irradiance_kwh_m2" not in weather.columns or weather["irradiance_kwh_m2"].isna().all():
            return np.nan
        h = g["date"].map(weather["irradiance_kwh_m2"])
        y = g["day_energy_kwh"] / h.replace(0, np.nan)
        hist = y[(g["date"] < today) & (g["weather_tag"] == "clear") & (g["limited"] == 0)].dropna().tail(30)
        cur = y[g["date"] == today]
        if len(hist) < 5 or cur.empty or pd.isna(cur.iloc[0]):
            return np.nan
        return float(cur.iloc[0] / hist.median() - 1)

    def _inverter_baselines(self, ctx: AgentContext, station_id: str, d: date, f: pd.DataFrame,
                            inv_daily: pd.DataFrame, strings: pd.DataFrame) -> pd.DataFrame:
        """机间互比：每台逆变器取"各串中午电流中位数 / 组件 Imp"作为整机参照（对单串故障鲁棒），
        与站内其它逆变器同日互比。整机 MPPT 异常/限发会表现为整机参照偏低，而互比看不出。"""
        if f.empty:
            return pd.DataFrame()
        imp = strings.set_index("string_id")["module_imp"] if "module_imp" in strings.columns else None
        x = f[f["valid"]].copy()
        x["imp"] = x["string_id"].map(imp).fillna(1.0) if imp is not None else 1.0
        x["norm_i"] = x["noon_median_i"] / x["imp"]
        x = x.groupby(["date", "inverter_id"]).agg(norm=("norm_i", "median"), n=("norm_i", "size"),
                                                   weather_tag=("weather_tag", "first")).reset_index()
        x = x[x["n"] >= 3]
        if not inv_daily.empty:
            flags = inv_daily.set_index(["inverter_id", "date"])[["offline_flag", "limited_flag"]]
            x = x.merge(flags, left_on=["inverter_id", "date"], right_index=True, how="left")
            x = x[(x["offline_flag"].fillna(0) == 0) & (x["limited_flag"].fillna(0) == 0)]
        if x.empty:
            return pd.DataFrame()
        med = x.groupby("date")["norm"].median().rename("st_med")
        cnt = x.groupby("date")["norm"].size().rename("st_n")
        x = x.merge(med, left_on="date", right_index=True).merge(cnt, left_on="date", right_index=True)
        x = x[x["st_n"] >= 2]
        x["inv_dev"] = x["norm"] / x["st_med"] - 1
        rows = []
        today = d.isoformat()
        for iid, g in x.groupby("inverter_id"):
            g = g.sort_values("date")
            cur = g[g["date"] == today]
            if cur.empty:
                continue
            clear = g[g["weather_tag"] == "clear"]
            streak = self._streak(clear["inv_dev"].to_numpy(dtype=float), ctx.settings.thresholds.fault_floor)
            rows.append({"inverter_id": iid, "date": today, "norm_noon": float(cur["norm"].iloc[0]),
                         "station_median_norm": float(cur["st_med"].iloc[0]), "station_n": int(cur["st_n"].iloc[0]),
                         "inv_dev": float(cur["inv_dev"].iloc[0]), "persist_days": int(streak),
                         "weather_tag": cur["weather_tag"].iloc[0]})
        return pd.DataFrame(rows)
