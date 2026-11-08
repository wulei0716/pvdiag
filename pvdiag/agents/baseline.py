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


class BaselineAgent(Agent):
    name = "baseline"
    stage = "阶段3 基线建模"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        n_rows, n_cold, n_inv = 0, 0, 0
        for sid in ctx.stations():
            bl, ibl = self.compute_for_date(ctx, sid, ctx.run_date)
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

        f = feats.merge(strings[["string_id", "inverter_id", "group_key", "azimuth", "tilt", "known_shading"]],
                        on="string_id", how="left")
        f["weather_tag"] = f["date"].map(weather["weather_tag"]) if not weather.empty else "nodata"
        f["valid"] = (f["noon_median_i"] >= th.zero_current_a) & (f["stuck"] == 0) & (f["zero_all_day"] == 0) & f["noon_median_i"].notna()
        # 限发标记（逆变器当天限发 → 该日不参与竖向/机间比，互比仍有效）
        lim = inv_daily.set_index(["inverter_id", "date"])["limited_flag"] if not inv_daily.empty else None
        f["limited"] = [int(lim.get((r.inverter_id, r.date), 0)) if lim is not None else 0 for r in f.itertuples()]

        # ---- 群组中位数（三级回退：配置群组 → 整机 → 无）
        v = f[f["valid"]]
        g1 = v.groupby(["date", "group_key"])["noon_median_i"].agg(["median", "count"]).rename(
            columns={"median": "g_med", "count": "g_n"})
        g2 = v.groupby(["date", "inverter_id"])["noon_median_i"].agg(["median", "count"]).rename(
            columns={"median": "i_med", "count": "i_n"})
        f = f.merge(g1, left_on=["date", "group_key"], right_index=True, how="left")
        f = f.merge(g2, left_on=["date", "inverter_id"], right_index=True, how="left")
        use_group = f["g_n"] >= cfg.min_group_size
        use_inv = (~use_group) & (f["i_n"] >= cfg.min_group_size)
        f["group_level"] = np.where(use_group, "group", np.where(use_inv, "inverter", "none"))
        f["group_median_i"] = np.where(use_group, f["g_med"], np.where(use_inv, f["i_med"], np.nan))
        f["group_n"] = np.where(use_group, f["g_n"], np.where(use_inv, f["i_n"], 0)).astype(int)
        f["ratio"] = f["noon_median_i"] / f["group_median_i"]
        f.loc[~f["valid"], "ratio"] = np.where(f.loc[~f["valid"], "zero_all_day"] == 1, 0.0, np.nan)  # 0 电流串 ratio=0
        f["dev"] = f["ratio"] - 1

        # ---- 每串滚动统计
        today = d.isoformat()
        rows: List[Dict] = []
        for sid, g in f.groupby("string_id", sort=False):
            g = g.sort_values("date")
            srow = strings.loc[strings["string_id"] == sid].iloc[0]
            seg_start = ProfileAgent.segment_start_for(srow, events, d)
            if seg_start:
                g = g[g["date"] >= seg_start]
            cur = g[g["date"] == today]
            if cur.empty:
                continue
            cur = cur.iloc[0]
            clear_hist = g[(g["weather_tag"] == "clear") & (g["date"] < today) & g["ratio"].notna() & g["valid"]]
            clear_hist = clear_hist.tail(cfg.rolling_clear_days)
            baseline_n = int(len(clear_hist))
            baseline_ratio = float(clear_hist["ratio"].median()) if baseline_n else np.nan
            cold = int(baseline_n < cfg.min_clear_days)
            ratio_today = cur["ratio"]
            dev_today = cur["dev"]
            dev_vs_own = (ratio_today / baseline_ratio - 1) if (not cold and pd.notna(ratio_today) and baseline_ratio) else np.nan
            # 晴好天序列（含今天若为晴好天）用于持续度
            clear_seq = g[(g["weather_tag"] == "clear") & g["dev"].notna()].sort_values("date")
            clear_seq = clear_seq[clear_seq["date"] <= today]
            devs = clear_seq["dev"].to_numpy(dtype=float)
            dates = clear_seq["date"].tolist()
            p5 = self._streak(devs, th.fault_floor)
            p3 = self._streak(devs, th.normal_floor)
            first_breach = dates[len(dates) - p3] if p3 > 0 else None
            roll7 = float(np.median(devs[-7:])) if len(devs) else np.nan
            # 阴天（散射光）偏差：朝向差异在阴天消失 → 用于 C 级"疑朝向"判定；只取纯阴雨天
            oc = g[(g["weather_tag"] == "overcast") & g["dev"].notna() & g["valid"]].tail(5)
            overcast_dev = float(oc["dev"].median()) if len(oc) >= 2 else np.nan
            recent = g[g["date"] > (d - timedelta(days=th.rainy_streak_caution_days)).isoformat()]
            n_clear_recent = int((recent["weather_tag"] == "clear").sum())
            rows.append({
                "string_id": sid, "date": today, "segment_start": seg_start, "group_key": cur["group_key"],
                "group_n": int(cur["group_n"]), "group_level": cur["group_level"],
                "group_median_i": cur["group_median_i"], "ratio_today": ratio_today,
                "baseline_ratio": baseline_ratio, "baseline_n": baseline_n,
                "dev_today": dev_today, "dev_vs_own": dev_vs_own, "dev_roll7": roll7,
                "persist_days_5": int(p5), "persist_days_3": int(p3), "first_breach_date": first_breach,
                "v_dev": self._vertical_dev(g, weather, today), "overcast_dev": overcast_dev,
                "n_clear_recent": n_clear_recent, "cold_start": cold,
                "weather_tag": cur["weather_tag"], "limited": int(cur["limited"]),
            })
        bl = pd.DataFrame(rows)

        # ---- 机间互比（整机问题兜底）
        ibl = self._inverter_baselines(ctx, station_id, d, f, inv_daily, strings)
        return bl, ibl

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _streak(devs: np.ndarray, floor: float) -> int:
        n = 0
        for x in devs[::-1]:
            if np.isnan(x) or x > floor:
                break
            n += 1
        return n

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
