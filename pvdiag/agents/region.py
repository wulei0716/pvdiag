"""阶段9 · 区域检修/清洗建议（针对性提效）

① 地理聚合：按屋面/区域聚类异常串 → 区域任务包（一次上人处理一片），按 损失×持续×修复成本 排序
② 清洗建议：灰尘累积签名 = 慢渐变负偏差 + 雨后部分恢复（故障是阶跃或持续恒定，动作完全不同）
③ 清洗/检修效果验证：事件后偏差归零检测 + 增益统计（并防"清洗超发"被误判——基线分段重置联动）
注：全站均匀积灰互比看不出，需竖向（辐照）或站间比兜底，这里以站级中午功率/峰值参考趋势给出提示。
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ..models import Grade, SuspectStatus
from .base import Agent, AgentContext, AgentResult


class RegionAgent(Agent):
    name = "region"
    stage = "阶段9 区域检修/清洗建议"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        packages = self.region_packages(ctx)
        dust = self.dust_candidates(ctx)
        effects = self.cleaning_effects(ctx)
        station_trend = self.station_soiling_trend(ctx)
        res.metrics = {"packages": len(packages), "dust_candidates": len(dust), "cleaning_events_verified": len(effects)}
        res.artifacts = {"packages": packages, "dust_candidates": dust, "cleaning_effects": effects, "station_trend": station_trend}
        ctx.shared["region_artifacts"] = res.artifacts
        for t in station_trend:
            if t.get("suspect_soiling"):
                res.alerts.append(f"{t['station_id']} 站级产出近 {t['days']} 天缓慢下滑 {t['slope_pct_per_day']:+.2f}%/天，疑全站积灰（互比看不出），建议安排清洗评估")
        res.summary = f"区域任务包 {len(packages)}，积灰疑点 {len(dust)}，清洗效果验证 {len(effects)}"
        return res

    # ------------------------------------------------------------------ 区域任务包
    def region_packages(self, ctx: AgentContext) -> List[Dict]:
        susp = ctx.store.query(
            "SELECT s.*, v.cause ai_cause, v.confidence FROM suspects s LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id "
            "WHERE s.status IN ('open','reviewing','confirmed','dispatched') AND s.grade IN ('A','B') AND s.scope='string'")
        if susp.empty:
            return []
        prof = ctx.store.strings().set_index("string_id")
        susp["roof_area"] = susp["string_id"].map(prof["roof_area"]).fillna("未知区域")
        loss = ctx.store.query("SELECT suspect_id, SUM(lost_kwh) k, SUM(lost_yuan) y, COUNT(*) n FROM loss_ledger GROUP BY suspect_id").set_index("suspect_id")
        price = ctx.settings.loss.tou_price.get("flat", ctx.settings.loss.default_price_yuan_per_kwh)
        out = []
        for (st, roof), g in susp.groupby(["station_id", "roof_area"]):
            k = float(loss.reindex(g["suspect_id"])["k"].fillna(0).sum())
            y = float(loss.reindex(g["suspect_id"])["y"].fillna(0).sum())
            days = float(loss.reindex(g["suspect_id"])["n"].fillna(0).max() or 1)
            per_day = k / days if days else 0.0
            cost = max(ctx.settings.loss.repair_cost_by_cause.get(c, 300.0) for c in g["rule_cause"]) + 200.0 * (len(g) - 1)
            out.append({"station_id": st, "roof_area": roof, "n_strings": int(len(g)),
                        "strings": g["string_id"].tolist(), "causes": g["rule_cause"].value_counts().to_dict(),
                        "loss_kwh_total": round(k, 1), "loss_yuan_total": round(y, 1), "loss_kwh_per_day": round(per_day, 2),
                        "priority": round(per_day * price * ctx.settings.loss.expected_fault_days_default / cost, 2)})
        return sorted(out, key=lambda r: -r["priority"])

    # ------------------------------------------------------------------ 积灰签名
    def dust_candidates(self, ctx: AgentContext, window_days: int = 30) -> List[Dict]:
        """单串：近 N 个晴好天偏差呈慢渐变下滑（斜率 ≤ -0.1%/天）且雨后偏差回升 ≥ 1 个百分点"""
        start = (ctx.run_date - timedelta(days=window_days)).isoformat()
        bl = ctx.store.query(
            "SELECT b.string_id, b.date, b.dev_today, b.weather_tag, s.station_id FROM baselines b JOIN strings s ON s.string_id=b.string_id "
            "WHERE b.date BETWEEN ? AND ? AND b.dev_today IS NOT NULL ORDER BY b.date", (start, ctx.run_date.isoformat()))
        if bl.empty:
            return []
        # 无气象数据时以"阴雨天（overcast）"作为候选降雨日：比较其前后晴好天的偏差变化
        rain = ctx.store.query("SELECT station_id, date FROM station_daily WHERE (rain=1 OR weather_tag='overcast') AND date BETWEEN ? AND ?",
                               (start, ctx.run_date.isoformat()))
        rain_by_station = rain.groupby("station_id")["date"].apply(list).to_dict() if not rain.empty else {}
        out = []
        for sid, g in bl.groupby("string_id"):
            clear = g[g["weather_tag"] == "clear"]
            if len(clear) < 8:
                continue
            devs = clear["dev_today"].to_numpy(dtype=float)
            cur = float(devs[-1])
            if cur > -0.03 or cur < -0.30:          # 太深的偏差是故障量级，不是积灰
                continue
            # 阶跃排除：相邻晴好天之间出现 ≥2 个百分点的"向下阶跃" → 故障（断线/二极管/遮挡），不是积灰
            # 相邻晴好天之间可能隔着几个阴雨天，积灰会连续累积：按"每日历日"的变化率判阶跃
            gaps = np.diff(pd.to_datetime(clear["date"]).to_numpy()).astype("timedelta64[D]").astype(float)
            diffs = np.diff(devs) / np.maximum(gaps, 1.0)
            if len(diffs) and float(np.min(diffs)) <= -0.02:
                continue
            x = np.arange(len(clear))
            slope = float(np.polyfit(x, devs, 1)[0])   # 每个晴好天的偏差变化
            if slope > -0.001 or slope < -0.012:       # 慢渐变：-0.1% ~ -1.2% / 晴好天
                continue
            # 雨后恢复：比较雨前最后一个晴天与雨后首个晴天
            recov = []
            for rd in rain_by_station.get(g["station_id"].iloc[0], []):
                before = clear[clear["date"] < rd].tail(1)
                after = clear[clear["date"] > rd].head(1)
                if not before.empty and not after.empty:
                    recov.append(float(after["dev_today"].iloc[0] - before["dev_today"].iloc[0]))
            rain_recovery = max(recov) if recov else None
            kind = "积灰（慢渐变 + 雨后恢复）" if (rain_recovery is not None and rain_recovery >= 0.01) else "慢渐变低发（积灰或衰减，雨后未见恢复）"
            out.append({"string_id": sid, "station_id": g["station_id"].iloc[0], "dev_now": round(cur, 4),
                        "slope_per_clear_day": round(slope, 5), "rain_recovery": None if rain_recovery is None else round(rain_recovery, 4),
                        "pattern": kind, "n_clear_days": int(len(clear)),
                        "advice": "单串/局部清洗并登记清洗事件；清洗后系统自动验证增益" if "积灰（" in kind else "先清洗验证，若无恢复查组件衰减/接触电阻"})
        return sorted(out, key=lambda r: r["dev_now"])

    # ------------------------------------------------------------------ 清洗效果验证
    def cleaning_effects(self, ctx: AgentContext) -> List[Dict]:
        ev = ctx.store.query("SELECT * FROM events WHERE event_type IN ('cleaning','repair','module_replace','rewire') ORDER BY event_date")
        out = []
        for e in ev.to_dict("records"):
            if e["scope"] != "string":
                continue
            d0 = e["event_date"]
            before = ctx.store.query("SELECT dev_today FROM baselines WHERE string_id=? AND date<? AND weather_tag='clear' ORDER BY date DESC LIMIT 5",
                                     (e["target_id"], d0))
            after = ctx.store.query("SELECT dev_today FROM baselines WHERE string_id=? AND date>=? AND weather_tag='clear' ORDER BY date LIMIT 5",
                                    (e["target_id"], d0))
            if len(before) < 2 or len(after) < 2:
                continue
            b, a = float(before["dev_today"].median()), float(after["dev_today"].median())
            out.append({"string_id": e["target_id"], "event": e["event_type"], "date": d0,
                        "dev_before": round(b, 4), "dev_after": round(a, 4), "gain_pct": round((a - b) * 100, 2),
                        "verdict": "偏差归零" if a > ctx.settings.thresholds.normal_floor else "仍低发"})
        return out

    # ------------------------------------------------------------------ 站级积灰趋势（互比盲区提示）
    def station_soiling_trend(self, ctx: AgentContext, window_days: int = 45) -> List[Dict]:
        start = (ctx.run_date - timedelta(days=window_days)).isoformat()
        out = []
        for st in ctx.stations():
            sd = ctx.store.station_daily(st, start, ctx.run_date)
            sd = sd[(sd["weather_tag"] == "clear") & sd["noon_power"].notna() & sd["peak_ref"].notna()]
            if len(sd) < 10:
                continue
            # 以季节性去趋势：用一阶多项式拟合 noon_power/peak_ref 相对变化；斜率明显为负 + 无雨天则疑积灰
            y = (sd["noon_power"] / sd["peak_ref"]).to_numpy(dtype=float)
            x = np.arange(len(y))
            slope = float(np.polyfit(x, y, 1)[0]) * 100
            rain_days = int(ctx.store.scalar("SELECT COUNT(*) FROM station_daily WHERE station_id=? AND rain=1 AND date BETWEEN ? AND ?",
                                             (st, start, ctx.run_date.isoformat())) or 0)
            out.append({"station_id": st, "days": int(len(sd)), "slope_pct_per_day": round(slope, 3), "rain_days": rain_days,
                        "suspect_soiling": bool(slope < -0.25 and rain_days <= 2),
                        "note": "季节性日照下降会贡献负斜率，建议结合辐照数据/站间互比确认"})
        return out
