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
                res.alerts.append(f"{t['station_id']} 站级（{t['basis']}）近 {t['days']} 天斜率 {t['slope_pct_per_day']:+.2f}%/天，积灰损失约 {t['soiling_loss_pct']}%"
                                  f"（≈{t['lost_kwh_per_day']} kWh/天），{t['advice']}")
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
                        "priority": round(per_day * price * ctx.settings.loss.expected_fault_days_default / max(cost, 1.0), 2)})   # 朝向差异修复成本 0 → 防除零（压测发现）
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
        """站级积灰/衰减量化（互比盲区：全站一起脏、一起衰减，互比看不出）

        指标口径：
        - 有辐照：PR_d = 日电量 / (日辐照 × 装机)；无辐照：用 中午功率/晴天峰值参考 近似（受季节日照影响，需去趋势）
        - 清洁参考 = 最近一次降雨后头 2 个晴好天的中位（雨把灰冲掉），没有雨就用窗口内最高的 3 天
        - 当前水平 = 最近 3 个晴好天中位 → 积灰损失% = 1 − 当前/清洁参考
        - 日损电量 ≈ 积灰损失% × 近 7 个晴好天日均电量；清洗成本 = 装机 × cleaning_cost_per_kwp；回本天数 = 成本 / (日损 × 电价)
        """
        start = (ctx.run_date - timedelta(days=window_days)).isoformat()
        out = []
        price = ctx.settings.loss.tou_price.get("flat", ctx.settings.loss.default_price_yuan_per_kwh)
        for st in ctx.stations():
            sd_all = ctx.store.station_daily(st, start, ctx.run_date)
            sd = sd_all[(sd_all["weather_tag"] == "clear") & sd_all["noon_power"].notna() & sd_all["peak_ref"].notna()].copy()
            if len(sd) < 10:
                continue
            cap = ctx.store.scalar("SELECT capacity_kwp FROM stations WHERE station_id=?", (st,))
            # 站级日电量（全部组串 I·V 积分）
            e = ctx.store.query("SELECT f.date, SUM(f.day_energy_kwh) e FROM daily_features f JOIN strings s ON s.string_id=f.string_id "
                                "WHERE s.station_id=? AND f.date BETWEEN ? AND ? GROUP BY f.date", (st, start, ctx.run_date.isoformat())).set_index("date")["e"]
            sd["energy"] = sd["date"].map(e)
            use_irr = "irradiance_kwh_m2" in sd.columns and sd["irradiance_kwh_m2"].notna().sum() >= 10 and cap
            if use_irr:
                sd["level"] = sd["energy"] / (sd["irradiance_kwh_m2"] * float(cap))        # PR
                basis = "PR（电量/辐照/装机）"
            else:
                sd["level"] = sd["noon_power"] / sd["peak_ref"]
                basis = "中午功率/晴天峰值参考（无辐照，含季节项）"
            sd = sd[sd["level"].notna() & np.isfinite(sd["level"])]
            if len(sd) < 10:
                continue
            y = sd["level"].to_numpy(dtype=float)
            slope = float(np.polyfit(np.arange(len(y)), y, 1)[0]) * 100
            rain = sd_all[sd_all["rain"] == 1]["date"].tolist()
            rain_days = len(rain)
            if rain:
                after = sd[sd["date"] > rain[-1]].head(2)
                clean_ref = float(after["level"].median()) if len(after) >= 1 else float(np.percentile(y, 90))
                ref_note = f"最近降雨 {rain[-1]} 后首个晴好天"
            else:
                clean_ref = float(np.percentile(y, 90)); ref_note = "窗口内 90 分位（无降雨记录；晴好天离散 3%~5% 会让它偏高）"
            current = float(sd["level"].tail(3).median())
            soiling = max(0.0, 1 - current / clean_ref) if clean_ref > 0 else 0.0
            daily_kwh = float(sd["energy"].tail(7).mean()) if sd["energy"].notna().any() else 0.0
            lost_kwh_day = soiling * daily_kwh / max(1 - soiling, 0.5)
            cost = float(cap or 0) * ctx.settings.loss.cleaning_cost_per_kwp
            payback = cost / (lost_kwh_day * price) if lost_kwh_day * price > 0 else None
            out.append({"station_id": st, "days": int(len(sd)), "basis": basis, "slope_pct_per_day": round(slope, 3), "rain_days": rain_days,
                        "clean_ref": round(clean_ref, 4), "clean_ref_note": ref_note, "current": round(current, 4),
                        "soiling_loss_pct": round(soiling * 100, 2), "lost_kwh_per_day": round(lost_kwh_day, 1),
                        "lost_yuan_per_day": round(lost_kwh_day * price, 1), "cleaning_cost_yuan": round(cost, 0),
                        "payback_days": None if payback is None else round(payback, 1),
                        # 无辐照时季节项会混进斜率与"清洁参考"，只按老口径（斜率+无雨）触发；有 PR 才允许按积灰损失触发
                        "suspect_soiling": bool(slope < -0.25 and rain_days <= 2) or (bool(use_irr) and soiling >= 0.05),
                        "advice": ("建议全站清洗：回本约 %.0f 天" % payback) if payback is not None and payback <= 45 and soiling >= 0.03
                                  else ("积灰 %.1f%%，暂不划算清洗（回本 %s 天）" % (soiling * 100, "∞" if payback is None else round(payback))) if soiling > 0
                                  else "无积灰迹象",
                        "note": "无辐照时季节性日照变化会混入斜率；有气象站/卫星辐照时自动改用 PR"})
        return out
