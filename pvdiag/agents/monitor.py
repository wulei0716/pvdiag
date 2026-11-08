"""阶段10 · 闭环自学习与运维监控

① 系统自身监控：拉数成功率 / 数据时延、配额余量、token 失效告警、流水线各智能体状态
② 模型管理：标注累计、重训到期、特征分布漂移（PSI）
③ 指标看板：误报率 / 漏报率 / 发现时延 / 挽回电量电费 / 待修复损失总额 / 自动派单门控状态
"整个系统最大的单一风险不是算法而是数据管道中断"——所有管道问题都落告警。
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from .base import Agent, AgentContext, AgentResult


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index：>0.25 视为显著漂移"""
    if len(expected) < 30 or len(actual) < 30:
        return float("nan")
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected) + 1e-6
    a = np.histogram(actual, qs)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


class MonitorAgent(Agent):
    name = "monitor"
    stage = "阶段10 运维监控"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        today = ctx.run_date.isoformat()
        board: Dict = {"date": today}

        # ---- 管道健康
        runs = ctx.store.query("SELECT agent, status, message FROM pipeline_runs WHERE run_date=? ORDER BY run_id", (today,))
        failed = runs[runs["status"] == "fail"]["agent"].tolist() if not runs.empty else []
        board["pipeline"] = {"agents_run": runs["agent"].tolist() if not runs.empty else [], "failed": failed}
        for a in failed:
            res.alerts.append(f"智能体 {a} 今日运行失败")

        # ---- 数据接入：每站数据时延 / 缺测
        ingest: List[Dict] = []
        for st in ctx.stations():
            dates = ctx.store.available_dates("string", st)
            last = dates[-1] if dates else None
            lag = (ctx.run_date - date.fromisoformat(last)).days if last else None
            sd = ctx.store.station_daily(st, ctx.run_date - timedelta(days=7), ctx.run_date)
            nodata = int((sd["weather_tag"] == "nodata").sum()) if not sd.empty else 7
            row = {"station_id": st, "last_data_date": last, "lag_days": lag, "nodata_days_7d": nodata}
            ingest.append(row)
            if lag is None or lag > 2:
                res.alerts.append(f"{st} 数据时延 {lag} 天（最后数据 {last}）—— 检查拉数任务/授权/配额")
        board["ingest"] = ingest

        # ---- 配额 / token
        plat = ctx.store.query("SELECT key, value_json, updated_at FROM platform_state")
        quota, tokens = {}, {}
        for r in plat.to_dict("records"):
            if ":calls:" in r["key"]:
                quota[r["key"]] = json.loads(r["value_json"])
            if r["key"].endswith(":token"):
                v = json.loads(r["value_json"]) if r["value_json"] else None
                exp = v.get("expires") if isinstance(v, dict) else None
                tokens[r["key"]] = exp
                if exp and datetime.fromisoformat(exp) < datetime.now() + timedelta(hours=6):
                    res.alerts.append(f"{r['key']} 即将/已经失效（{exp}），阳光平台需业主重新授权")
        hq = ctx.settings.platforms.huawei
        for k, used in quota.items():
            if k.endswith(today):
                n_dev = ctx.store.scalar("SELECT COUNT(*) FROM inverters") or 0
                limit = int(np.ceil(max(n_dev, 1) / hq.devices_per_unit)) + hq.base_quota
                if used >= 0.9 * limit:
                    res.alerts.append(f"华为配额今日已用 {used}/{limit}，接近上限")
        board["quota"] = quota
        board["tokens"] = tokens

        # ---- 模型与漂移
        mr = ctx.store.query("SELECT * FROM model_registry WHERE active=1")
        n_labels = int(ctx.store.scalar("SELECT COUNT(DISTINCT suspect_id) FROM reviews WHERE decision IN ('confirm','modify','reject')") or 0)
        model_info = {"active": mr["model_version"].iloc[0] if not mr.empty else "rule-proxy-v1", "n_labels": n_labels,
                      "min_labels_for_ml": ctx.settings.confidence.min_labels_for_ml}
        if not mr.empty:
            trained = datetime.fromisoformat(mr["trained_at"].iloc[0])
            model_info["days_since_trained"] = (datetime.now() - trained).days
            if model_info["days_since_trained"] >= ctx.settings.confidence.retrain_every_days:
                res.alerts.append(f"模型 {model_info['active']} 已 {model_info['days_since_trained']} 天未重训（周期 {ctx.settings.confidence.retrain_every_days} 天）")
        recent = ctx.store.query("SELECT dev_today FROM baselines WHERE date>? AND weather_tag='clear' AND dev_today IS NOT NULL",
                                 ((ctx.run_date - timedelta(days=7)).isoformat(),))["dev_today"].to_numpy(dtype=float)
        ref = ctx.store.query("SELECT dev_today FROM baselines WHERE date BETWEEN ? AND ? AND weather_tag='clear' AND dev_today IS NOT NULL",
                              ((ctx.run_date - timedelta(days=45)).isoformat(), (ctx.run_date - timedelta(days=8)).isoformat()))["dev_today"].to_numpy(dtype=float)
        drift = psi(ref, recent)
        model_info["dev_psi_7d_vs_prev"] = None if np.isnan(drift) else round(drift, 3)
        if not np.isnan(drift) and drift > 0.25:
            res.alerts.append(f"互比偏差分布显著漂移 PSI={drift:.2f}（设备更换/季节/集体衰减？），建议检查基线分段与模型重训")
        board["model"] = model_info

        # ---- KPI
        board["kpi"] = self.kpis(ctx)
        from .workorder import WorkOrderAgent
        ok, why = WorkOrderAgent().auto_gate(ctx)
        board["auto_dispatch_gate"] = {"mode": ctx.settings.dispatch.mode, "passed": ok, "reason": why}
        board["open_alerts"] = int(ctx.store.scalar("SELECT COUNT(*) FROM alerts WHERE resolved=0 AND run_date>=?",
                                                    ((ctx.run_date - timedelta(days=7)).isoformat(),)) or 0)
        res.metrics = {"alerts": len(res.alerts), **{k: v for k, v in board["kpi"].items() if isinstance(v, (int, float))}}
        res.artifacts = board
        ctx.shared["board"] = board
        res.summary = (f"误报率 {board['kpi'].get('false_alarm_rate')}，发现时延 {board['kpi'].get('detection_latency_days')} 天，"
                       f"待修复损失 {board['kpi'].get('open_loss_yuan')} 元，告警 {len(res.alerts)}")
        return res

    # ------------------------------------------------------------------ KPI
    @staticmethod
    def kpis(ctx: AgentContext) -> Dict:
        from .review import ReviewAgent
        from .loss import LossAgent
        st = ReviewAgent.stats(ctx.store, days=ctx.settings.dispatch.auto_gate_window_days)
        # 发现时延：案例首次报警日 − 推定起算日
        susp = ctx.store.query("SELECT suspect_id, first_date FROM suspects WHERE scope='string' AND grade IN ('A','B')")
        lat = []
        for s in susp.to_dict("records"):
            try:
                onset = LossAgent.onset_date(s)
                lat.append((date.fromisoformat(s["first_date"]) - onset).days)
            except Exception:
                pass
        loss = LossAgent.ledger_summary(ctx.store)["totals"]
        # 挽回：已验证工单 的 日损失 × 自验证以来天数（近似）
        ver = ctx.store.query("SELECT loss_kwh_per_day, closed_at FROM workorders WHERE status='verified'")
        recovered = 0.0
        for r in ver.to_dict("records"):
            days = max(0, (ctx.run_date - date.fromisoformat(r["closed_at"][:10])).days)
            recovered += (r["loss_kwh_per_day"] or 0) * days
        price = ctx.settings.loss.tou_price.get("flat", ctx.settings.loss.default_price_yuan_per_kwh)
        return {
            "reviewed_30d": st.get("n_reviewed", 0), "false_alarm_rate": st.get("false_alarm_rate"),
            "confirm_rate_by_grade": st.get("confirm_rate_by_grade"), "field_missed": st.get("field_missed", 0),
            "detection_latency_days": round(float(np.mean(lat)), 1) if lat else None,
            "open_cases": int(ctx.store.scalar("SELECT COUNT(*) FROM suspects WHERE status IN ('open','reviewing','confirmed','dispatched') AND grade IN ('A','B')") or 0),
            "open_loss_kwh": round(loss["lost_kwh"], 1), "open_loss_yuan": round(loss["lost_yuan"], 1),
            "recovered_kwh": round(recovered, 1), "recovered_yuan": round(recovered * price, 1),
            "workorders": ctx.store.query("SELECT status, COUNT(*) n FROM workorders GROUP BY status").set_index("status")["n"].to_dict(),
        }
