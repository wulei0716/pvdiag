"""阶段6 · 人工复核智能体（数据标注入口）

① 复核队列：AI 判定为"转人工复核"的嫌疑 + 未自动派单的 A/B
② 人工操作：确认 / 驳回(误报，记驳回原因：云影/限发/数据脏/已备案/停机) / 修改原因 → 回写标注库
③ 复核统计：确认率 / 驳回率（按分级、按原因）→ 阈值调优建议（确认率<50% 收紧到 -6%/-7%；漏报高放松到 -4%）
④ 效率：按站周批复核清单（清单式批量复核，一人一天几百串）
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ..models import Action, Cause, Grade, REJECT_REASONS, SuspectStatus
from .base import Agent, AgentContext, AgentResult


class ReviewAgent(Agent):
    name = "review"
    stage = "阶段6 人工复核"

    # ------------------------------------------------------------------ 每日：组队列 + 统计
    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        today = ctx.run_date.isoformat()
        q = ctx.store.query(
            "SELECT s.suspect_id FROM suspects s JOIN verdicts v ON v.suspect_id=s.suspect_id "
            "WHERE s.date=? AND s.status='open' AND v.action=?", (today, Action.REVIEW.value))
        if not q.empty:
            ctx.store.executemany("UPDATE suspects SET status=? WHERE suspect_id=? AND status='open'",
                                  [(SuspectStatus.REVIEWING.value, sid) for sid in q["suspect_id"]])
        stats = self.stats(ctx.store, days=ctx.settings.dispatch.auto_gate_window_days)
        advice = self.threshold_advice(stats, ctx.settings.thresholds.fault_floor)
        pending = self.queue(ctx.store)
        res.metrics = {"queued_today": int(len(q)), "pending_total": int(len(pending)), **stats}
        res.artifacts = {"advice": advice, "pending_by_station": pending.groupby("station_id").size().to_dict() if not pending.empty else {}}
        ctx.shared["review_stats"] = stats
        if advice:
            res.alerts.append(advice)
        res.summary = f"今日入复核队列 {len(q)}，待复核累计 {len(pending)}；近 {ctx.settings.dispatch.auto_gate_window_days} 天确认率 {stats.get('confirm_rate')}"
        return res

    # ------------------------------------------------------------------ 队列
    @staticmethod
    def queue(store, station_id: Optional[str] = None, grades: Optional[List[str]] = None) -> pd.DataFrame:
        sql = ("SELECT s.*, v.cause AS ai_cause, v.confidence, v.action, v.explanation FROM suspects s "
               "LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id WHERE s.status IN ('open','reviewing') "
               "AND s.grade != '已备案'")
        params: List = []
        if station_id:
            sql += " AND s.station_id=?"
            params.append(station_id)
        if grades:
            sql += f" AND s.grade IN ({','.join('?' * len(grades))})"
            params.extend(grades)
        sql += " ORDER BY CASE s.grade WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'C' THEN 2 WHEN 'D' THEN 3 ELSE 4 END, s.dev_today"
        return store.query(sql, params)

    # ------------------------------------------------------------------ 操作
    @staticmethod
    def confirm(store, suspect_id: str, true_cause: Optional[str] = None, reviewer: str = "reviewer", note: str = "") -> None:
        row = store.query("SELECT rule_cause FROM suspects WHERE suspect_id=?", (suspect_id,))
        if row.empty:
            raise KeyError(f"嫌疑 {suspect_id} 不存在")
        cause = true_cause or row["rule_cause"].iloc[0]
        decision = "modify" if (true_cause and true_cause != row["rule_cause"].iloc[0]) else "confirm"
        store.execute("INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                      (suspect_id, reviewer, decision, cause, None, note, datetime.now().isoformat()))
        store.execute("UPDATE suspects SET status=?, rule_cause=? WHERE suspect_id=?", (SuspectStatus.CONFIRMED.value, cause, suspect_id))

    @staticmethod
    def reject(store, suspect_id: str, reason: str = "其他", reviewer: str = "reviewer", note: str = "") -> None:
        if reason not in REJECT_REASONS:
            reason = "其他" if not reason else reason
        store.execute("INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                      (suspect_id, reviewer, "reject", Cause.NORMAL.value, reason, note, datetime.now().isoformat()))
        store.execute("UPDATE suspects SET status=? WHERE suspect_id=?", (SuspectStatus.REJECTED.value, suspect_id))

    # ------------------------------------------------------------------ 统计与调优建议
    @staticmethod
    def stats(store, days: int = 30) -> Dict:
        since = (date.today() - timedelta(days=days)).isoformat()
        rv = store.query(
            "SELECT r.suspect_id, r.decision, r.reject_reason, r.reviewed_at, s.grade, s.rule_cause FROM reviews r JOIN suspects s ON s.suspect_id=r.suspect_id "
            "WHERE r.decision IN ('confirm','modify','reject') AND r.reviewed_at>=?", (since,))
        if rv.empty:
            return {"n_reviewed": 0, "confirm_rate": None, "false_alarm_rate": None}
        # 一个嫌疑可能既有复核员记录又有现场回填记录：按嫌疑去重，以最后一条为准（与训练集口径一致）
        rv = rv.sort_values("reviewed_at").drop_duplicates(subset=["suspect_id"], keep="last")
        n = len(rv)
        confirmed = int(rv["decision"].isin(["confirm", "modify"]).sum())
        by_grade = {g: round(float((x["decision"] != "reject").mean()), 3) for g, x in rv.groupby("grade")}
        by_reason = rv[rv["decision"] == "reject"]["reject_reason"].value_counts().to_dict()
        # 现场漏报：工单回填中 source=field 的（巡检发现但系统未报）
        missed = store.scalar("SELECT COUNT(*) FROM workorders WHERE field_note LIKE '%漏报%' AND created_at>=?", (since,)) or 0
        return {"n_reviewed": n, "confirm_rate": round(confirmed / n, 3), "false_alarm_rate": round(1 - confirmed / n, 3),
                "confirm_rate_by_grade": by_grade, "reject_reasons": by_reason, "field_missed": int(missed)}

    @staticmethod
    def threshold_advice(stats: Dict, current_floor: float) -> str:
        if not stats or not stats.get("n_reviewed") or stats["n_reviewed"] < 20:
            return ""
        b_rate = (stats.get("confirm_rate_by_grade") or {}).get(Grade.B.value)
        if b_rate is not None and b_rate < 0.5:
            return (f"B 级确认率 {b_rate:.0%} < 50%：建议把持续低发阈值从 {current_floor:.0%} 收紧到 "
                    f"{max(current_floor - 0.01, -0.07):.0%}（config.thresholds.fault_floor），并检查驳回原因分布 {stats.get('reject_reasons')}")
        if stats.get("field_missed", 0) >= 3:
            return f"近期现场发现 {stats['field_missed']} 起系统未报的故障：建议把阈值放松到 {min(current_floor + 0.01, -0.04):.0%}"
        return ""

    # ------------------------------------------------------------------ 周批清单
    @staticmethod
    def weekly_batch(store, station_id: str) -> pd.DataFrame:
        """每站每周集中复核：把本周巡检出的嫌疑攒成一张清单（一屏看全：偏差、持续、签名、AI 假设）"""
        q = ReviewAgent.queue(store, station_id)
        if q.empty:
            return q
        cols = ["suspect_id", "string_id", "inverter_id", "grade", "rule_cause", "ai_cause", "confidence", "dev_today",
                "persist_days", "first_date", "date", "action", "explanation"]
        return q[[c for c in cols if c in q.columns]]
