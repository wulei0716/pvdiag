"""每日闭环流水线编排：接入 → 档案校验 → 基线 → 规则诊断 → AI 判定 → 损失 → 复核队列 → 工单 → 区域建议 → 监控 → 报告

- 每个智能体独立留痕（pipeline_runs），单点失败不拖垮全链路但必告警
- backfill(start, end)：历史回放（逐日跑基线/诊断，构建案例与台账）
- 每日报告：JSON（供下游消费）+ Markdown（人读）
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .agents.base import Agent, AgentContext, AgentResult
from .agents.baseline import BaselineAgent
from .agents.classify import ClassifyAgent
from .agents.detect import DetectAgent
from .agents.ingest import IngestAgent
from .agents.loss import LossAgent
from .agents.monitor import MonitorAgent
from .agents.profile import ProfileAgent
from .agents.region import RegionAgent
from .agents.review import ReviewAgent
from .agents.workorder import WorkOrderAgent
from .config import Settings
from .storage import Store

log = logging.getLogger("pvdiag")


class DailyPipeline:
    def __init__(self, settings: Settings, store: Optional[Store] = None, ingest: bool = True, backfill_days: int = 0):
        self.settings = settings
        settings.ensure_dirs()
        self.store = store or Store(settings.db_path, settings.samples_dir)
        self.ingest = ingest
        self.backfill_days = backfill_days

    def agents(self, with_ingest: bool = True, light: bool = False) -> List[Agent]:
        chain: List[Agent] = []
        if with_ingest:
            chain.append(IngestAgent(backfill_days=self.backfill_days))
        chain += [ProfileAgent(), BaselineAgent(), DetectAgent(), ClassifyAgent(), LossAgent()]
        if not light:
            chain += [ReviewAgent(), WorkOrderAgent(), RegionAgent(), MonitorAgent()]
        return chain

    # ------------------------------------------------------------------
    def run(self, run_date: Optional[date] = None, station_ids: Optional[List[str]] = None, write_report: bool = True) -> Dict:
        run_date = run_date or date.today()
        ctx = AgentContext(settings=self.settings, store=self.store, run_date=run_date, station_ids=station_ids)
        results: List[AgentResult] = []
        for agent in self.agents(with_ingest=self.ingest):
            results.append(agent(ctx))
        # 月度重训钩子
        retrain = self._maybe_retrain(ctx)
        report = self.build_report(ctx, results, retrain)
        if write_report:
            self.write_report(report, run_date)
        return report

    def backfill(self, start: date, end: date, station_ids: Optional[List[str]] = None) -> List[Dict]:
        """历史回放：先一次性接入，再逐日跑轻量链路（基线/诊断/判定/损失），最后一天跑完整链路"""
        out = []
        if self.ingest:
            ctx = AgentContext(settings=self.settings, store=self.store, run_date=end, station_ids=station_ids)
            IngestAgent(backfill_days=(end - start).days)(ctx)
        d = start
        while d <= end:
            ctx = AgentContext(settings=self.settings, store=self.store, run_date=d, station_ids=station_ids)
            light = d < end
            for agent in self.agents(with_ingest=False, light=light):
                agent(ctx)
            if not light:
                rep = self.build_report(ctx, list(ctx.shared.values()), self._maybe_retrain(ctx))
                self.write_report(rep, d)
                out.append(rep)
            d += timedelta(days=1)
        return out

    # ------------------------------------------------------------------
    def _maybe_retrain(self, ctx: AgentContext) -> Dict:
        last = ctx.store.query("SELECT trained_at FROM model_registry WHERE active=1")
        due = last.empty or (datetime.now() - datetime.fromisoformat(last["trained_at"].iloc[0])).days >= ctx.settings.confidence.retrain_every_days
        if not due:
            return {"trained": False, "reason": "未到重训周期"}
        try:
            return ClassifyAgent().train(ctx)
        except Exception as e:
            ctx.store.alert(ctx.run_date, "warn", "classify", f"重训失败: {e}")
            return {"trained": False, "reason": str(e)}

    # ------------------------------------------------------------------ 报告
    def build_report(self, ctx: AgentContext, results: List[AgentResult], retrain: Dict) -> Dict:
        res_by = {r.agent: r for r in results if isinstance(r, AgentResult)}
        today = ctx.run_date.isoformat()
        weather = ctx.store.query("SELECT station_id, weather_tag, noon_power, peak_ref FROM station_daily WHERE date=?", (today,))
        suspects = ctx.store.query(
            "SELECT s.suspect_id, s.station_id, s.inverter_id, s.string_id, s.scope, s.grade, s.rule_cause, s.dev_today, s.persist_days, "
            "s.first_date, s.status, s.low_confidence, v.cause ai_cause, v.confidence, v.action, v.explanation "
            "FROM suspects s LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id WHERE s.date=? AND s.status NOT IN ('rejected','closed') "
            "ORDER BY CASE s.grade WHEN 'A' THEN 0 WHEN 'B' THEN 1 WHEN 'C' THEN 2 WHEN 'D' THEN 3 ELSE 4 END, s.dev_today", (today,))
        wos = ctx.store.query("SELECT wo_id, station_id, roof_area, cause, priority, loss_kwh_per_day, loss_yuan_total, status FROM workorders "
                              "WHERE status IN ('draft','sent','reopened') ORDER BY priority DESC")
        alerts = ctx.store.query("SELECT level, source, message FROM alerts WHERE run_date=? ORDER BY alert_id", (today,))
        clean = lambda df: df.astype(object).where(pd.notna(df), None).to_dict("records") if not df.empty else []
        return {
            "date": today,
            "weather": clean(weather),
            "agents": {k: {"status": v.status, "summary": v.summary, "metrics": v.metrics} for k, v in res_by.items()},
            "suspects": clean(suspects),
            "workorders": clean(wos),
            "loss": ctx.shared.get("loss_summary", {}),
            "review": {"stats": ctx.shared.get("review_stats", {}), "advice": (res_by.get("review").artifacts.get("advice") if res_by.get("review") else "")},
            "region": ctx.shared.get("region_artifacts", {}),
            "board": ctx.shared.get("board", {}),
            "retrain": retrain,
            "alerts": clean(alerts),
        }

    def write_report(self, report: Dict, run_date: date) -> Path:
        folder = self.settings.reports_dir / run_date.isoformat()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        (folder / "report.md").write_text(render_markdown(report), encoding="utf-8")
        return folder


def _json_default(o):
    if isinstance(o, (date, datetime)):
        return o.isoformat()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def _pct(x) -> str:
    try:
        return f"{float(x):+.1%}" if x is not None and not pd.isna(x) else "-"
    except Exception:
        return "-"


def _obj(s: Dict) -> str:
    sid = s.get("string_id")
    if isinstance(sid, str) and sid:
        return sid
    return f"{s.get('inverter_id')}(整机)"


def render_markdown(r: Dict) -> str:
    L: List[str] = [f"# 组串发电量异常诊断 · 每日巡检报告 {r['date']}", ""]
    if r.get("weather"):
        L.append("## 天气 / 晴好天判定")
        for w in r["weather"]:
            L.append(f"- {w['station_id']}: **{w['weather_tag']}** 中午功率 {w['noon_power'] or 0:.0f} kW / 晴天峰值参考 {w['peak_ref'] or 0:.0f} kW")
        L.append("")
    L.append("## 智能体运行")
    for k, v in r.get("agents", {}).items():
        flag = "✅" if v["status"] == "ok" else ("⚠️" if v["status"] in ("warn", "skipped") else "❌")
        L.append(f"- {flag} `{k}` {v['summary']}")
    L.append("")
    L.append(f"## 嫌疑清单（{len(r.get('suspects', []))} 条）")
    if r.get("suspects"):
        L.append("| 分级 | 对象 | 偏差 | 持续 | 规则原因 | AI 原因 | 置信 | 动作 | 状态 |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s in r["suspects"]:
            obj = _obj(s)
            conf = f"{s['confidence']:.0%}" if s.get("confidence") is not None and not pd.isna(s["confidence"]) else "-"
            L.append(f"| {s['grade']} | {obj} | {_pct(s['dev_today'])} | {s['persist_days']}d | {s['rule_cause']} | {s.get('ai_cause') or '-'} | {conf} | {s.get('action') or '-'} | {s['status']} |")
        L.append("")
        L.append("<details><summary>判定依据（为什么）</summary>\n")
        for s in r["suspects"]:
            if s.get("explanation"):
                L.append(f"- **{_obj(s)}**：{s['explanation']}")
        L.append("\n</details>\n")
    loss = r.get("loss", {})
    if loss:
        t = loss.get("totals", {})
        L.append(f"## 损失台账（钱在漏）：累计未修复 **{t.get('lost_kwh', 0):.0f} kWh ≈ {t.get('lost_yuan', 0):.0f} 元**")
        for st, v in (loss.get("by_station") or {}).items():
            L.append(f"- {st}: {v['lost_kwh']:.0f} kWh / {v['lost_yuan']:.0f} 元")
        L.append("")
    if r.get("workorders"):
        L.append("## 工单（草稿需人工一键确认发送）")
        L.append("| 工单 | 电站 | 区域 | 原因 | 优先级 | 损失/天 | 累计损失 | 状态 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for w in r["workorders"]:
            L.append(f"| {w['wo_id']} | {w['station_id']} | {w['roof_area']} | {w['cause']} | {w['priority']:.1f} | {w['loss_kwh_per_day']:.1f} kWh | {w['loss_yuan_total']:.0f} 元 | {w['status']} |")
        L.append("")
    reg = r.get("region", {})
    if reg.get("packages"):
        L.append("## 区域任务包（该去哪修什么）")
        for p in reg["packages"][:10]:
            L.append(f"- {p['station_id']} / {p['roof_area']}：{p['n_strings']} 串 {p['causes']}，损失 {p['loss_kwh_per_day']:.1f} kWh/天，优先级 {p['priority']}")
        L.append("")
    if reg.get("dust_candidates"):
        L.append("## 清洗建议（积灰签名）")
        for d in reg["dust_candidates"][:10]:
            L.append(f"- {d['string_id']}：{d['pattern']}，当前 {_pct(d['dev_now'])}，斜率 {d['slope_per_clear_day'] * 100:+.2f}%/晴天 → {d['advice']}")
        L.append("")
    rv = r.get("review", {})
    if rv.get("stats"):
        s = rv["stats"]
        L.append(f"## 复核统计：近 30 天复核 {s.get('n_reviewed', 0)} 条，确认率 {s.get('confirm_rate')}，驳回原因 {s.get('reject_reasons')}")
        if rv.get("advice"):
            L.append(f"> 阈值调优建议：{rv['advice']}")
        L.append("")
    b = r.get("board", {})
    if b:
        k = b.get("kpi", {})
        L.append("## 运营看板")
        L.append(f"- 误报率 {k.get('false_alarm_rate')} · 现场漏报 {k.get('field_missed')} · 发现时延 {k.get('detection_latency_days')} 天 · 在办案例 {k.get('open_cases')}")
        L.append(f"- 挽回电量 {k.get('recovered_kwh')} kWh ≈ {k.get('recovered_yuan')} 元 · 待修复损失 {k.get('open_loss_yuan')} 元")
        L.append(f"- 模型：{b.get('model', {}).get('active')}，标注 {b.get('model', {}).get('n_labels')} 条，偏差分布 PSI {b.get('model', {}).get('dev_psi_7d_vs_prev')}")
        g = b.get("auto_dispatch_gate", {})
        L.append(f"- 自动派单门控：mode={g.get('mode')} passed={g.get('passed')}（{g.get('reason')}）")
        L.append("")
    if r.get("alerts"):
        L.append("## 告警")
        for a in r["alerts"]:
            L.append(f"- [{a['level']}] {a['source']}: {a['message']}")
    return "\n".join(L)
