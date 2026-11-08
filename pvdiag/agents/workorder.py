"""阶段7 · 工单派发与现场闭环

① 确认异常自动生成工单：串位置（屋面/pv 编号）、症状、原因假设、按原因模板化的检查项、损失量级、优先级
② 派发渠道：dryrun（写 outbox markdown）/ 企业微信群机器人 webhook；初期 draft 模式 = 生成草稿 + 人工一键确认发送
③ 现场结果回填：实际原因、处理动作（维修/清洗/更换）、照片 → 自动登记事件（驱动基线分段重置）
④ 自动复查：处理后 N 个晴好天偏差是否归零，未归零自动重开工单
优先级公式：损失电量/天 × 电价 × 预计持续天数 ÷ 修复成本；同屋面多串异常合并为区域工单。
"""
from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..models import Action, Cause, EventType, SuspectStatus, WorkOrderStatus
from .base import Agent, AgentContext, AgentResult
from .loss import LossAgent
from .profile import ProfileAgent

CHECKLIST: Dict[str, List[str]] = {
    Cause.OPEN_CIRCUIT.value: ["核对逆变器直流侧该路 MC4 接头/保险丝/直流开关", "测开路电压 Voc 与串长理论值比对", "沿线缆查断点、烧蚀、鼠咬", "恢复后在平台确认电流回升"],
    Cause.BYPASS_DIODE.value: ["按凹陷时段到现场观察阴影落点（时间点见工单）", "红外测温查热斑组件/接线盒", "测故障组件 Voc 判旁路二极管短路", "更换组件或接线盒，拍照回填"],
    Cause.SHADING.value: ["按凹陷时段现场比对阴影源（新增设备/广告牌/植被/积尘带）", "拍照并与档案排布图比对", "可清除遮挡则清除；不可清除则登记备案（备案预期偏差）"],
    Cause.DEGRADATION_DUST.value: ["现场查该串积灰/鸟粪/局部污渍情况并拍照", "必要时单串清洗，清洗后登记清洗事件", "若清洗后仍低发，查组件衰减（EL/IV 曲线）与接触电阻"],
    Cause.MPPT.value: ["查逆变器告警日志、MPPT 电压/电流", "对比各 MPPT 路输入功率，判 MPPT 失效或限功率设置", "联系厂家/检查固件与限发参数"],
    Cause.DATA_ISSUE.value: ["检查采集器/通讯链路，核对平台时钟", "重启采集模块后观察电流是否恢复变化", "若现场发电正常则关闭为数据问题"],
    Cause.PROFILE_ERROR.value: ["核对排布图：该串实际串长/MPPT 接线", "更正档案（n_modules/mppt），重算基线"],
    Cause.CURTAILMENT.value: ["核对电网调度/逆变器限功率设置", "确认是否为 EMC 协议允许的限发，不计组件故障"],
    Cause.OUTAGE.value: ["查逆变器/采集器告警与离线日志，核对停机时段是否与工单一致", "查直流开关/保险丝是否跳脱、通讯链路是否中断",
                         "若为电网/调度停机或计划检修 → 登记事件并驳回；否则排查设备并回填实际原因"],
    Cause.ORIENTATION.value: ["核对排布图：该串朝向/倾角是否与档案一致", "更正档案后系统自动重分组，不派现场"],
    Cause.UNKNOWN.value: ["现场巡检该串组件、接线、阴影", "拍照回填，记录实际原因"],
}


class DispatchChannel:
    name = "dryrun"

    def send(self, wo: Dict, text: str) -> str:
        return "dryrun"


class OutboxChannel(DispatchChannel):
    name = "dryrun"

    def __init__(self, outbox: Path):
        self.outbox = outbox
        outbox.mkdir(parents=True, exist_ok=True)

    def send(self, wo: Dict, text: str) -> str:
        p = self.outbox / f"{wo['wo_id']}.md"
        p.write_text(text, encoding="utf-8")
        return f"written:{p}"


class WeComChannel(DispatchChannel):
    name = "wecom"

    def __init__(self, webhook: str):
        self.webhook = webhook

    def send(self, wo: Dict, text: str) -> str:
        import requests
        r = requests.post(self.webhook, json={"msgtype": "markdown", "markdown": {"content": text[:4000]}}, timeout=20)
        r.raise_for_status()
        body = r.json()
        if body.get("errcode") not in (0, None):
            raise RuntimeError(f"企业微信发送失败: {body}")
        return "wecom:ok"


class WorkOrderAgent(Agent):
    name = "workorder"
    stage = "阶段7 工单闭环"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        created = self.generate(ctx)
        sent = 0
        effective, gate = self.effective_mode(ctx)
        for t in gate.get("transitions", []):
            res.alerts.append(t)
        if effective == "auto":
            for wo_id in created:
                self.send(ctx, wo_id)
                sent += 1
        elif ctx.settings.dispatch.mode in ("auto", "gated") and created:
            res.alerts.append(f"派单门控未放开（{gate.get('reason')}），{len(created)} 张新工单保持草稿待人工确认")
        rechecked = self.recheck(ctx)
        res.metrics = {"created": len(created), "sent": sent, "rechecked": rechecked, "effective_mode": effective,
                       "drafts_pending": int(ctx.store.scalar("SELECT COUNT(*) FROM workorders WHERE status='draft'") or 0)}
        res.artifacts = {"created": created, "gate": gate}
        ctx.shared["dispatch_gate"] = gate
        res.summary = f"新建工单 {len(created)}（已发送 {sent}，模式 {effective}），待确认草稿 {res.metrics['drafts_pending']}，复查 {rechecked}"
        return res

    # ------------------------------------------------------------------ 门控自动开合（带滞回）
    def effective_mode(self, ctx: AgentContext) -> Tuple[str, Dict]:
        """draft → 永远草稿；auto → 门控当天通过就自动；gated → 连续 N 个运行日通过才放开，任一恶化条件立即收回。
        状态存 platform_state['dispatch:gate']，每次开合都留痕告警。"""
        d = ctx.settings.dispatch
        gate_ok, why = self.auto_gate(ctx)
        healthy, hwhy = self.gate_health(ctx)
        state = ctx.store.get_state("dispatch:gate") or {"effective": "draft", "streak": 0, "since": None, "history": []}
        transitions: List[str] = []
        if d.mode == "draft":
            effective = "draft"
            state.update({"effective": "draft", "streak": 0, "reason": "配置 mode=draft"})
        elif d.mode == "auto":
            effective = "auto" if (gate_ok and healthy) else "draft"
            state.update({"effective": effective, "reason": why if not gate_ok else (hwhy if not healthy else "ok")})
        else:  # gated
            prev = state.get("effective", "draft")
            if gate_ok and healthy:
                if state.get("date") != ctx.run_date.isoformat():          # 同一天重跑不重复累计
                    state["streak"] = int(state.get("streak", 0)) + 1
                if prev != "auto" and state["streak"] >= d.gate_open_streak_days:
                    state.update({"effective": "auto", "since": ctx.run_date.isoformat()})
                    transitions.append(f"派单门控放开：连续 {state['streak']} 个运行日误报率达标（{why}），自动派单开启")
                state["reason"] = "ok" if state["effective"] == "auto" else f"门控通过，连续 {state['streak']}/{d.gate_open_streak_days} 天后放开"
            else:
                reason = why if not gate_ok else hwhy
                if prev == "auto":
                    transitions.append(f"派单门控收回：{reason}，回到草稿模式（需人工确认发送）")
                state.update({"effective": "draft", "streak": 0, "reason": reason})
            effective = state["effective"]
        state["date"] = ctx.run_date.isoformat()
        state["gate_ok"], state["healthy"] = bool(gate_ok), bool(healthy)
        hist = state.get("history", [])
        for t in transitions:
            hist.append({"date": ctx.run_date.isoformat(), "event": t})
        state["history"] = hist[-20:]
        state["transitions"] = transitions
        ctx.store.set_state("dispatch:gate", {k: v for k, v in state.items() if k != "transitions"})
        return effective, state

    def gate_health(self, ctx: AgentContext) -> Tuple[bool, str]:
        """数据管道不健康时不自动派人：当天智能体失败 / 数据时延 / 现场漏报多"""
        d = ctx.settings.dispatch
        today = ctx.run_date.isoformat()
        if d.gate_close_on_pipeline_fail:
            since = ctx.shared.get("pipeline_started_at") or "0000"
            failed = ctx.store.query("SELECT DISTINCT agent FROM pipeline_runs WHERE run_date=? AND status='fail' AND started_at>=?", (today, since))
            if not failed.empty:
                return False, f"今日智能体失败：{failed['agent'].tolist()}"
            for st in ctx.stations():
                dates = ctx.store.available_dates("string", st)
                lag = (ctx.run_date - date.fromisoformat(dates[-1])).days if dates else None
                if lag is None or lag > ctx.settings.ingest.stale_data_alert_days:
                    return False, f"{st} 数据时延 {lag} 天"
        from .review import ReviewAgent
        st = ReviewAgent.stats(ctx.store, days=d.auto_gate_window_days)
        if int(st.get("field_missed", 0) or 0) >= d.gate_close_field_missed:
            return False, f"近 {d.auto_gate_window_days} 天现场漏报 {st['field_missed']} 起"
        return True, "ok"

    # ------------------------------------------------------------------ 生成
    def generate(self, ctx: AgentContext) -> List[str]:
        today = ctx.run_date.isoformat()
        cand = ctx.store.query(
            "SELECT s.*, v.cause ai_cause, v.confidence, v.action, v.explanation FROM suspects s "
            "LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id "
            "WHERE s.grade != '已备案' AND (s.status=? OR (s.status='open' AND v.action=?))",
            (SuspectStatus.CONFIRMED.value, Action.DISPATCH.value))
        if cand.empty:
            return []
        # 已在工单中的嫌疑不重复
        in_wo = set()
        for js in ctx.store.query("SELECT suspect_ids_json FROM workorders WHERE status IN ('draft','sent','done','reopened')")["suspect_ids_json"]:
            in_wo.update(json.loads(js))
        cand = cand[~cand["suspect_id"].isin(in_wo)]
        if cand.empty:
            return []
        # 档案问题不派现场（朝向/档案差异）→ 只在复核界面处理
        cand = cand[~cand["rule_cause"].isin([Cause.ORIENTATION.value, Cause.PROFILE_ERROR.value, Cause.CURTAILMENT.value])]
        if cand.empty:
            return []
        prof = ctx.store.strings().set_index("string_id")
        cand["roof_area"] = cand["string_id"].map(prof["roof_area"]).fillna("")
        cand.loc[cand["scope"] == "inverter", "roof_area"] = "逆变器:" + cand.loc[cand["scope"] == "inverter", "inverter_id"]
        groups = cand.groupby(["station_id", "roof_area"]) if ctx.settings.dispatch.merge_same_roof else cand.groupby(["station_id", "suspect_id"])
        created = []
        for (station_id, roof), g in groups:
            wo = self._build(ctx, station_id, roof, g, prof)
            ctx.store.upsert_df("workorders", pd.DataFrame([wo]))
            ctx.store.executemany("UPDATE suspects SET status=? WHERE suspect_id=?",
                                  [(SuspectStatus.DISPATCHED.value, sid) for sid in g["suspect_id"]])
            created.append(wo["wo_id"])
        return created

    def _build(self, ctx: AgentContext, station_id: str, roof: str, g: pd.DataFrame, prof: pd.DataFrame) -> Dict:
        cfg = ctx.settings
        causes = [c for c in (g["rule_cause"].tolist()) if c]
        uniq = list(dict.fromkeys(causes))
        main_cause = " / ".join(uniq) if uniq else Cause.UNKNOWN.value      # 区域工单列出全部原因
        checklist: List[str] = []
        for c in dict.fromkeys(causes):
            checklist.extend([f"[{c}] {x}" for x in CHECKLIST.get(c, CHECKLIST[Cause.UNKNOWN.value])])
        kwh_day = kwh_total = yuan_total = 0.0
        for sid in g["suspect_id"]:
            cl = LossAgent.case_loss(ctx.store, sid)
            kwh_day += cl["per_day_kwh"]; kwh_total += cl["lost_kwh"]; yuan_total += cl["lost_yuan"]
        price = cfg.loss.tou_price.get("flat", cfg.loss.default_price_yuan_per_kwh)
        repair_cost = max(cfg.loss.repair_cost_by_cause.get(c, 300.0) for c in dict.fromkeys(causes)) if causes else 300.0
        priority = kwh_day * price * cfg.loss.expected_fault_days_default / max(repair_cost, 1.0)
        strings = [s for s in g["string_id"].dropna().tolist()]
        locs = []
        for s in strings:
            if s in prof.index:
                p = prof.loc[s]
                locs.append(f"{s}（{p['inverter_id']} pv{int(p['pv_index']) if pd.notna(p['pv_index']) else '?'}，{p['roof_area'] or '位置待补'}）")
        for inv in g.loc[g["scope"] == "inverter", "inverter_id"].tolist():
            locs.append(f"逆变器 {inv}（整机）")
        symptom = "；".join(f"{r.string_id or r.inverter_id}: {r.grade}级 {r.rule_cause} 偏差 {(r.dev_today if r.dev_today is not None else float('nan')):+.1%} 持续 {r.persist_days} 天"
                            for r in g.itertuples())
        wo_id = f"WO-{ctx.run_date:%Y%m%d}-{uuid.uuid4().hex[:6].upper()}"
        return {
            "wo_id": wo_id, "station_id": station_id, "roof_area": roof,
            "suspect_ids_json": json.dumps(g["suspect_id"].tolist(), ensure_ascii=False),
            "string_ids_json": json.dumps(strings, ensure_ascii=False),
            "cause": main_cause, "symptom": symptom, "checklist_json": json.dumps(checklist, ensure_ascii=False),
            "loss_kwh_per_day": round(kwh_day, 2), "loss_kwh_total": round(kwh_total, 1), "loss_yuan_total": round(yuan_total, 1),
            "priority": round(priority, 2), "status": WorkOrderStatus.DRAFT.value, "channel": cfg.dispatch.channel,
            "dispatch_result": None, "created_at": datetime.now().isoformat(), "sent_at": None, "closed_at": None,
            "field_cause": None, "field_action": None, "field_note": None, "field_photos_json": "[]",
            "recheck_due": None, "recheck_result": None,
            "_locations": locs, "_explanations": [e for e in g["explanation"].fillna("").tolist()],
        }

    # ------------------------------------------------------------------ 渲染 / 发送
    @staticmethod
    def render(store, wo: Dict) -> str:
        prof = store.strings().set_index("string_id")
        strings = json.loads(wo["string_ids_json"])
        locs = []
        for s in strings:
            if s in prof.index:
                p = prof.loc[s]
                locs.append(f"- {s}（{p['inverter_id']} pv{int(p['pv_index']) if pd.notna(p['pv_index']) else '?'}，{p['roof_area'] or '位置待补'}）")
        expl = store.query(f"SELECT v.explanation FROM verdicts v WHERE v.suspect_id IN ({','.join('?' * len(json.loads(wo['suspect_ids_json'])))})",
                           json.loads(wo["suspect_ids_json"]))
        lines = [
            f"## 运维工单 {wo['wo_id']}",
            f"**电站**：{wo['station_id']}　**区域**：{wo['roof_area'] or '-'}　**优先级**：{wo['priority']:.1f}　**状态**：{wo['status']}",
            f"**原因假设**：{wo['cause']}",
            f"**损失**：约 {wo['loss_kwh_per_day']:.1f} kWh/天，累计 {wo['loss_kwh_total']:.0f} kWh ≈ {wo['loss_yuan_total']:.0f} 元",
            "", "**涉及组串**：", *locs, "", "**症状**：", wo["symptom"], "", "**判定依据**：",
            *[f"- {e}" for e in expl["explanation"].tolist()], "", "**检查项**：",
            *[f"{i + 1}. {c}" for i, c in enumerate(json.loads(wo["checklist_json"]))],
            "", "处理后请回填：实际原因 / 处理动作 / 照片（系统将在 3 个晴好天后自动复查偏差是否归零）",
        ]
        return "\n".join(lines)

    def _channel(self, ctx: AgentContext) -> DispatchChannel:
        d = ctx.settings.dispatch
        if d.channel == "wecom" and d.wecom_webhook:
            return WeComChannel(d.wecom_webhook)
        return OutboxChannel(ctx.settings.outbox_dir)

    def send(self, ctx: AgentContext, wo_id: str) -> str:
        wo = ctx.store.query("SELECT * FROM workorders WHERE wo_id=?", (wo_id,))
        if wo.empty:
            raise KeyError(wo_id)
        wo = wo.iloc[0].to_dict()
        text = self.render(ctx.store, wo)
        result = self._channel(ctx).send(wo, text)
        ctx.store.execute("UPDATE workorders SET status=?, sent_at=?, dispatch_result=?, channel=? WHERE wo_id=?",
                          (WorkOrderStatus.SENT.value, datetime.now().isoformat(), result, self._channel(ctx).name, wo_id))
        return result

    # ------------------------------------------------------------------ 门控：误报率连续 1 个月 <10% 才放开自动派单
    def auto_gate(self, ctx: AgentContext):
        from .review import ReviewAgent
        d = ctx.settings.dispatch
        st = ReviewAgent.stats(ctx.store, days=d.auto_gate_window_days)
        if not st.get("n_reviewed") or st["n_reviewed"] < d.auto_gate_min_reviews:
            return False, f"近 {d.auto_gate_window_days} 天复核样本 {st.get('n_reviewed', 0)} < {d.auto_gate_min_reviews}"
        if (st.get("false_alarm_rate") or 1.0) > d.auto_gate_fa_rate_max:
            return False, f"误报率 {st['false_alarm_rate']:.0%} > {d.auto_gate_fa_rate_max:.0%}"
        return True, "ok"

    # ------------------------------------------------------------------ 现场回填 / 复查
    @staticmethod
    def close(store, wo_id: str, field_cause: str, field_action: str, note: str = "", photos: Optional[List[str]] = None,
              done_date: Optional[date] = None, recheck_clear_days: int = 3) -> None:
        wo = store.query("SELECT * FROM workorders WHERE wo_id=?", (wo_id,))
        if wo.empty:
            raise KeyError(wo_id)
        wo = wo.iloc[0]
        done_date = done_date or date.today()
        ev = {"维修": EventType.REPAIR, "清洗": EventType.CLEANING, "更换": EventType.MODULE_REPLACE,
              "重接线": EventType.REWIRE, "换逆变器": EventType.INVERTER_REPLACE}.get(field_action, EventType.MAINTENANCE)
        for s in json.loads(wo["string_ids_json"]):
            ProfileAgent.add_event(store, "string", s, ev.value, done_date, f"{wo_id} {field_cause}")
        for sid in json.loads(wo["suspect_ids_json"]):
            r = store.query("SELECT scope, inverter_id FROM suspects WHERE suspect_id=?", (sid,))
            if not r.empty and r["scope"].iloc[0] == "inverter":
                ProfileAgent.add_event(store, "inverter", r["inverter_id"].iloc[0], ev.value, done_date, f"{wo_id} {field_cause}")
        # closed_at 记录"现场处理日"（而非录入时刻），复查以此为起点
        closed_at = datetime.combine(done_date, datetime.now().time()).isoformat()
        store.execute(
            "UPDATE workorders SET status=?, closed_at=?, field_cause=?, field_action=?, field_note=?, field_photos_json=?, recheck_due=? WHERE wo_id=?",
            (WorkOrderStatus.DONE.value, closed_at, field_cause, field_action, note,
             json.dumps(photos or [], ensure_ascii=False), (done_date + timedelta(days=recheck_clear_days)).isoformat(), wo_id))

    def recheck(self, ctx: AgentContext) -> int:
        """done 且到复查期：处理后 ≥N 个晴好天偏差回到正常带 → verified（嫌疑 closed）；否则 reopened"""
        th, dcfg = ctx.settings.thresholds, ctx.settings.dispatch
        today = ctx.run_date.isoformat()
        wos = ctx.store.query("SELECT * FROM workorders WHERE status=? AND recheck_due<=?", (WorkOrderStatus.DONE.value, today))
        n = 0
        for wo in wos.to_dict("records"):
            closed = (wo["closed_at"] or today)[:10]
            strings = json.loads(wo["string_ids_json"])
            ok_all, pending = True, False
            for s in strings:
                b = ctx.store.query("SELECT dev_today, weather_tag, zero_all_day FROM baselines bl JOIN daily_features f "
                                    "ON f.string_id=bl.string_id AND f.date=bl.date WHERE bl.string_id=? AND bl.date>? ORDER BY bl.date", (s, closed))
                clear = b[b["weather_tag"] == "clear"]
                if len(clear) < dcfg.recheck_clear_days:
                    pending = True
                    break
                recent = clear.tail(dcfg.recheck_clear_days)
                if (recent["dev_today"] <= th.normal_floor).any() or (recent["zero_all_day"] == 1).any():
                    ok_all = False
            if pending:
                continue
            n += 1
            if ok_all:
                ctx.store.execute("UPDATE workorders SET status=?, recheck_result=? WHERE wo_id=?",
                                  (WorkOrderStatus.VERIFIED.value, f"{today} 复查：偏差归零", wo["wo_id"]))
                ctx.store.executemany("UPDATE suspects SET status=? WHERE suspect_id=?",
                                      [(SuspectStatus.CLOSED.value, sid) for sid in json.loads(wo["suspect_ids_json"])])
            else:
                ctx.store.execute("UPDATE workorders SET status=?, recheck_result=?, recheck_due=NULL WHERE wo_id=?",
                                  (WorkOrderStatus.REOPENED.value, f"{today} 复查：偏差未归零，重开", wo["wo_id"]))
                ctx.store.alert(ctx.run_date, "warn", self.name, f"工单 {wo['wo_id']} 处理后偏差未归零，已重开")
        return n
