"""LLM 解释与复核助手（可选，离线可关）

做什么：把一个案例的全部事实（偏差序列、签名、档案、历史复核、同类案例的现场结论、损失、工单）整理成"案卷"，
让大模型写一段复核员能直接读的分析（结论 / 依据 / 建议 / 不确定性），并回答复核员的追问。

不做什么：不替代规则引擎和分类模型做判定；模型只能引用案卷里的事实，不得编造数据；没有 API key 时退化为模板文案，功能全部可用。

提供方：Anthropic Messages API（`ANTHROPIC_API_KEY`），只用 requests，不引入 SDK；`llm.enabled=false` 彻底关闭（私有化/内网）。
缓存：同一案卷（指纹）只调用一次，存 platform_state。
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from .agents.loss import LossAgent
from .config import Settings
from .storage import Store

SYSTEM_PROMPT = (
    "你是光伏电站运维诊断分析师。用户给你一份组串异常案卷（JSON），请用简体中文写给一线复核员看的分析。"
    "要求：只引用案卷中的事实与数字，不要编造；结构固定为四段：【结论】一句话；【依据】3~5 条，每条带数字；"
    "【建议】现场该做什么、按什么顺序、不该做什么；【不确定性】哪些证据还不够、什么情况下结论会反转。"
    "总长 250 字以内。若案卷里规则判定与分类模型判定不一致，要明确指出并说明更信哪一个、为什么。"
)


# ---------------------------------------------------------------------------- 案卷
def case_dossier(store: Store, suspect_id: str, settings: Optional[Settings] = None) -> Dict[str, Any]:
    s = store.query("SELECT s.*, v.cause ai_cause, v.confidence, v.action, v.explanation, v.model_version "
                    "FROM suspects s LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id WHERE s.suspect_id=?", (suspect_id,))
    if s.empty:
        raise KeyError(suspect_id)
    s = s.iloc[0].to_dict()
    sig = json.loads(s.get("signature_json") or "{}")
    sig.pop("notes", None)
    sig = {k: v for k, v in sig.items() if v not in (None, False, 0, 0.0)}
    d: Dict[str, Any] = {
        "case": {k: s.get(k) for k in ("suspect_id", "station_id", "inverter_id", "string_id", "scope", "grade", "rule_cause",
                                       "dev_today", "persist_days", "first_date", "date", "status", "low_confidence")},
        "ai_verdict": {"cause": s.get("ai_cause"), "confidence": s.get("confidence"), "action": s.get("action"), "model": s.get("model_version")},
        "signature": sig,
        "rule_notes": json.loads(s.get("signature_json") or "{}").get("notes", []),
    }
    if s["scope"] == "string":
        p = store.query("SELECT pv_index, n_modules, module_model, azimuth, tilt, roof_area, known_shading, shading_note, group_key FROM strings WHERE string_id=?", (s["string_id"],))
        d["profile"] = p.iloc[0].to_dict() if not p.empty else {}
        tr = store.query("SELECT date, weather_tag, dev_today, group_noise, baseline_ratio, baseline_n, orient_exp_ratio FROM baselines "
                         "WHERE string_id=? AND date<=? ORDER BY date DESC LIMIT 20", (s["string_id"], s["date"]))
        d["recent_days"] = [{k: (None if pd.isna(v) else (round(float(v), 4) if isinstance(v, float) else v)) for k, v in r.items()}
                            for r in tr.to_dict("records")]
        ev = store.query("SELECT event_type, event_date, notes FROM events WHERE target_id=? ORDER BY event_date DESC LIMIT 5", (s["string_id"],))
        d["events"] = ev.to_dict("records")
    rv = store.query("SELECT reviewer, decision, true_cause, reject_reason, note, reviewed_at FROM reviews WHERE suspect_id=? AND decision!='note' ORDER BY reviewed_at", (suspect_id,))
    d["reviews"] = rv.to_dict("records")
    try:
        d["loss"] = LossAgent.case_loss(store, suspect_id)
    except Exception:
        d["loss"] = {}
    wo = store.query("SELECT wo_id, status, field_cause, field_action, recheck_result FROM workorders WHERE suspect_ids_json LIKE ?", (f"%{suspect_id}%",))
    d["workorders"] = wo.to_dict("records")
    # 同类历史：同一规则原因、已有现场结论的案例，看现场最终是什么（经验库）
    hist = store.query(
        "SELECT s.rule_cause, r.decision, r.true_cause, COUNT(*) n FROM reviews r JOIN suspects s ON s.suspect_id=r.suspect_id "
        "WHERE s.rule_cause=? AND r.suspect_id!=? AND r.decision IN ('confirm','modify','reject') GROUP BY r.decision, r.true_cause ORDER BY n DESC LIMIT 6",
        (s["rule_cause"], suspect_id))
    d["similar_cases_outcomes"] = hist.to_dict("records")
    return d


def dossier_fingerprint(d: Dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(d, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------- 客户端
class LLMClient:
    def __init__(self, settings: Settings):
        self.cfg = settings.llm
        self.api_key = os.environ.get(self.cfg.api_key_env, "") or ""

    @property
    def available(self) -> bool:
        return bool(self.cfg.enabled and self.api_key)

    def complete(self, system: str, user: str) -> str:
        import requests
        r = requests.post(
            f"{self.cfg.base_url.rstrip('/')}/v1/messages",
            headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": self.cfg.model, "max_tokens": self.cfg.max_tokens, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=self.cfg.timeout_seconds)
        r.raise_for_status()
        body = r.json()
        return "".join(part.get("text", "") for part in body.get("content", []) if part.get("type") == "text").strip()


# ---------------------------------------------------------------------------- 模板回退
def template_narrative(d: Dict[str, Any]) -> str:
    c, ai = d["case"], d["ai_verdict"]
    obj = c.get("string_id") or f"{c.get('inverter_id')}（整机）"
    dev = c.get("dev_today")
    dev_s = f"{dev:+.1%}" if isinstance(dev, (int, float)) and dev is not None and not pd.isna(dev) else "—"
    lines = [f"【结论】{obj} {c['grade']} 级，规则判 {c['rule_cause']}" + (f"，模型判 {ai['cause']}（置信 {ai['confidence']:.0%}）" if ai.get("cause") and ai.get("confidence") is not None else "") + "。"]
    basis = [f"互比偏差 {dev_s}，持续 {c.get('persist_days')} 天，案例起于 {c.get('first_date')}"]
    basis += d.get("rule_notes", [])[:3]
    lines.append("【依据】" + "；".join(basis))
    if d.get("similar_cases_outcomes"):
        top = d["similar_cases_outcomes"][0]
        lines.append(f"【经验】同为「{c['rule_cause']}」的历史案例中，最常见的复核结果是 {top['decision']}→{top.get('true_cause') or '—'}（{top['n']} 次）。")
    lines.append(f"【建议】{ai.get('action') or '转人工复核'}；" + ("已有工单 " + "、".join(w['wo_id'] for w in d.get('workorders', [])) if d.get("workorders") else "尚未派单"))
    loss = d.get("loss") or {}
    if loss.get("lost_kwh"):
        lines.append(f"【损失】累计约 {loss['lost_kwh']:.0f} kWh ≈ {loss.get('lost_yuan', 0):.0f} 元。")
    lines.append("【不确定性】以上为模板文案（未启用大模型：配置 llm.enabled 并设置 ANTHROPIC_API_KEY 后由模型生成分析）。")
    return "\n".join(lines)


def template_answer(d: Dict[str, Any], question: str) -> str:
    c = d["case"]
    q = question
    if any(k in q for k in ("损失", "电量", "钱")):
        loss = d.get("loss") or {}
        return f"该案例累计损失约 {loss.get('lost_kwh', 0):.0f} kWh ≈ {loss.get('lost_yuan', 0):.0f} 元（起算日 {c.get('first_date')}）。"
    if any(k in q for k in ("为什么", "依据", "原因")):
        return "规则依据：" + "；".join(d.get("rule_notes", [])[:4] or ["无"]) + f"。模型判定：{d['ai_verdict'].get('cause')}。"
    if any(k in q for k in ("历史", "以前", "同类")):
        h = d.get("similar_cases_outcomes") or []
        return "同类案例复核结果：" + ("；".join(f"{x['decision']}→{x.get('true_cause') or '—'} ×{x['n']}" for x in h) if h else "暂无") + "。"
    if any(k in q for k in ("怎么办", "建议", "现场", "处理")):
        return f"建议动作：{d['ai_verdict'].get('action') or '转人工复核'}；工单：{', '.join(w['wo_id'] + '(' + w['status'] + ')' for w in d.get('workorders', [])) or '尚未派单'}。"
    return "未启用大模型（llm.enabled + ANTHROPIC_API_KEY）。可问：损失多少 / 为什么这样判 / 同类历史 / 现场怎么办。"


# ---------------------------------------------------------------------------- 入口
def narrative(settings: Settings, store: Store, suspect_id: str, force: bool = False) -> Dict[str, Any]:
    d = case_dossier(store, suspect_id, settings)
    fp = dossier_fingerprint(d)
    key = f"llm:narrative:{suspect_id}"
    cached = store.get_state(key)
    if cached and cached.get("fp") == fp and not force:
        return cached
    client = LLMClient(settings)
    if client.available:
        try:
            text = client.complete(SYSTEM_PROMPT, "案卷：\n" + json.dumps(d, ensure_ascii=False, default=str))
            out = {"text": text, "source": f"llm:{settings.llm.model}", "fp": fp, "at": datetime.now().isoformat()}
        except Exception as e:      # 瞬时失败不缓存，下次打开再试
            return {"text": template_narrative(d) + f"\n（大模型调用失败：{e}）", "source": "template", "fp": fp, "at": datetime.now().isoformat()}
    else:
        out = {"text": template_narrative(d), "source": "template", "fp": fp, "at": datetime.now().isoformat()}
    store.set_state(key, out)
    return out


def ask(settings: Settings, store: Store, suspect_id: str, question: str) -> Dict[str, Any]:
    d = case_dossier(store, suspect_id, settings)
    client = LLMClient(settings)
    if client.available:
        try:
            text = client.complete(SYSTEM_PROMPT + " 现在用户在追问，请直接回答问题，仍只引用案卷事实，100 字以内。",
                                   "案卷：\n" + json.dumps(d, ensure_ascii=False, default=str) + f"\n\n问题：{question}")
            source = f"llm:{settings.llm.model}"
        except Exception as e:
            text, source = template_answer(d, question) + f"\n（大模型调用失败：{e}）", "template"
    else:
        text, source = template_answer(d, question), "template"
    store.execute("INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                  (suspect_id, "assistant", "note", None, None, f"问：{question}\n答：{text}", datetime.now().isoformat()))
    return {"text": text, "source": source}
