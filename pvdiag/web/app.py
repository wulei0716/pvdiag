"""阶段6 · 运维/复核控制台（FastAPI + Jinja2 + 服务端 SVG，零外网依赖）

页面：
  /                 看板：组串热力图（电站一眼看全）+ 今日待办 + 损失台账 + 流水线状态
  /strings          全站热力图（按站/按日）
  /string/{id}      任意组串详情（当日曲线 vs 群组、45 天偏差趋势、档案、案例）
  /review           复核队列 → /suspect/{id} 确认 / 驳回 / 修改原因（回写标注库）
  /workorders       工单 → /workorder/{id} 一键发送 / 现场回填
  /reports          每日巡检报告（结构化渲染 report.json）
  /api/*            JSON（供自有工单系统/下游消费）
启动：pvdiag serve --port 8080
"""
from __future__ import annotations

import html
import json
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from jinja2 import DictLoader, Environment, select_autoescape

from ..agents.base import AgentContext
from ..agents.loss import LossAgent
from ..agents.review import ReviewAgent
from ..agents.workorder import WorkOrderAgent
from ..config import Settings
from ..features import slot_label
from ..models import CAUSE_CHOICES, REJECT_REASONS
from ..storage import Store
from ..connections import input_evidence
from .svgchart import bar_chart, hbar_chart, heatmap, line_chart
from .templates import TEMPLATES

WEATHER_CN = {"clear": "晴好", "cloudy": "多云", "overcast": "阴雨", "nodata": "无数据"}
STATUS_CN = {"open": "待处理", "reviewing": "复核中", "confirmed": "已确认", "rejected": "已驳回（误报）",
             "dispatched": "已派单", "closed": "已关闭", "draft": "草稿", "sent": "已派发", "done": "待复查",
             "verified": "复查归零", "reopened": "已重开"}


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.load()
    settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    env = Environment(loader=DictLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda x: ("—" if _isnan(x) else f"{float(x):+.1%}")
    env.filters["num"] = lambda x, d=1: ("—" if _isnan(x) else f"{float(x):,.{d}f}")
    env.filters["cn"] = lambda x: STATUS_CN.get(x, x) if x else "—"
    env.filters["wcn"] = lambda x: WEATHER_CN.get(x, x or "—")
    env.filters["md"] = lambda s: md_to_html(s or "")
    app = FastAPI(title="pvdiag · 组串异常诊断控制台")

    # ------------------------------------------------------------------ 公共
    def render(name: str, **kw) -> HTMLResponse:
        kw.setdefault("now", datetime.now().strftime("%m-%d %H:%M"))
        kw.setdefault("stations", store.stations()[["station_id", "name"]].pipe(recs))
        kw.setdefault("latest", latest_date())
        kw.setdefault("pending_n", int(store.scalar(
            "SELECT COUNT(*) FROM suspects WHERE status IN ('open','reviewing') AND grade!='已备案'") or 0))
        kw.setdefault("draft_n", int(store.scalar("SELECT COUNT(*) FROM workorders WHERE status IN ('draft','reopened')") or 0))
        return HTMLResponse(env.get_template(name).render(**kw))

    def latest_date() -> Optional[str]:
        return store.scalar("SELECT MAX(date) FROM baselines")

    def report_json(d: str) -> Dict:
        p = settings.reports_dir / d / "report.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # ------------------------------------------------------------------ 看板
    @app.get("/", response_class=HTMLResponse)
    def index(station: Optional[str] = None):
        d = latest_date()
        rep = report_json(d) if d else {}
        sids = [station] if station else store.stations()["station_id"].tolist()
        heat = [station_heatmap(store, s, d, settings.thresholds.zero_current_a) for s in sids] if d else []
        q = ReviewAgent.queue(store, station or None)
        unused = set().union(*(input_evidence(store, sid, date.fromisoformat(d), settings.thresholds.zero_current_a)[0] for sid in sids)) if d else set()
        q = q[~q["string_id"].isin(unused)]
        by_grade = q["grade"].value_counts().to_dict() if not q.empty else {}
        todo = q.head(8).pipe(recs)
        wos = store.query("SELECT wo_id, station_id, roof_area, cause, priority, loss_kwh_per_day, status FROM workorders "
                          "WHERE status IN ('draft','reopened','sent') ORDER BY priority DESC LIMIT 6")
        alerts = store.query("SELECT run_date, level, source, message FROM alerts WHERE resolved=0 ORDER BY alert_id DESC LIMIT 8")
        loss = LossAgent.ledger_summary(store, unused)
        loss_rows = [{"label": k, "value": v["lost_yuan"], "color": "var(--c-low)"} for k, v in (loss.get("by_station") or {}).items()]
        case_rows = [{"label": (c["string_id"] or "").replace("inv:", "整机 "), "value": c["lost_yuan"],
                      "color": "var(--c-fault)" if c["grade"] == "A" else "var(--c-low)"} for c in (loss.get("by_case") or [])[:8]]
        weather = store.query("SELECT station_id, weather_tag, noon_power, peak_ref FROM station_daily WHERE date=?", (d,)) if d else pd.DataFrame()
        stats = ReviewAgent.stats(store)
        return render("index.html", date=d, report=rep, heat=heat, by_grade=by_grade, todo=todo, wos=wos.pipe(recs),
                      alerts=alerts.pipe(recs), loss=loss, loss_svg=hbar_chart(loss_rows, unit=" 元"),
                      case_svg=hbar_chart(case_rows, unit=" 元"), weather=weather.pipe(recs),
                      board=rep.get("board", {}), stats=stats, station=station or "")

    # ------------------------------------------------------------------ 热力图 / 组串
    @app.get("/strings", response_class=HTMLResponse)
    def strings(station: Optional[str] = None, d: Optional[str] = None):
        d = d or latest_date()
        if not d:
            return render("strings.html", heat=[], date=None, station=station or "", dates=[])
        sids = [station] if station else store.stations()["station_id"].tolist()
        heat = [station_heatmap(store, s, d, settings.thresholds.zero_current_a) for s in sids]
        dates = store.query("SELECT DISTINCT date FROM baselines ORDER BY date DESC LIMIT 14")["date"].tolist()
        return render("strings.html", heat=heat, date=d, station=station or "", dates=dates)

    @app.get("/string/{string_id}", response_class=HTMLResponse)
    def string_detail(string_id: str, d: Optional[str] = None):
        p = store.query("SELECT * FROM strings WHERE string_id=?", (string_id,))
        if p.empty:
            raise HTTPException(404, "组串不存在")
        prof = p.iloc[0].to_dict()
        d = d or store.scalar("SELECT MAX(date) FROM baselines WHERE string_id=?", (string_id,)) or latest_date()
        b = store.query("SELECT * FROM baselines WHERE string_id=? AND date=?", (string_id, d))
        bl = b.iloc[0].to_dict() if not b.empty else {}
        case = store.query("SELECT s.*, v.cause ai_cause, v.confidence, v.action FROM suspects s LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id "
                           "WHERE s.string_id=? ORDER BY s.date DESC", (string_id,))
        sig = store.query("SELECT * FROM signatures WHERE string_id=? ORDER BY date DESC LIMIT 10", (string_id,))
        chart = curve_chart(store, string_id, d, prof["inverter_id"])
        trend = trend_data(store, "string", string_id, d)
        events = store.query("SELECT * FROM events WHERE (scope='string' AND target_id=?) OR (scope='inverter' AND target_id=?) ORDER BY event_date DESC",
                             (string_id, prof["inverter_id"]))
        return render("string.html", prof=prof, bl=bl, date=d, cases=case.pipe(recs), sigs=sig.pipe(recs),
                      chart_svg=line_chart(chart["labels"], chart["series"], chart["title"], "A"),
                      trend_svg=bar_chart(trend["labels"], trend["dev"], trend["weather"], "互比偏差 %（虚线 -3% / -5%）"),
                      events=events.pipe(recs))

    # ------------------------------------------------------------------ 复核
    @app.get("/review", response_class=HTMLResponse)
    def review_list(station: Optional[str] = None, grade: Optional[str] = None):
        q = ReviewAgent.queue(store, station or None, [grade] if grade else None)
        counts = ReviewAgent.queue(store, station or None)["grade"].value_counts().to_dict() if station else \
            ReviewAgent.queue(store)["grade"].value_counts().to_dict()
        return render("review.html", rows=q.pipe(recs), station=station or "", grade=grade or "",
                      counts=counts, causes=CAUSE_CHOICES, reasons=REJECT_REASONS, stats=ReviewAgent.stats(store))

    @app.post("/review/{suspect_id}/confirm")
    def review_confirm(suspect_id: str, true_cause: str = Form(""), note: str = Form(""), reviewer: str = Form("web"),
                       back: str = Form("/review")):
        ReviewAgent.confirm(store, suspect_id, true_cause or None, reviewer, note)
        return RedirectResponse(back, status_code=303)

    @app.post("/review/{suspect_id}/reject")
    def review_reject(suspect_id: str, reason: str = Form("其他"), note: str = Form(""), reviewer: str = Form("web"),
                      back: str = Form("/review")):
        ReviewAgent.reject(store, suspect_id, reason, reviewer, note)
        return RedirectResponse(back, status_code=303)

    @app.get("/suspect/{suspect_id}", response_class=HTMLResponse)
    def suspect_detail(suspect_id: str):
        s = store.query("SELECT s.*, v.cause ai_cause, v.confidence, v.action, v.explanation, v.model_version, v.proba_json "
                        "FROM suspects s LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id WHERE s.suspect_id=?", (suspect_id,))
        if s.empty:
            raise HTTPException(404, "嫌疑不存在")
        s = s.iloc[0].to_dict()
        sig = json.loads(s.get("signature_json") or "{}")
        if s["scope"] == "inverter":
            chart = inverter_chart(store, s["inverter_id"], s["date"])
        else:
            chart = curve_chart(store, s["string_id"], s["date"], s["inverter_id"])
        trend = trend_data(store, s["scope"], s["string_id"] if s["scope"] == "string" else s["inverter_id"], s["date"])
        prof = {}
        if s["scope"] == "string":
            p = store.query("SELECT * FROM strings WHERE string_id=?", (s["string_id"],))
            prof = p.iloc[0].to_dict() if not p.empty else {}
        reviews = store.query("SELECT * FROM reviews WHERE suspect_id=? ORDER BY reviewed_at DESC", (suspect_id,))
        loss = LossAgent.case_loss(store, suspect_id)
        wo = store.query("SELECT wo_id, status FROM workorders WHERE suspect_ids_json LIKE ?", (f"%{suspect_id}%",))
        # 上一条/下一条（按队列顺序）用于连续复核
        q = ReviewAgent.queue(store)["suspect_id"].tolist() if True else []
        nxt = q[q.index(suspect_id) + 1] if suspect_id in q and q.index(suspect_id) + 1 < len(q) else (q[0] if q and q[0] != suspect_id else None)
        from ..llm import narrative
        try:
            analysis = narrative(settings, store, suspect_id)
        except Exception as e:      # 助手失败不影响复核页
            analysis = {"text": f"分析暂不可用：{e}", "source": "error"}
        qa = store.query("SELECT note, reviewed_at FROM reviews WHERE suspect_id=? AND decision='note' ORDER BY reviewed_at DESC LIMIT 5", (suspect_id,))
        return render("suspect.html", s=s, sig=sig, chart_svg=line_chart(chart["labels"], chart["series"], chart["title"], "A"),
                      trend_svg=bar_chart(trend["labels"], trend["dev"], trend["weather"], "互比偏差 %（虚线 -3% / -5%）"),
                      prof=prof, reviews=reviews[reviews["decision"] != "note"].pipe(recs) if not reviews.empty else [], loss=loss,
                      causes=CAUSE_CHOICES, reasons=REJECT_REASONS,
                      proba=json.loads(s.get("proba_json") or "{}"), wo=wo.pipe(recs), next_id=nxt, analysis=analysis, qa=qa.pipe(recs))

    @app.post("/suspect/{suspect_id}/ask")
    def suspect_ask(suspect_id: str, question: str = Form(...)):
        from ..llm import ask
        ask(settings, store, suspect_id, question.strip()[:300])
        return RedirectResponse(f"/suspect/{suspect_id}", status_code=303)

    # ------------------------------------------------------------------ 工单
    @app.get("/workorders", response_class=HTMLResponse)
    def workorders(status: Optional[str] = None):
        sql, params = "SELECT * FROM workorders", []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY CASE status WHEN 'draft' THEN 0 WHEN 'reopened' THEN 1 WHEN 'sent' THEN 2 WHEN 'done' THEN 3 ELSE 4 END, priority DESC"
        df = store.query(sql, params)
        counts = store.query("SELECT status, COUNT(*) n FROM workorders GROUP BY status")
        return render("workorders.html", rows=df.pipe(recs), status=status or "",
                      counts=counts.set_index("status")["n"].to_dict() if not counts.empty else {})

    @app.get("/workorder/{wo_id}", response_class=HTMLResponse)
    def workorder_detail(wo_id: str):
        wo = store.query("SELECT * FROM workorders WHERE wo_id=?", (wo_id,))
        if wo.empty:
            raise HTTPException(404)
        wo = wo.iloc[0].to_dict()
        return render("workorder.html", wo=wo, body_html=md_to_html(WorkOrderAgent.render(store, wo)),
                      suspect_ids=json.loads(wo["suspect_ids_json"]), checklist=json.loads(wo["checklist_json"] or "[]"),
                      recheck_days=settings.dispatch.recheck_clear_days)

    @app.post("/workorder/{wo_id}/send")
    def workorder_send(wo_id: str):
        ctx = AgentContext(settings=settings, store=store, run_date=date.today())
        WorkOrderAgent().send(ctx, wo_id)
        return RedirectResponse(f"/workorder/{wo_id}", status_code=303)

    @app.post("/workorder/{wo_id}/close")
    def workorder_close(wo_id: str, field_cause: str = Form(...), field_action: str = Form("维修"), note: str = Form(""),
                        done_date: str = Form("")):
        WorkOrderAgent.close(store, wo_id, field_cause, field_action, note,
                             done_date=date.fromisoformat(done_date) if done_date else None,
                             recheck_clear_days=settings.dispatch.recheck_clear_days)
        return RedirectResponse(f"/workorder/{wo_id}", status_code=303)

    # ------------------------------------------------------------------ 报告
    @app.get("/reports", response_class=HTMLResponse)
    def reports():
        dates = sorted([p.name for p in settings.reports_dir.iterdir() if (p / "report.json").exists()], reverse=True) if settings.reports_dir.exists() else []
        return RedirectResponse(f"/report/{dates[0]}", status_code=303) if dates else render("report.html", r={}, dates=[], date=None)

    @app.get("/report/{d}", response_class=HTMLResponse)
    def report(d: str):
        r = report_json(d)
        if not r:
            raise HTTPException(404, "无该日报告")
        dates = sorted([p.name for p in settings.reports_dir.iterdir() if (p / "report.json").exists()], reverse=True)
        return render("report.html", r=r, dates=dates[:30], date=d)

    # ------------------------------------------------------------------ 自主运行
    @app.get("/autonomy", response_class=HTMLResponse)
    def autonomy():
        from ..agents.tuning import TuningAgent, apply_overrides
        eff = apply_overrides(settings, store)
        gate = store.get_state("dispatch:gate") or {"effective": "draft", "streak": 0}
        changes = store.query("SELECT run_date, param, old_value, new_value, status, reason FROM param_changes ORDER BY change_id DESC LIMIT 30")
        models = store.query("SELECT model_version, trained_at, n_labels, active, status, metrics_json FROM model_registry ORDER BY trained_at DESC LIMIT 10")
        if not models.empty:
            def _exam(m):
                ex = (json.loads(m or "{}").get("exam") or {})
                f = lambda d: (f"{d['accuracy']:.0%}" if d and d.get("accuracy") is not None else "—")
                return f"{f(ex.get('challenger'))} / {f(ex.get('rule_proxy'))} / {f(ex.get('incumbent'))}"
            models["exam"] = models["metrics_json"].map(_exam)
        hd = store.scalar("SELECT MAX(run_date) FROM health_checks")
        health = store.query("SELECT name, status, detail FROM health_checks WHERE run_date=? ORDER BY check_id", (hd,)) if hd else pd.DataFrame()
        ingest = [{"station_id": st, "last": (store.available_dates("string", st) or [None])[-1],
                   "failures": int(store.get_state(f"ingest:{st}:failures", 0) or 0)} for st in store.stations()["station_id"]]
        pushed = store.query("SELECT run_date, message FROM alerts WHERE source='notify' ORDER BY alert_id DESC LIMIT 10")
        latest_rep = report_json(latest_date() or "") if latest_date() else {}
        n_labels = int(store.scalar("SELECT COUNT(DISTINCT suspect_id) FROM reviews WHERE decision IN ('confirm','modify','reject')") or 0)
        return render("autonomy.html", cfg_mode=settings.dispatch.mode, gate=gate, open_days=settings.dispatch.gate_open_streak_days,
                      autotune=settings.autotune, current=TuningAgent.current_params(eff),
                      overrides=store.query("SELECT param FROM param_overrides").pipe(recs),
                      tuning_decision=(latest_rep.get("tuning") or {}).get("decision"), changes=changes.pipe(recs),
                      models=models.pipe(recs) if not models.empty else [], live=store.get_state("model:live_agreement"),
                      n_labels=n_labels, min_labels=settings.confidence.min_labels_for_ml, health=health.pipe(recs) if not health.empty else [],
                      health_date=hd, ingest=ingest, pushed=pushed.pipe(recs))

    # ------------------------------------------------------------------ JSON API
    @app.get("/api/queue")
    def api_queue(station: Optional[str] = None):
        return JSONResponse(json.loads(ReviewAgent.queue(store, station).to_json(orient="records", force_ascii=False)))

    @app.get("/api/report/{d}")
    def api_report(d: str):
        r = report_json(d)
        if not r:
            raise HTTPException(404)
        return JSONResponse(r)

    @app.get("/api/heatmap/{station_id}")
    def api_heatmap(station_id: str, d: Optional[str] = None):
        return JSONResponse(station_heatmap(store, station_id, d or latest_date(), settings.thresholds.zero_current_a) | {"svg": None})

    @app.get("/api/string/{string_id}/chart")
    def api_chart(string_id: str, d: Optional[str] = None):
        p = store.query("SELECT inverter_id FROM strings WHERE string_id=?", (string_id,))
        if p.empty:
            raise HTTPException(404)
        return JSONResponse(curve_chart(store, string_id, d or latest_date(), p["inverter_id"].iloc[0]))

    return app


# ---------------------------------------------------------------------- 数据组装
def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def recs(df: pd.DataFrame) -> List[Dict]:
    """DataFrame → 字典列表，NaN 统一转 None（模板里 `or` 判断才正确）"""
    if df is None or df.empty:
        return []
    return df.astype(object).where(pd.notna(df), None).to_dict("records")


def station_heatmap(store: Store, station_id: str, d: Optional[str], zero_current_a: float = 0.5) -> Dict:
    prof = store.strings(station_id)
    unused = input_evidence(store, station_id, date.fromisoformat(d), zero_current_a)[0] if d else set()
    prof = prof[~prof["string_id"].isin(unused)]
    inv = store.inverters(station_id)
    bl = store.baselines(station_id, d, d).set_index("string_id") if d else pd.DataFrame()
    ft = store.query("SELECT string_id, zero_all_day, stuck FROM daily_features WHERE date=?", (d,)).set_index("string_id") if d else pd.DataFrame()
    active = store.query("SELECT string_id, grade, rule_cause FROM suspects WHERE station_id=? AND status IN ('open','reviewing','confirmed','dispatched') "
                         "AND scope='string'", (station_id,)).set_index("string_id") if True else pd.DataFrame()
    inv_cases = store.query("SELECT inverter_id, grade, rule_cause FROM suspects WHERE station_id=? AND scope='inverter' AND status IN ('open','reviewing','confirmed','dispatched')",
                            (station_id,)).set_index("inverter_id")
    rows, max_pv, n_cases, n_dead = [], 1, 0, 0
    for r in inv.itertuples():
        cells: Dict[int, Dict] = {}
        for s in prof[prof["inverter_id"] == r.inverter_id].itertuples():
            pv = int(s.pv_index) if pd.notna(s.pv_index) else len(cells) + 1
            max_pv = max(max_pv, pv)
            dev = float(bl.loc[s.string_id, "dev_today"]) if (not bl.empty and s.string_id in bl.index and pd.notna(bl.loc[s.string_id, "dev_today"])) else None
            zero = bool(ft.loc[s.string_id, "zero_all_day"]) if (not ft.empty and s.string_id in ft.index) else False
            nodata = (bl.empty or s.string_id not in bl.index)
            grade = cause = None
            if not active.empty and s.string_id in active.index:
                grade, cause = active.loc[s.string_id, "grade"], active.loc[s.string_id, "rule_cause"]
                n_cases += 1
            n_dead += int(zero)
            cells[pv] = {"string_id": s.string_id, "dev": dev, "zero": zero, "nodata": nodata, "grade": grade, "cause": cause}
        name = r.name or r.inverter_id
        if not inv_cases.empty and r.inverter_id in inv_cases.index:
            name = f"⚠ {name}"
        rows.append({"inverter_id": r.inverter_id, "name": name.replace(f"{station_id}-", ""), "cells": cells})
    weather = store.query("SELECT weather_tag, noon_power, peak_ref FROM station_daily WHERE station_id=? AND date=?", (station_id, d)) if d else pd.DataFrame()
    st = store.query("SELECT name, capacity_kwp FROM stations WHERE station_id=?", (station_id,))
    return {"station_id": station_id, "name": (st["name"].iloc[0] if not st.empty else station_id) or station_id,
            "capacity_kwp": float(st["capacity_kwp"].iloc[0]) if not st.empty and pd.notna(st["capacity_kwp"].iloc[0]) else None,
            "weather": weather.iloc[0].to_dict() if not weather.empty else {}, "n_strings": int(len(prof)), "n_cases": n_cases,
            "n_unused": len(unused), "n_dead": n_dead, "svg": heatmap(rows, max_pv), "inverters": [{"inverter_id": x["inverter_id"], "n": len(x["cells"])} for x in rows]}


def curve_chart(store: Store, sid: str, d: str, inverter_id: str) -> Dict:
    labels = [slot_label(i) for i in range(48)]
    me = store.query("SELECT i_profile_json FROM daily_features WHERE string_id=? AND date=?", (sid, d))
    bl = store.query("SELECT group_key, group_level FROM baselines WHERE string_id=? AND date=?", (sid, d))
    peers = pd.DataFrame()
    if not bl.empty:
        key, level = bl["group_key"].iloc[0], bl["group_level"].iloc[0]
        peers = store.query(
            "SELECT f.i_profile_json FROM daily_features f JOIN strings p ON p.string_id=f.string_id WHERE f.date=? AND f.string_id!=? "
            "AND f.zero_all_day=0 AND f.stuck=0 AND " + ("p.group_key=?" if level == "group" else "p.inverter_id=?"),
            (d, sid, key if level == "group" else inverter_id))
    mine = json.loads(me["i_profile_json"].iloc[0]) if not me.empty else [None] * 48
    gmed = [None] * 48
    if len(peers) >= 2:
        M = np.array([[np.nan if v is None else v for v in json.loads(x)] for x in peers["i_profile_json"]], dtype=float)
        med = np.nanmedian(M, axis=0)
        gmed = [None if np.isnan(v) else round(float(v), 3) for v in med]
    return {"labels": labels, "series": [{"label": sid, "data": mine}, {"label": "群组中位", "data": gmed}],
            "title": f"{d} 半小时平均电流（A）"}


def inverter_chart(store: Store, inverter_id: str, d: str) -> Dict:
    rows = store.query("SELECT f.string_id, f.i_profile_json FROM daily_features f JOIN strings p ON p.string_id=f.string_id "
                       "WHERE p.inverter_id=? AND f.date=?", (inverter_id, d))
    series = [{"label": r.string_id.rsplit("-", 1)[-1], "data": json.loads(r.i_profile_json)} for r in rows.itertuples()]
    return {"labels": [slot_label(i) for i in range(48)], "series": series, "title": f"{inverter_id} 各串电流 {d}"}


def trend_data(store: Store, scope: str, target: str, d: str) -> Dict:
    start = (date.fromisoformat(d) - timedelta(days=45)).isoformat()
    if scope == "inverter":
        b = store.query("SELECT date, inv_dev dev, weather_tag FROM inverter_baselines WHERE inverter_id=? AND date BETWEEN ? AND ? ORDER BY date",
                        (target, start, d))
    else:
        b = store.query("SELECT date, dev_today dev, weather_tag FROM baselines WHERE string_id=? AND date BETWEEN ? AND ? ORDER BY date",
                        (target, start, d))
    return {"labels": b["date"].tolist(), "dev": [None if pd.isna(v) else round(float(v) * 100, 2) for v in b["dev"]],
            "weather": b["weather_tag"].tolist()}


def md_to_html(md: str) -> str:
    """极简 Markdown → HTML（只覆盖工单文本用到的语法：## 标题、**粗体**、- 列表、1. 列表、段落）"""
    out, in_ul, in_ol = [], False, False

    def close():
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    def inline(t: str) -> str:
        t = html.escape(t)
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)

    for line in md.splitlines():
        s = line.strip()
        if not s:
            close()
            continue
        if s.startswith("## "):
            close()
            out.append(f"<h3>{inline(s[3:])}</h3>")
        elif s.startswith("- "):
            if not in_ul:
                close()
                out.append("<ul>")
                in_ul = True
            out.append(f"<li>{inline(s[2:])}</li>")
        elif re.match(r"^\d+\. ", s):
            if not in_ol:
                close()
                out.append("<ol>")
                in_ol = True
            item = re.sub(r"^\d+\. ", "", s)          # Python 3.11 的 f-string 表达式里不能带反斜杠
            out.append(f"<li>{inline(item)}</li>")
        else:
            close()
            out.append(f"<p>{inline(s)}</p>")
    close()
    return "\n".join(out)
