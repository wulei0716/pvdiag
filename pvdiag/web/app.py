"""阶段6 · 人工复核 Web 界面（FastAPI + Jinja2 + Chart.js）

一屏看全：嫌疑串曲线图（vs 群组中位）+ 偏差趋势 + 配置档案 + AI 原因假设 → 确认 / 驳回 / 修改原因
复核结果自动回写标注库（reviews），成为下一轮训练数据。工单页支持一键确认发送与现场回填。
启动：pvdiag serve --port 8080
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, Form, HTTPException, Request
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
from .svgchart import bar_chart, line_chart
from .templates import TEMPLATES


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.load()
    settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    env = Environment(loader=DictLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda x: ("-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):+.1%}")
    env.filters["num"] = lambda x, d=1: ("-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.{d}f}")
    app = FastAPI(title="pvdiag · 组串异常复核")

    def render(name: str, **kw) -> HTMLResponse:
        kw.setdefault("now", datetime.now().strftime("%Y-%m-%d %H:%M"))
        kw.setdefault("stations", store.stations()["station_id"].tolist())
        return HTMLResponse(env.get_template(name).render(**kw))

    def latest_date() -> Optional[str]:
        return store.scalar("SELECT MAX(date) FROM baselines")

    # ------------------------------------------------------------------ 首页看板
    @app.get("/", response_class=HTMLResponse)
    def index():
        d = latest_date()
        rep = {}
        if d:
            p = settings.reports_dir / d / "report.json"
            if p.exists():
                rep = json.loads(p.read_text(encoding="utf-8"))
        q = ReviewAgent.queue(store)
        by_grade = q["grade"].value_counts().to_dict() if not q.empty else {}
        wos = store.query("SELECT status, COUNT(*) n FROM workorders GROUP BY status")
        alerts = store.query("SELECT run_date, level, source, message FROM alerts WHERE resolved=0 ORDER BY alert_id DESC LIMIT 15")
        loss = LossAgent.ledger_summary(store)
        return render("index.html", date=d, report=rep, by_grade=by_grade, pending=len(q),
                      wos=wos.set_index("status")["n"].to_dict() if not wos.empty else {}, alerts=alerts.to_dict("records"),
                      loss=loss, board=rep.get("board", {}), stats=ReviewAgent.stats(store))

    # ------------------------------------------------------------------ 复核队列
    @app.get("/review", response_class=HTMLResponse)
    def review_list(station: Optional[str] = None, grade: Optional[str] = None):
        q = ReviewAgent.queue(store, station or None, [grade] if grade else None)
        return render("review.html", rows=q.to_dict("records"), station=station or "", grade=grade or "",
                      causes=CAUSE_CHOICES, reasons=REJECT_REASONS)

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

    # ------------------------------------------------------------------ 嫌疑详情
    @app.get("/suspect/{suspect_id}", response_class=HTMLResponse)
    def suspect_detail(suspect_id: str):
        s = store.query("SELECT s.*, v.cause ai_cause, v.confidence, v.action, v.explanation, v.model_version, v.proba_json "
                        "FROM suspects s LEFT JOIN verdicts v ON v.suspect_id=s.suspect_id WHERE s.suspect_id=?", (suspect_id,))
        if s.empty:
            raise HTTPException(404, "嫌疑不存在")
        s = s.iloc[0].to_dict()
        sig = json.loads(s.get("signature_json") or "{}")
        chart = _curve_chart(store, s)
        trend = _trend_chart(store, s)
        prof = {}
        if s["scope"] == "string":
            p = store.query("SELECT * FROM strings WHERE string_id=?", (s["string_id"],))
            prof = p.iloc[0].to_dict() if not p.empty else {}
        reviews = store.query("SELECT * FROM reviews WHERE suspect_id=? ORDER BY reviewed_at DESC", (suspect_id,))
        loss = LossAgent.case_loss(store, suspect_id)
        wo = store.query("SELECT wo_id, status FROM workorders WHERE suspect_ids_json LIKE ?", (f"%{suspect_id}%",))
        colors = ["#0f766e" if w == "clear" else ("#f59e0b" if w == "cloudy" else "#9ca3af") for w in trend["weather"]]
        chart_svg = line_chart(chart["labels"], chart["series"], chart["title"], "A")
        trend_svg = bar_chart(trend["labels"], trend["dev"], colors, "互比偏差 %（红虚线 -3% / -5%）")
        return render("suspect.html", s=s, sig=sig, chart_svg=chart_svg, trend_svg=trend_svg,
                      prof=prof, reviews=reviews.to_dict("records"), loss=loss, causes=CAUSE_CHOICES, reasons=REJECT_REASONS,
                      proba=json.loads(s.get("proba_json") or "{}"), wo=wo.to_dict("records"))

    # ------------------------------------------------------------------ 工单
    @app.get("/workorders", response_class=HTMLResponse)
    def workorders(status: Optional[str] = None):
        sql = "SELECT * FROM workorders"
        params = []
        if status:
            sql += " WHERE status=?"
            params.append(status)
        sql += " ORDER BY CASE status WHEN 'draft' THEN 0 WHEN 'reopened' THEN 1 WHEN 'sent' THEN 2 WHEN 'done' THEN 3 ELSE 4 END, priority DESC"
        df = store.query(sql, params)
        return render("workorders.html", rows=df.to_dict("records"), status=status or "")

    @app.get("/workorder/{wo_id}", response_class=HTMLResponse)
    def workorder_detail(wo_id: str):
        wo = store.query("SELECT * FROM workorders WHERE wo_id=?", (wo_id,))
        if wo.empty:
            raise HTTPException(404)
        wo = wo.iloc[0].to_dict()
        return render("workorder.html", wo=wo, text=WorkOrderAgent.render(store, wo), suspect_ids=json.loads(wo["suspect_ids_json"]))

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

    # ------------------------------------------------------------------ JSON API（供下游/自有工单系统消费）
    @app.get("/api/queue")
    def api_queue(station: Optional[str] = None):
        return JSONResponse(json.loads(ReviewAgent.queue(store, station).to_json(orient="records", force_ascii=False)))

    @app.get("/api/report/{d}")
    def api_report(d: str):
        p = settings.reports_dir / d / "report.json"
        if not p.exists():
            raise HTTPException(404)
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))

    @app.get("/api/suspect/{suspect_id}/chart")
    def api_chart(suspect_id: str):
        s = store.query("SELECT * FROM suspects WHERE suspect_id=?", (suspect_id,))
        if s.empty:
            raise HTTPException(404)
        return JSONResponse(_curve_chart(store, s.iloc[0].to_dict()))

    return app


# ---------------------------------------------------------------------- 图表数据
def _curve_chart(store: Store, s: dict) -> dict:
    """今天该串半小时电流曲线 vs 群组中位曲线（整机级则各串曲线）"""
    d = s["date"]
    labels = [slot_label(i) for i in range(48)]
    if s["scope"] == "inverter":
        rows = store.query("SELECT f.string_id, f.i_profile_json FROM daily_features f JOIN strings p ON p.string_id=f.string_id "
                           "WHERE p.inverter_id=? AND f.date=?", (s["inverter_id"], d))
        series = [{"label": r.string_id, "data": json.loads(r.i_profile_json)} for r in rows.itertuples()]
        return {"labels": labels, "series": series, "title": f"{s['inverter_id']} 各串电流 {d}"}
    sid = s["string_id"]
    me = store.query("SELECT i_profile_json FROM daily_features WHERE string_id=? AND date=?", (sid, d))
    bl = store.query("SELECT group_key, group_level FROM baselines WHERE string_id=? AND date=?", (sid, d))
    peers = pd.DataFrame()
    if not bl.empty:
        key, level = bl["group_key"].iloc[0], bl["group_level"].iloc[0]
        peers = store.query(
            "SELECT f.i_profile_json FROM daily_features f JOIN strings p ON p.string_id=f.string_id WHERE f.date=? AND f.string_id!=? "
            "AND f.zero_all_day=0 AND f.stuck=0 AND " + ("p.group_key=?" if level == "group" else "p.inverter_id=?"),
            (d, sid, key if level == "group" else s["inverter_id"]))
    mine = json.loads(me["i_profile_json"].iloc[0]) if not me.empty else [None] * 48
    gmed = [None] * 48
    if len(peers) >= 2:
        M = np.array([[np.nan if v is None else v for v in json.loads(x)] for x in peers["i_profile_json"]], dtype=float)
        med = np.nanmedian(M, axis=0)
        gmed = [None if np.isnan(v) else round(float(v), 3) for v in med]
    return {"labels": labels, "series": [{"label": f"{sid}", "data": mine}, {"label": "群组中位", "data": gmed}],
            "title": f"{sid} 电流曲线 {d}（半小时均值，A）"}


def _trend_chart(store: Store, s: dict) -> dict:
    d = date.fromisoformat(s["date"])
    start = (d - timedelta(days=45)).isoformat()
    if s["scope"] == "inverter":
        b = store.query("SELECT date, inv_dev dev, weather_tag FROM inverter_baselines WHERE inverter_id=? AND date BETWEEN ? AND ? ORDER BY date",
                        (s["inverter_id"], start, s["date"]))
    else:
        b = store.query("SELECT date, dev_today dev, weather_tag FROM baselines WHERE string_id=? AND date BETWEEN ? AND ? ORDER BY date",
                        (s["string_id"], start, s["date"]))
    return {"labels": b["date"].tolist(),
            "dev": [None if pd.isna(v) else round(float(v) * 100, 2) for v in b["dev"]],
            "weather": b["weather_tag"].tolist()}
