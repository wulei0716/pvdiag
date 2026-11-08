"""端到端：合成故障 → 召回/误报；闭环：确认 → 工单 → 回填 → 复查归零；训练：标注 → 模型"""
from datetime import date, timedelta

import pandas as pd

from pvdiag.agents.base import AgentContext
from pvdiag.agents.baseline import BaselineAgent
from pvdiag.agents.classify import ClassifyAgent
from pvdiag.agents.detect import DetectAgent
from pvdiag.agents.loss import LossAgent
from pvdiag.agents.review import ReviewAgent
from pvdiag.agents.workorder import WorkOrderAgent
from pvdiag.models import Cause, Grade
from pvdiag.orchestrator import DailyPipeline
from pvdiag.sim.synth import Fault
from tests.conftest import load_sim, small_sim

EXPECTED_CAUSE = {
    "open_circuit": {Cause.OPEN_CIRCUIT.value},
    "diode_bypass": {Cause.BYPASS_DIODE.value},
    "shading_fixed": {Cause.SHADING.value},
    "orientation": {Cause.ORIENTATION.value},
    "dust_local": {Cause.DEGRADATION_DUST.value, Cause.UNKNOWN.value},   # 规则层可能只到 D/E；积灰由区域智能体的签名识别
    "stuck": {Cause.DATA_ISSUE.value},
    "wrong_profile": {Cause.PROFILE_ERROR.value},
    "archived_shading": {Cause.SHADING.value},
    "mppt_fault": {Cause.MPPT.value},
    "outage_intermittent": {Cause.OUTAGE.value},
    "stale_partial": {Cause.DATA_ISSUE.value},
}


def run_days(settings, store, start: date, end: date, light=True):
    p = DailyPipeline(settings, store=store, ingest=False)
    if light:
        p.backfill(start, end)          # 中间日轻量链路，最后一天完整链路 + 报告
    else:
        d = start
        while d <= end:
            p.run(d)
            d += timedelta(days=1)
    return AgentContext(settings=settings, store=store, run_date=end)


def test_recall_and_false_alarms(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    run_days(settings, store, end - timedelta(days=19), end)
    susp = store.query("SELECT * FROM suspects")
    assert not susp.empty
    truth = data["truth"]
    by_target = {}
    for r in susp.to_dict("records"):
        key = r["string_id"] if r["scope"] == "string" else r["inverter_id"]
        by_target.setdefault(key, []).append(r)
    # 召回：每个注入故障都被发现，且原因分型正确
    for t in truth.to_dict("records"):
        hits = by_target.get(t["target"], [])
        assert hits, f"漏报 {t['target']} ({t['kind']})"
        causes = {h["rule_cause"] for h in hits}
        assert causes & EXPECTED_CAUSE[t["kind"]], f"{t['target']} {t['kind']} 原因分型 {causes}"
        if t["kind"] == "archived_shading":
            assert all(h["grade"] == Grade.ARCHIVED.value for h in hits), "已备案遮挡应被抑制"
    # 误报：健康串不得出现 A/B 级嫌疑
    faulty = set(truth["target"])
    healthy_ab = susp[(susp["scope"] == "string") & (~susp["string_id"].isin(faulty)) & (susp["grade"].isin(["A", "B"]))]
    assert healthy_ab.empty, f"健康串误报: {healthy_ab['string_id'].tolist()}"
    # 案例连续性：每个故障串只有一个案例
    counts = susp[susp["scope"] == "string"].groupby("string_id").size()
    assert (counts <= 1).all(), counts[counts > 1]
    # 报告落盘 + 区域智能体识别出积灰签名（慢渐变 + 雨后恢复）
    rep_dir = settings.reports_dir / end.isoformat()
    assert (rep_dir / "report.md").exists() and (rep_dir / "report.json").exists()
    import json
    rep = json.loads((rep_dir / "report.json").read_text(encoding="utf-8"))
    dust_target = truth[truth["kind"] == "dust_local"]["target"].iloc[0]
    dust_hits = [c for c in rep["region"]["dust_candidates"] if c["string_id"] == dust_target]
    assert dust_hits, f"区域智能体未识别积灰串 {dust_target}: {rep['region']['dust_candidates']}"
    assert "积灰" in dust_hits[0]["pattern"]


def test_loss_positive_for_open_circuit(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    run_days(settings, store, end - timedelta(days=15), end)
    sid = store.scalar("SELECT suspect_id FROM suspects WHERE rule_cause=?", (Cause.OPEN_CIRCUIT.value,))
    loss = LossAgent.case_loss(store, sid)
    assert loss["lost_kwh"] > 50, loss        # 12 kWp 串断线 ≥ 两周，损失应达数百 kWh
    assert loss["lost_yuan"] > 0
    # 二极管凹陷在中午窗口外的遮挡串也应有正损失
    sh = store.scalar("SELECT suspect_id FROM suspects WHERE rule_cause=?", (Cause.SHADING.value,))
    assert LossAgent.case_loss(store, sh)["lost_kwh"] > 0


def test_closed_loop_confirm_workorder_recheck(tmp_path):
    """故障第 30 天起，第 42 天修复：确认 → 工单 → 回填 → 复查归零 → 案例关闭"""
    s = lambda k, n: f"T01-INV{k:02d}-pv{n}"
    faults = [Fault("string", s(1, 3), "open_circuit", 30, end_day=42)]
    cfg = small_sim(days=50, faults=faults, seed=5)
    settings, store, data = load_sim(tmp_path, cfg)
    d0 = cfg.start
    # 跑到第 40 天：应出现 A 级断线案例并生成工单草稿
    ctx = run_days(settings, store, d0 + timedelta(days=25), d0 + timedelta(days=40), light=False)
    susp = store.query("SELECT * FROM suspects WHERE string_id=?", (s(1, 3),))
    assert len(susp) == 1 and susp["grade"].iloc[0] == "A"
    wos = store.query("SELECT * FROM workorders")
    assert len(wos) == 1 and wos["status"].iloc[0] == "draft"
    wo_id = wos["wo_id"].iloc[0]
    # 一键确认发送（dryrun 写 outbox）
    out = WorkOrderAgent().send(ctx, wo_id)
    assert out.startswith("written:")
    assert (settings.outbox_dir / f"{wo_id}.md").exists()
    # 现场回填：第 42 天维修
    WorkOrderAgent.close(store, wo_id, "MC4 接头烧蚀", "维修", done_date=d0 + timedelta(days=42), recheck_clear_days=3)
    assert store.scalar("SELECT status FROM workorders WHERE wo_id=?", (wo_id,)) == "done"
    assert store.scalar("SELECT COUNT(*) FROM events WHERE target_id=?", (s(1, 3),)) == 1
    # 继续跑到第 49 天：复查应 verified，案例 closed，基线分段从维修日重置
    run_days(settings, store, d0 + timedelta(days=41), d0 + timedelta(days=49), light=False)
    assert store.scalar("SELECT status FROM workorders WHERE wo_id=?", (wo_id,)) == "verified"
    assert store.scalar("SELECT status FROM suspects WHERE string_id=?", (s(1, 3),)) == "closed"
    seg = store.scalar("SELECT segment_start FROM baselines WHERE string_id=? AND date=?", (s(1, 3), (d0 + timedelta(days=49)).isoformat()))
    assert seg == (d0 + timedelta(days=42)).isoformat()


def test_review_labels_train_model(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    ctx = run_days(settings, store, end - timedelta(days=15), end)
    settings.confidence.min_labels_for_ml = 10
    susp = store.query("SELECT suspect_id, rule_cause, grade FROM suspects WHERE grade != '已备案'")
    # 人工复核：确认一部分、驳回一部分（制造两类标注）
    for i, r in enumerate(susp.to_dict("records")):
        if i % 3 == 0:
            ReviewAgent.reject(store, r["suspect_id"], "云影", "tester")
        else:
            ReviewAgent.confirm(store, r["suspect_id"], None, "tester")
    # 复制标注以达到最小样本（仅为测试训练链路）
    rv = store.query("SELECT * FROM reviews")
    for _ in range(3):
        for r in rv.to_dict("records"):
            store.execute("INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                          (r["suspect_id"], "dup", r["decision"], r["true_cause"], r["reject_reason"], "", r["reviewed_at"]))
    X, y = ClassifyAgent.training_set(ctx)
    assert len(y) >= 1 and y.nunique() >= 2
    res = ClassifyAgent().train(ctx, force=True)
    assert res["trained"], res
    assert store.scalar("SELECT COUNT(*) FROM model_registry WHERE active=1") == 1
    # 再跑一次判定：应使用 ML 模型版本
    r = ClassifyAgent()(ctx)
    assert r.metrics["model"] != "rule-proxy-v1" and r.metrics["model"].split("-")[0] in ("rf", "hgb")
    st = ReviewAgent.stats(store, days=365)
    assert st["n_reviewed"] > 0 and 0 <= st["false_alarm_rate"] <= 1
