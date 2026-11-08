"""自主运行：自调参（影子回放/生效/回滚/覆盖层）、派单门控滞回、模型生命周期（留出集考试/上线/线上降级）、
数据接入自愈（缺口补拉/重试/降级/升级告警）、健康自检、关键事件推送去重、控制台自主页。"""
import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from pvdiag.agents.base import AgentContext
from pvdiag.agents.classify import ClassifyAgent
from pvdiag.agents.ingest import IngestAgent
from pvdiag.agents.review import ReviewAgent
from pvdiag.agents.tuning import TuningAgent, apply_overrides, get_param
from pvdiag.agents.workorder import WorkOrderAgent
from pvdiag.config import Settings
from pvdiag.health import run_health_checks, summarize
from pvdiag.models import Cause
from pvdiag.notify import Notifier
from pvdiag.orchestrator import DailyPipeline
from pvdiag.sim.synth import Fault, write_csv_bundle
from pvdiag.storage import Store
from tests.conftest import load_sim, small_sim


def _ctx(settings, store, d):
    return AgentContext(settings=settings, store=store, run_date=d)


# ---------------------------------------------------------------------------- 自调参
def test_shadow_replay_prefers_noise_gating():
    """构造考题：确认案例 = 真低发（-8%，噪声小）；驳回案例 = 云影日 -6%（群组噪声 3.5%）。
    当前 noise_k=2.5 会把云影日算成低发 → 误报；候选 noise_k=3.0 应把它过滤掉而不丢真低发。"""
    agent = TuningAgent()
    quiet = np.full(8, 0.012)
    cases = []
    for i in range(12):   # 真低发：-8% 持续
        cases.append({"label": 1, "devs": np.full(8, -0.08), "noises": quiet})
    for i in range(10):   # 云影误报：两天 -0.085/-0.09，群组噪声 0.034（2.5×0.034=0.085 刚好算显著；3.0×0.034=0.102 不算）
        cases.append({"label": 0, "devs": np.array([-0.01, -0.085, -0.09, -0.01, 0.0, 0.01, 0.0, -0.01]),
                      "noises": np.array([0.012, 0.034, 0.034, 0.012, 0.012, 0.012, 0.012, 0.012])})
    cur = agent.evaluate(cases, -0.05, 2, 2.5, 0.04)
    better = agent.evaluate(cases, -0.05, 2, 3.0, 0.04)
    assert cur["fp"] == 10 and cur["recall"] == 1.0
    assert better["fp"] == 0 and better["recall"] == 1.0 and better["f1"] > cur["f1"]


def test_tuning_grid_pick_and_apply_with_overrides(tmp_path):
    settings = Settings(data_dir=tmp_path / "d")
    settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    ctx = _ctx(settings, store, date(2026, 10, 30))
    agent = TuningAgent()
    cases = [{"label": 1, "devs": np.full(8, -0.08), "noises": np.full(8, 0.012)} for _ in range(12)] + \
            [{"label": 0, "devs": np.array([-0.01, -0.085, -0.09, -0.01, 0.0, 0.01, 0.0, -0.01]),
              "noises": np.array([0.012, 0.034, 0.034, 0.012, 0.012, 0.012, 0.012, 0.012])} for _ in range(10)]
    cur = TuningAgent.current_params(settings)
    table = agent.evaluate_grid(ctx, cases, cur)
    cur_row = next(r for r in table if r["is_current"])
    best = agent.pick(table, cur_row, settings.autotune)
    assert best is not None and best["n_changes"] == 1 and best["params"]["thresholds.noise_k"] == 3.0
    # 生效 → 覆盖层 + 留痕 + 当前 settings 立即更新
    desc = agent.apply(ctx, best, cur_row)
    assert "noise_k" in desc and settings.thresholds.noise_k == 3.0
    assert store.scalar("SELECT COUNT(*) FROM param_overrides") == 1
    assert store.scalar("SELECT status FROM param_changes ORDER BY change_id DESC LIMIT 1") == "applied"
    # 新加载的 Settings 叠加覆盖层后口径一致；config.yaml 本身不变
    fresh = Settings(data_dir=tmp_path / "d")
    assert fresh.thresholds.noise_k == 2.5
    assert get_param(apply_overrides(fresh, store), "thresholds.noise_k") == 3.0
    # 冷却期：再跑一次 run → 不会重复生效
    last = store.get_state("tuning:last_change")
    assert last["at"] == "2026-10-30"


def test_rollback_when_confirm_rate_drops(tmp_path):
    settings = Settings(data_dir=tmp_path / "d")
    settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    # 伪造一条 4 天前生效的调整（生效前 B 级确认率 80%）
    applied_on = date(2026, 10, 26)
    store.execute("INSERT OR REPLACE INTO param_overrides VALUES (?,?,?,?)", ("thresholds.fault_floor", json.dumps(-0.06), datetime.now().isoformat(), "autotune"))
    store.execute("INSERT INTO param_changes (run_date, param, old_value, new_value, reason, metrics_json, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                  (applied_on.isoformat(), "thresholds.fault_floor", json.dumps(-0.05), json.dumps(-0.06), "test",
                   json.dumps({"before_confirm_rate_b": 0.8}), "applied", datetime.combine(applied_on, datetime.min.time()).isoformat()))
    # 生效后 12 条 B 级复核，只有 4 条确认 → 33%，跌幅 > 10 个百分点
    for i in range(12):
        sid = f"S{i}@2026-10-27"
        store.execute("INSERT INTO suspects (suspect_id, string_id, inverter_id, station_id, date, first_date, scope, grade, rule_cause, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (sid, f"S{i}", "I", "T", "2026-10-28", "2026-10-27", "string", "B", Cause.UNKNOWN.value, "reviewing", datetime.now().isoformat()))
        store.execute("INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                      (sid, "t", "confirm" if i < 4 else "reject", None, None, "", "2026-10-29T10:00:00"))
    settings = apply_overrides(settings, store)
    assert settings.thresholds.fault_floor == -0.06
    rb = TuningAgent().check_rollback(_ctx(settings, store, date(2026, 10, 30)))
    assert rb["rolled_back"] and rb["param"] == "thresholds.fault_floor" and rb["old_value"] == -0.05
    assert settings.thresholds.fault_floor == -0.05
    assert json.loads(store.scalar("SELECT value_json FROM param_overrides WHERE param='thresholds.fault_floor'")) == -0.05
    assert store.scalar("SELECT status FROM param_changes WHERE change_id=1") == "rolled_back"


def test_tuning_agent_runs_in_pipeline_with_few_labels(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    rep = DailyPipeline(settings, store=store, ingest=False).run(end)
    assert rep["agents"]["tuning"]["status"] == "ok"
    assert "不调参" in rep["tuning"]["decision"]
    assert "自主运行" in (settings.reports_dir / end.isoformat() / "report.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------- 派单门控
def _seed_reviews(store, n: int, fa_rate: float, when: datetime):
    for i in range(n):
        sid = f"G{i}@{when:%Y-%m-%d}"
        store.execute("INSERT OR REPLACE INTO suspects (suspect_id, string_id, inverter_id, station_id, date, first_date, scope, grade, rule_cause, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (sid, f"G{i}", "I", "T01", when.date().isoformat(), when.date().isoformat(), "string", "B", Cause.UNKNOWN.value, "confirmed", when.isoformat()))
        store.execute("INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                      (sid, "t", "reject" if i < int(n * fa_rate) else "confirm", None, None, "", when.isoformat()))


def test_gate_opens_after_streak_and_closes_on_failure(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    settings.dispatch.mode = "gated"
    settings.dispatch.gate_open_streak_days = 3
    agent = WorkOrderAgent()
    # 门控条件不满足（无复核）→ draft
    eff, st = agent.effective_mode(_ctx(settings, store, end))
    assert eff == "draft" and st["streak"] == 0
    # 40 条复核、误报 5% → 门控通过；连续 3 个运行日后放开
    _seed_reviews(store, 40, 0.05, datetime.now() - timedelta(days=1))
    for k in range(3):
        eff, st = agent.effective_mode(_ctx(settings, store, end + timedelta(days=k)))
    assert eff == "auto" and st["streak"] == 3 and any("放开" in h["event"] for h in st["history"])
    # 当天某智能体失败 → 立即收回
    d = end + timedelta(days=3)
    store.log_run(d, "detect", "fail", datetime.now(), {}, "boom")
    eff, st = agent.effective_mode(_ctx(settings, store, d))
    assert eff == "draft" and st["streak"] == 0 and any("收回" in h["event"] for h in st["history"])
    # mode=draft 永远草稿
    settings.dispatch.mode = "draft"
    assert agent.effective_mode(_ctx(settings, store, d + timedelta(days=1)))[0] == "draft"


# ---------------------------------------------------------------------------- 模型生命周期
def test_model_exam_promote_and_live_demote(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    DailyPipeline(settings, store=store, ingest=False).backfill(end - timedelta(days=15), end)
    ctx = _ctx(settings, store, end)
    settings.confidence.min_labels_for_ml = 6
    susp = store.query("SELECT suspect_id, rule_cause, grade, date FROM suspects WHERE grade != '已备案' ORDER BY date")
    for i, r in enumerate(susp.to_dict("records")):
        (ReviewAgent.reject if i % 3 == 0 else ReviewAgent.confirm)(store, r["suspect_id"], *(("云影", "t") if i % 3 == 0 else (None, "t")))
    rv = store.query("SELECT * FROM reviews")
    for _ in range(3):
        for r in rv.to_dict("records"):
            store.execute("INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                          (r["suspect_id"], "dup", r["decision"], r["true_cause"], r["reject_reason"], "", r["reviewed_at"]))
    # 触发条件：标注够了且无模型
    due, why = ClassifyAgent.retrain_due(ctx)
    assert due and "首次" in why
    res = ClassifyAgent().train(ctx, reason=why)
    assert res["trained"] and "exam" in res and res["exam"]["rule_proxy"]["n"] > 0
    row = store.query("SELECT * FROM model_registry")
    assert len(row) == 1 and row["status"].iloc[0] in ("active", "rejected")
    assert json.loads(row["holdout_json"].iloc[0])["method"] in ("time_holdout", "cv")
    # 刚训练完 → 不再触发
    assert ClassifyAgent.retrain_due(ctx)[0] is False
    # 线上监督：伪造 30 条"模型全错、规则全对"的判定 → 自动降级回规则代理
    if row["active"].iloc[0] == 1:
        mv = row["model_version"].iloc[0]
        labeled = store.query("SELECT DISTINCT r.suspect_id, s.rule_cause, r.decision FROM reviews r JOIN suspects s ON s.suspect_id=r.suspect_id")
        for r in labeled.to_dict("records"):
            wrong = Cause.MPPT.value if r["decision"] != "reject" else Cause.OPEN_CIRCUIT.value
            store.execute("INSERT OR REPLACE INTO verdicts (suspect_id, model_version, cause, confidence, action, explanation, proba_json, novelty_score, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                          (r["suspect_id"], mv, wrong, 0.9, "转人工复核", "", "{}", None, datetime.now().isoformat()))
        store.execute("UPDATE reviews SET reviewed_at=?", (datetime.now().isoformat(),))
        if len(labeled) >= settings.confidence.demote_min_samples:
            demoted, why = ClassifyAgent.live_supervision(ctx, mv)
            assert demoted, why
            assert store.scalar("SELECT active FROM model_registry WHERE model_version=?", (mv,)) == 0
            assert ClassifyAgent()._load_active_model(ctx) is None


# ---------------------------------------------------------------------------- 数据接入自愈
def test_ingest_heals_gaps_and_retries_then_degrades(tmp_path, monkeypatch):
    cfg = small_sim(days=20, faults=[], seed=3)
    settings, store, data = load_sim(tmp_path, cfg)
    root = tmp_path / "import"
    write_csv_bundle(root, data, cfg.station_id)
    end = cfg.start + timedelta(days=cfg.days - 1)
    # 挖掉两天明细 → 自愈补拉
    for k in (5, 9):
        (settings.samples_dir / "string" / cfg.station_id / f"{(cfg.start + timedelta(days=k)).isoformat()}.parquet").unlink()
        (settings.samples_dir / "inverter" / cfg.station_id / f"{(cfg.start + timedelta(days=k)).isoformat()}.parquet").unlink()
    store.execute("UPDATE stations SET platform='csv', extra_json=?", (json.dumps({"platform_config": {"root": str(root)}}),))
    res = IngestAgent()(_ctx(settings, store, end))
    assert res.metrics["healed_gap_days"] == 2 and any("缺口" in a for a in res.alerts)
    assert len(store.available_dates("string", cfg.station_id)) == cfg.days
    # 平台拉取总失败：重试 → 降级读本地文件 → 不算失败；无降级时连续失败升级 error
    import pvdiag.agents.ingest as ing
    real_build = ing.build_adapter

    class Boom:
        station_id = cfg.station_id

        def fetch(self, a, b):
            raise RuntimeError("401 token expired")

    def fake_build(platform, station_id, config, store=None):
        return Boom() if platform == "huawei" else real_build(platform, station_id, config, store)
    monkeypatch.setattr(ing, "build_adapter", fake_build)
    settings.ingest.retries, settings.ingest.backoff_seconds = 2, 0.0
    store.execute("UPDATE stations SET platform='huawei', extra_json=?", (json.dumps({"platform_config": {"user": "x", "fallback_csv_root": str(root)}}),))
    res = IngestAgent(backfill_days=1)(_ctx(settings, store, end))
    assert any("降级读取本地文件" in a for a in res.alerts) and res.metrics["rows_written"] > 0
    assert store.get_state(f"ingest:{cfg.station_id}:failures") == 0
    store.execute("UPDATE stations SET extra_json=?", (json.dumps({"platform_config": {"user": "x"}}),))
    for _ in range(3):
        res = IngestAgent(backfill_days=1)(_ctx(settings, store, end))
    assert store.get_state(f"ingest:{cfg.station_id}:failures") == 3
    assert store.scalar("SELECT COUNT(*) FROM alerts WHERE level='error' AND source='ingest'") >= 1


# ---------------------------------------------------------------------------- 自检 / 推送 / 控制台
def test_health_checks_and_notifier(sim_env):
    settings, store, data, cfg = sim_env
    end = cfg.start + timedelta(days=cfg.days - 1)
    DailyPipeline(settings, store=store, ingest=False).run(end)
    checks = run_health_checks(settings, store, end)
    sm = summarize(checks)
    assert sm["fail"] == 0 and store.scalar("SELECT COUNT(*) FROM health_checks WHERE run_date=?", (end.isoformat(),)) == len(checks)
    n = Notifier(settings, store)
    out = n.send(end, "测试", ["[error] 管道断了"], level="error")
    assert out and out.startswith("outbox:") and (settings.outbox_dir / "alerts" / f"{end.isoformat()}.md").exists()
    assert n.send(end, "测试", ["[error] 管道断了"], level="error") is None         # 24h 去重
    assert n.send(end, "测试", ["[warn] 小事"], level="warn") is None                # 低于门槛
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from pvdiag.web.app import create_app
    c = TestClient(create_app(settings))
    r = c.get("/autonomy")
    assert r.status_code == 200 and "自主运行" in r.text and "派单门控" in r.text and "健康自检" in r.text
