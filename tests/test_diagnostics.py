"""诊断能力 v0.3：季节性朝向物理归一与档案校核、跟踪支架卡滞、站级积灰量化"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pvdiag.models import Cause
from pvdiag.orchestrator import DailyPipeline
from pvdiag.sim import SimConfig
from pvdiag.sim.synth import Fault
from pvdiag.solar import best_fit_orientation, expected_noon_ratio, expected_profile_ratio
from tests.conftest import load_sim


def _run(tmp_path, cfg, days_back=19):
    settings, store, data = load_sim(tmp_path, cfg)
    end = cfg.start + timedelta(days=cfg.days - 1)
    DailyPipeline(settings, store=store, ingest=False).backfill(end - timedelta(days=days_back), end)
    return settings, store


def test_solar_seasonal_expectation():
    # 东向 vs 南向：冬天差得多、夏天差得少；参考朝向自身恒为 1
    winter = expected_noon_ratio(31.2, date(2026, 1, 15), 90, 15, 180, 15)
    summer = expected_noon_ratio(31.2, date(2026, 7, 15), 90, 15, 180, 15)
    assert winter < summer < 1.0 and winter < 0.8
    assert expected_noon_ratio(31.2, date(2026, 4, 1), 180, 15, 180, 15) == 1.0
    curve = expected_profile_ratio(31.2, date(2026, 10, 15), 90, 15, 180, 15)
    fit = best_fit_orientation(31.2, date(2026, 10, 15), curve, 180, 15)
    assert fit[:2] == (90.0, 15.0) and fit[3] > fit[2] + 0.03
    # 经度换算：乌鲁木齐（87°E，北京时间）时钟正午太阳还在东边 → 东向串中午期望比明显高于 120°E 的站
    from pvdiag.solar import lon_offset_minutes
    off = lon_offset_minutes(87.0, 120.0)
    assert abs(off - 132) < 1e-6
    assert expected_noon_ratio(43.8, date(2026, 10, 15), 90, 15, 180, 15, lon_offset_min=off) > expected_noon_ratio(43.8, date(2026, 10, 15), 90, 15, 180, 15) + 0.1


def test_orientation_archive_correct_no_alarm_and_wrong_archive_flagged(tmp_path):
    s = lambda k, n: f"OR1-INV{k:02d}-pv{n}"
    faults = [Fault("string", s(1, 3), "azimuth_actual", 0, {"azimuth": 90, "tilt": 15, "archive_correct": True}),
              Fault("string", s(2, 3), "azimuth_actual", 0, {"azimuth": 90, "tilt": 15, "archive_correct": False})]
    cfg = SimConfig(station_id="OR1", n_inverters=2, strings_per_inverter=8, days=45, seed=9, start=date(2026, 10, 1), faults=faults,
                    weather_probs=(0.7, 0.15, 0.15))
    settings, store = _run(tmp_path, cfg)
    susp = store.query("SELECT string_id, grade, rule_cause, signature_json FROM suspects WHERE scope='string'")
    # 档案如实登记东向：物理归一后不应有 A/B/C 级嫌疑
    ok = susp[susp["string_id"] == s(1, 3)]
    assert ok.empty or set(ok["grade"]) <= {"E", "D"}, ok.to_dict("records")
    # 档案错写成南向：应被判为 朝向差异（C 级），并给出最像的朝向
    bad = susp[susp["string_id"] == s(2, 3)]
    assert not bad.empty and bad["rule_cause"].iloc[0] == Cause.ORIENTATION.value and bad["grade"].iloc[0] == "C", bad.to_dict("records")
    import json, re
    notes = " ".join(json.loads(bad["signature_json"].iloc[0])["notes"])
    m = re.search(r"方位 (\d+)°", notes)
    assert m and int(m.group(1)) <= 135, notes          # 指向东侧（阴天半权重时拟合可能落在 90~120°）
    # 其它健康串无 A/B
    healthy = susp[~susp["string_id"].isin([s(1, 3), s(2, 3)]) & susp["grade"].isin(["A", "B"])]
    assert healthy.empty, healthy.to_dict("records")


def test_tracker_stuck_detected_only_on_tracker_stations(tmp_path):
    s = lambda k, n: f"TR1-INV{k:02d}-pv{n}"
    cfg = SimConfig(station_id="TR1", n_inverters=2, strings_per_inverter=8, days=45, seed=4, start=date(2026, 9, 1), mount="tracker",
                    faults=[Fault("string", s(1, 5), "tracker_stuck", 30)], weather_probs=(0.7, 0.15, 0.15))
    settings, store = _run(tmp_path, cfg, days_back=14)
    susp = store.query("SELECT string_id, grade, rule_cause FROM suspects WHERE scope='string'")
    hit = susp[susp["string_id"] == s(1, 5)]
    assert not hit.empty and hit["rule_cause"].iloc[0] == Cause.TRACKER.value and hit["grade"].iloc[0] == "B", susp.to_dict("records")
    assert susp[(susp["string_id"] != s(1, 5)) & susp["grade"].isin(["A", "B"])].empty
    # 复核确认后生成工单，检查项带跟踪支架专用清单
    from pvdiag.agents.base import AgentContext
    from pvdiag.agents.review import ReviewAgent
    from pvdiag.agents.workorder import WorkOrderAgent
    sid = store.scalar("SELECT suspect_id FROM suspects WHERE string_id=?", (s(1, 5),))
    ReviewAgent.confirm(store, sid, None, "t")
    WorkOrderAgent().generate(AgentContext(settings=settings, store=store, run_date=cfg.start + timedelta(days=cfg.days - 1)))
    wo = store.query("SELECT checklist_json FROM workorders")
    assert not wo.empty and "跟踪" in wo["checklist_json"].iloc[0]


def test_dual_axis_tracker_stuck_in_winter(tmp_path):
    """双轴跟踪冬季卡滞：中午比值也掉到 0.85 左右，单轴的"中午≈1"规则不适用，要靠物理曲线校核认出来"""
    s = lambda k, n: f"TR2-INV{k:02d}-pv{n}"
    cfg = SimConfig(station_id="TR2", n_inverters=2, strings_per_inverter=8, days=45, seed=5, start=date(2026, 11, 20), mount="tracker_dual",
                    faults=[Fault("string", s(2, 3), "tracker_stuck", 30, {"stuck_tilt": 20.0})], weather_probs=(0.7, 0.15, 0.15))
    settings, store = _run(tmp_path, cfg, days_back=14)
    susp = store.query("SELECT string_id, grade, rule_cause FROM suspects WHERE scope='string'")
    hit = susp[susp["string_id"] == s(2, 3)]
    assert not hit.empty and hit["rule_cause"].iloc[0] == Cause.TRACKER.value and hit["grade"].iloc[0] == "B", susp.to_dict("records")
    assert susp[(susp["string_id"] != s(2, 3)) & susp["grade"].isin(["A", "B"])].empty
    sj = store.scalar("SELECT signature_json FROM suspects WHERE string_id=?", (s(2, 3),)) or ""
    assert "固定倾角" in sj, sj


def test_station_soiling_quantified(tmp_path):
    """全站一起积灰（互比盲区）：站级指标应给出积灰损失与清洗回本估算"""
    s = lambda k, n: f"SO1-INV{k:02d}-pv{n}"
    faults = [Fault("string", s(k, n), "dust_local", 10, {"rate_per_day": 0.006, "floor": 0.80, "rain_recover": 0.0})
              for k in (1, 2) for n in range(1, 9)]
    cfg = SimConfig(station_id="SO1", n_inverters=2, strings_per_inverter=8, days=45, seed=12, start=date(2026, 9, 1), faults=faults,
                    weather_probs=(0.8, 0.15, 0.05))
    settings, store = _run(tmp_path, cfg, days_back=5)
    end = cfg.start + timedelta(days=cfg.days - 1)
    import json
    rep = json.loads((settings.reports_dir / end.isoformat() / "report.json").read_text(encoding="utf-8"))
    tr = rep["region"]["station_trend"]
    assert tr and tr[0]["station_id"] == "SO1"
    t = tr[0]
    assert t["slope_pct_per_day"] < -0.2 and t["soiling_loss_pct"] >= 3, t
    assert t["lost_kwh_per_day"] > 0 and t["cleaning_cost_yuan"] > 0 and t["payback_days"] is not None
    # 互比看不出：全站同步积灰时不应有一堆 A/B 嫌疑
    susp = store.query("SELECT grade FROM suspects WHERE scope='string' AND grade IN ('A','B')")
    assert len(susp) <= 2


def test_station_compare_flags_whole_station_underperformance(tmp_path):
    """同区域 3 站：其中 1 站所有逆变器整机 -10%（互比盲区）→ 站间 PR 互比应标出该站；辐照无实测时用晴空模型"""
    from pvdiag.agents.base import AgentContext
    from pvdiag.agents.ingest import IngestAgent
    from pvdiag.agents.profile import ProfileAgent
    from pvdiag.config import Settings
    from pvdiag.sim import simulate
    from pvdiag.storage import Store
    settings = Settings(data_dir=tmp_path / "data"); settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    start = date(2026, 9, 1)
    for k in range(1, 4):
        sid = f"RG{k}"
        faults = [Fault("inverter", f"{sid}-INV{j:02d}", "mppt_fault", 20, {"factor": 0.90}) for j in (1, 2)] if k == 3 else []
        cfg = SimConfig(station_id=sid, n_inverters=2, strings_per_inverter=8, days=40, seed=20, start=start, faults=faults, use_default_faults=False,   # 同 seed = 同天气序列（同区域）
                        weather_probs=(0.8, 0.15, 0.05))
        data = simulate(cfg)
        ProfileAgent.import_stations(store, data["stations"]); ProfileAgent.import_inverters(store, data["inverters"]); ProfileAgent.import_strings(store, data["strings"])
        ctx = AgentContext(settings=settings, store=store, run_date=start + timedelta(days=39))
        IngestAgent().ingest_frames(ctx, sid, data["samples_strings"], data["samples_inverters"])
    end = start + timedelta(days=39)
    assert store.scalar("SELECT COUNT(*) FROM station_daily WHERE irradiance_source='clearsky'") > 0
    DailyPipeline(settings, store=store, ingest=False).backfill(end - timedelta(days=6), end)
    import json
    rep = json.loads((settings.reports_dir / end.isoformat() / "report.json").read_text(encoding="utf-8"))
    cmp_ = {r["station_id"]: r for r in rep["region"]["station_compare"]}
    assert set(cmp_) == {"RG1", "RG2", "RG3"}, cmp_
    assert cmp_["RG3"]["flag"] and cmp_["RG3"]["dev"] < -0.05, cmp_["RG3"]
    assert not cmp_["RG1"]["flag"] and not cmp_["RG2"]["flag"]
    assert any("站间互比" in a["message"] for a in rep["alerts"])


def test_power_only_devices_and_bifacial(tmp_path):
    """优化器/微逆：只有功率 → 等效电流；电压不可互比 → 不判电压异位。双面组件：档案登记后按增益归一不报高/不拉低别人"""
    from pvdiag.adapters.base import PlatformAdapter, POWER_ONLY_VIRTUAL_V
    df = pd.DataFrame({"string_id": ["X-pv1", "X-pv1"], "ts": ["2026-09-01 12:00", "2026-09-01 12:05"], "power_w": [400.0, 410.0]})
    n = PlatformAdapter.normalize_strings(df)
    assert abs(n["current_a"].iloc[0] - 400.0 / POWER_ONLY_VIRTUAL_V) < 1e-9 and n["voltage_v"].iloc[0] == POWER_ONLY_VIRTUAL_V
    # 同一串长档案错（电压 -9%）：组串型 → A 级档案差异；优化器型 → 不判电压异位
    s = lambda k, n: f"OP1-INV{k:02d}-pv{n}"
    for dev_type, expect_profile_error in (("string", True), ("optimizer", False)):
        cfg = SimConfig(station_id="OP1", n_inverters=1, strings_per_inverter=8, days=40, seed=31, start=date(2026, 9, 1), device_type=dev_type,
                        faults=[Fault("string", s(1, 3), "wrong_profile", 0, {"n_modules": 20}),
                                Fault("string", s(1, 6), "bifacial_actual", 0, {"gain": 0.08, "archive_correct": True})])
        settings, store = _run(tmp_path / dev_type, cfg, days_back=8)
        susp = store.query("SELECT string_id, grade, rule_cause FROM suspects WHERE scope='string'")
        has_pe = not susp[(susp["string_id"] == s(1, 3)) & (susp["rule_cause"] == Cause.PROFILE_ERROR.value)].empty
        assert has_pe == expect_profile_error, (dev_type, susp.to_dict("records"))
        # 双面串归一后偏差应接近 0（没有归一会 +8%），且不把别人拉成低发
        end = cfg.start + timedelta(days=cfg.days - 1)
        b = store.query("SELECT dev_today, orient_exp_ratio FROM baselines WHERE string_id=? AND date=?", (s(1, 6), end.isoformat()))
        assert abs(b["orient_exp_ratio"].iloc[0] - 1.08) < 1e-6 and abs(b["dev_today"].iloc[0]) < 0.03, b.to_dict("records")
        assert susp[(susp["string_id"] != s(1, 3)) & susp["grade"].isin(["A", "B"])].empty
