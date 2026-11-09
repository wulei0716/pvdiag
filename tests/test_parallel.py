"""按站并行：多站并行跑基线/诊断，落库结果必须与串行完全一致"""
from datetime import date, timedelta

import pandas as pd

from pvdiag.agents.base import AgentContext
from pvdiag.agents.ingest import IngestAgent
from pvdiag.agents.profile import ProfileAgent
from pvdiag.config import Settings
from pvdiag.orchestrator import DailyPipeline
from pvdiag.parallel import effective_workers
from pvdiag.sim.synth import Fault, SimConfig, simulate
from pvdiag.storage import Store

START = date(2026, 9, 1)
DAYS = 25


def _build(tmp_path, name, workers):
    settings = Settings(data_dir=tmp_path / name)
    settings.runtime.workers = workers
    settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    end = START + timedelta(days=DAYS - 1)
    for k in range(3):
        sid = f"PX{k}"
        faults = [Fault("string", f"{sid}-INV01-pv{2 + k}", "shading_fixed", 8, {"start": "14:00", "end": "16:30", "depth": 0.35})]
        cfg = SimConfig(station_id=sid, n_inverters=2, strings_per_inverter=6, days=DAYS, seed=100 + k, start=START, faults=faults,
                        weather_probs=(0.75, 0.15, 0.10))
        data = simulate(cfg)
        ProfileAgent.import_stations(store, data["stations"])
        ProfileAgent.import_inverters(store, data["inverters"])
        ProfileAgent.import_strings(store, data["strings"])
        IngestAgent().ingest_frames(AgentContext(settings=settings, store=store, run_date=end), sid, data["samples_strings"], data["samples_inverters"])
    DailyPipeline(settings, store, ingest=False).backfill(end - timedelta(days=6), end)
    return store


def _dump(store):
    cols = {"baselines": "string_id, date, dev_today, persist_days_5, cold_start",
            "signatures": "string_id, date, ratio_mean, dip_depth, dip_start, dip_end",
            "suspects": "suspect_id, date, grade, rule_cause, persist_days"}
    return {t: store.query(f"SELECT {c} FROM {t} ORDER BY 1, 2").round(6).reset_index(drop=True) for t, c in cols.items()}


def test_effective_workers():
    s = Settings()
    s.runtime.workers = 4
    assert effective_workers(s, 1) == 1           # 单站不起进程池
    assert effective_workers(s, 2) == 2           # 不超过站数
    assert effective_workers(s, 10) == 4
    s.runtime.workers = 0
    assert effective_workers(s, 10) >= 1          # 0 = CPU 核数


def test_parallel_matches_serial(tmp_path):
    a = _dump(_build(tmp_path, "serial", workers=1))
    b = _dump(_build(tmp_path, "parallel", workers=3))
    assert not a["suspects"].empty and not a["signatures"].empty
    assert a["suspects"]["station_id" if "station_id" in a["suspects"] else "suspect_id"].str[:3].nunique() == 3
    for t in a:
        pd.testing.assert_frame_equal(a[t], b[t], check_dtype=False)
