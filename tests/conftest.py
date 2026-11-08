import logging
from datetime import date, timedelta
from pathlib import Path

import pytest

from pvdiag.agents.base import AgentContext
from pvdiag.agents.ingest import IngestAgent
from pvdiag.agents.profile import ProfileAgent
from pvdiag.config import Settings
from pvdiag.sim import SimConfig, simulate
from pvdiag.sim.synth import Fault
from pvdiag.storage import Store

logging.getLogger("pvdiag").setLevel(logging.WARNING)


def small_sim(days: int = 50, faults=None, seed: int = 11) -> SimConfig:
    cfg = SimConfig(station_id="T01", n_inverters=2, strings_per_inverter=8, days=days, seed=seed,
                    start=date(2026, 9, 1), rated_ac_kw=80.0)
    if faults is not None:
        cfg.faults = faults
    return cfg


def load_sim(tmp_path: Path, cfg: SimConfig):
    data = simulate(cfg)
    s = Settings(data_dir=tmp_path / "data")
    s.ensure_dirs()
    store = Store(s.db_path, s.samples_dir)
    ProfileAgent.import_stations(store, data["stations"])
    ProfileAgent.import_inverters(store, data["inverters"])
    ProfileAgent.import_strings(store, data["strings"])
    ctx = AgentContext(settings=s, store=store, run_date=cfg.start + timedelta(days=cfg.days - 1))
    IngestAgent().ingest_frames(ctx, cfg.station_id, data["samples_strings"], data["samples_inverters"])
    return s, store, data


@pytest.fixture
def sim_env(tmp_path):
    inv = lambda k: f"T01-INV{k:02d}"
    s = lambda k, n: f"{inv(k)}-pv{n}"
    faults = [
        Fault("string", s(1, 2), "open_circuit", 32),
        Fault("string", s(1, 4), "diode_bypass", 31, {"start": "10:00", "end": "13:00", "depth": 0.33}),
        Fault("string", s(1, 6), "shading_fixed", 30, {"start": "14:30", "end": "17:30", "depth": 0.5}),
        Fault("string", s(2, 1), "orientation", 0, {"clear_factor": 0.94}),
        Fault("string", s(2, 3), "dust_local", 15, {"rate_per_day": 0.006, "floor": 0.80, "rain_recover": 0.5}),
        Fault("string", s(2, 5), "stuck", 34, {"value": 8.0}),
        Fault("string", s(2, 7), "wrong_profile", 0, {"n_modules": 20}),
        Fault("string", s(2, 8), "archived_shading", 0, {"start": "09:30", "end": "11:00", "depth": 0.3}),
        Fault("string", s(1, 7), "outage_intermittent", 33, {"prob": 0.7, "hours": 2.0}),
        Fault("string", s(2, 6), "stale_partial", 35, {"start": "10:30", "hours": 3.0}),
    ]
    cfg = small_sim(days=50, faults=faults)
    return load_sim(tmp_path, cfg) + (cfg,)
