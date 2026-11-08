"""单元：配额规划、天气标记、平顶检测、基线流式统计、CSV 宽表拆分"""
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from pvdiag.adapters import QuotaPlanner
from pvdiag.adapters.csv_adapter import CsvAdapter
from pvdiag.agents.baseline import BaselineAgent
from pvdiag.config import BaselineCfg, Settings
from pvdiag.features import detect_flat_top, tag_weather


def test_quota_planner_matches_spec():
    # 日限额 = Σ⌈设备数/100⌉ + 24；20 台 → 1 + 24 = 25
    q = QuotaPlanner(n_devices=20)
    assert q.daily_quota == 25
    plan = q.plan_backfill(date(2026, 1, 1), date(2026, 3, 31))      # 90 天 / 4 天一批 = 23 批
    assert plan["n_batches"] == 23
    assert plan["calendar_days_needed"] >= 1
    # 250 台 → 3 个调用/批，日限额 27
    q2 = QuotaPlanner(n_devices=250)
    assert q2.units == 3 and q2.daily_quota == 27


def test_weather_tagging():
    cfg = BaselineCfg()
    assert tag_weather(300, 0.02, 330, cfg) == "clear"
    assert tag_weather(200, 0.30, 330, cfg) == "cloudy"      # 够亮但抖动大 → 多云
    assert tag_weather(60, 0.05, 330, cfg) == "overcast"     # < 35% 峰值 → 阴雨
    assert tag_weather(float("nan"), 0.0, 330, cfg) == "nodata"


def test_flat_top_detection():
    cfg = BaselineCfg()
    ts = pd.date_range("2026-09-01 05:00", "2026-09-01 19:00", freq="5min")
    h = (ts.hour + ts.minute / 60).to_numpy()
    p = 100 * np.clip(np.sin((h - 6) / 13 * np.pi), 0, None)
    limited, slots = detect_flat_top(pd.Series(ts), pd.Series(np.minimum(p, 80.0)), cfg, rated_ac_kw=80.0)
    assert limited and len(slots) >= 4
    not_limited, _ = detect_flat_top(pd.Series(ts), pd.Series(p * (1 + np.random.default_rng(0).normal(0, 0.004, len(p)))), cfg, rated_ac_kw=120.0)
    assert not not_limited


def test_streak_helper():
    assert BaselineAgent._streak(np.array([-0.01, -0.06, -0.07, -0.08]), -0.05) == 3
    assert BaselineAgent._streak(np.array([-0.06, -0.02]), -0.05) == 0
    assert BaselineAgent._streak(np.array([-0.06, np.nan, -0.07]), -0.05) == 1


def test_csv_wide_table_melt(tmp_path):
    root = tmp_path / "imp"
    (root / "ST9" / "strings").mkdir(parents=True)
    ts = pd.date_range("2026-10-01 11:00", periods=3, freq="5min")
    wide = pd.DataFrame({"inverter_id": "INVX", "ts": ts, "pv1_i": [8.1, 8.2, 8.3], "pv1_u": [830, 831, 832],
                         "pv2_i": [7.9, 8.0, 8.1], "pv2_u": [829, 830, 830]})
    wide.to_csv(root / "ST9" / "strings" / "a.csv", index=False)
    ad = CsvAdapter("ST9", {"root": str(root)})
    res = ad.fetch(date(2026, 10, 1), date(2026, 10, 1))
    assert set(res.strings["string_id"]) == {"INVX-pv1", "INVX-pv2"}
    assert len(res.strings) == 6


def test_settings_roundtrip(tmp_path):
    s = Settings(data_dir=tmp_path / "d")
    s.thresholds.fault_floor = -0.06
    s.dump(tmp_path / "c.yaml")
    s2 = Settings.load(tmp_path / "c.yaml")
    assert s2.thresholds.fault_floor == -0.06 and s2.data_dir == tmp_path / "d"
