"""连续运行压测（缩短版）：60 天 · 1 站 · 16 串，模拟复核员/主管/现场一起跑，验证长周期不出错、不振荡、召回/误报达标。"""
import logging
from pathlib import Path

from pvdiag.sim.soak import SoakConfig, SoakRunner, write_report

logging.getLogger("pvdiag").setLevel(logging.ERROR)


def test_soak_short(tmp_path: Path):
    cfg = SoakConfig(days=80, n_stations=1, n_inverters=2, strings_per_inverter=8, seed=7, fault_rate=0.02,
                     settings_overrides={"confidence": {"min_labels_for_ml": 15}, "dispatch": {"gate_open_streak_days": 3, "auto_gate_min_reviews": 8}})
    r = SoakRunner(cfg, tmp_path / "soak")
    r.build_world()
    assert len(r.truth) >= 4, r.truth
    s = r.run()
    p = write_report(s, tmp_path / "out")
    assert p.exists()
    # 80 天流水线不中断
    assert s["agent_failures_days"] == 0, [d for d in s["daily"] if d["agents_failed"]]
    # 随机故障大多被 A/B 级召回；健康串零 A/B 误报
    assert s["recall"] >= 0.75, s
    assert s["healthy_false_alarm_ab_strings"] == 0, s
    # 现场回填会产生标注（复核数 ≥ 工单回填数）
    assert s["reviews_total"] >= 5
    # 自调参不振荡；每天都给出决定
    assert not s["autotune"]["oscillation"]
    assert all(d["tuning"] for d in s["daily"])
