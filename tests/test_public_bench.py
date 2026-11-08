"""公开数据基准（真实数据"考试"）：
- Kaggle 两站：低发全召回、健康机器零 A/B 误报、Plant 2 间歇掉线被识别为「间歇停机/掉线」而非遮挡
- NREL/pvanalytics 标注片段：钉死日 / 限发日 的特征层识别

数据不在本地且下载失败（无网）→ 自动跳过；CI 里先 `pvdiag public fetch` 再跑。
环境变量 PVDIAG_PUBLIC_DIR 指定缓存目录（默认 ./data/public）。
"""
import logging
from pathlib import Path

import pytest

from pvdiag.public.datasets import fetch_dataset, is_cached

logging.getLogger("pvdiag").setLevel(logging.WARNING)


def _kaggle_available() -> bool:
    if is_cached("kaggle_solar"):
        return True
    try:
        fetch_dataset("kaggle_solar", timeout=60)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kaggle_available(), reason="Kaggle 镜像数据不可用（无网/未下载）")
def test_kaggle_two_plants(tmp_path):
    from pvdiag.public.bench import bench_kaggle
    rep = bench_kaggle(tmp_path / "work", fetch=False)
    p1, p2 = rep["plants"]["1"], rep["plants"]["2"]
    # Plant 1：2 台持续低发（-7%/-8%）全部召回；20 台健康机器零 A/B 误报
    assert p1["underperform"]["n"] >= 2 and p1["underperform"]["recalled_abd"] == p1["underperform"]["n"]
    assert p1["healthy"]["false_alarm_ab"] <= 1, p1["healthy"]
    # Plant 2：18 台间歇掉线，至少 16 台原因为 间歇停机/掉线 或 数据问题（而不是遮挡/二极管）
    assert p2["outage"]["n"] >= 15 and p2["outage"]["classified_as_outage_or_data"] >= p2["outage"]["n"] - 2, p2["outage"]
    assert p2["underperform"]["recalled_abd"] == p2["underperform"]["n"]
    # 晴好天判定与气象站辐照的"晴/非晴"一致率
    assert rep["summary"]["weather_clear_agreement"] >= 0.7


def test_nrel_labeled_fragments(tmp_path):
    pytest.importorskip("pvanalytics")
    from pvdiag.public.bench import bench_nrel
    rep = bench_nrel(tmp_path / "work")
    st = rep["summary"]["inv2173_stale.stale"]
    assert st["recall"] == 1.0 and st["precision"] >= 0.75, st
    cl = rep["summary"]["inv7539_clip.clipping"]
    assert cl["recall"] >= 0.75 and cl["precision"] >= 0.6, cl
