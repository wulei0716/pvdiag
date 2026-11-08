"""NREL 真实逆变器片段（随 pvanalytics 分发）→ pvdiag 结构。

每个序列都只有一台逆变器的交流功率（15 分钟），没有组串，所以这里不考"互比"，只考特征层的两个口径：
- ac_power_inv_2173_stale_data：NREL 片段 + 人工标注的"数据钉死"掩码（31 天，其中 4 天白天大段钉死）→ 考 daily_features.stuck / stale_run_min
- ac_power_inv_7539：NREL 片段 + 人工逐点标注的限发（label 列，9 天其中 4 天限发 ≥1h）→ 考 inverter_daily.limited_flag

映射：一个序列 = 一座电站 + 一台逆变器 + 一路"组串"（I = P·1000/600V），时间戳按"日均功率峰值落在 12:00-13:00"整体平移到当地太阳时
（原始数据带 UTC/-07:00 时区，pvdiag 的中午窗口与日间窗口都按当地时间定义）。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .kaggle_solar import V_DC, upsample_5min

PVANALYTICS_SERIES: Dict[str, Dict] = {
    # 注意：pvanalytics 的 *_stale_data.csv 是"注入了钉死值"的版本（含掩码列），要用它自己的 value 列，不能用干净的原始文件
    "inv2173_stale": {"file": "ac_power_inv_2173_stale_data.csv", "col": "value_normalized", "scale_kw": 100.0,
                      "masks": {"stale": ("ac_power_inv_2173_stale_data.csv", "stale_data_mask")}},
    # NREL 人工标注的限发点（label 列），逆变器满功率 ≈ 1.0（归一化）→ 额定按 100 kW
    # 限发只考逆变器功率层（不合成"组串电流"：由功率反推的电流在限发时也是平的，会被钉死检测误当成数据冻结——真实电站里限发时组串电流仍在变）
    "inv7539_clip": {"file": "ac_power_inv_7539.csv", "col": "value_normalized", "scale_kw": 100.0, "rated_kw": 100.0,
                     "masks": {"clip": ("ac_power_inv_7539.csv", "label")}, "strings": False},
}


def _data_dir() -> Path:
    try:
        import pvanalytics
    except ImportError as e:  # pragma: no cover
        raise ImportError("需要 pvanalytics：pip install pvanalytics（或 pip install -e '.[bench]'）") from e
    return Path(os.path.dirname(pvanalytics.__file__)) / "data"


def _read_series(path: Path, col: str) -> pd.Series:
    df = pd.read_csv(path)
    tcol = df.columns[0]
    ts = pd.to_datetime(df[tcol], utc=True).dt.tz_convert(None)      # 统一去时区（先转 UTC 再去掉），之后按太阳时平移
    s = pd.Series(pd.to_numeric(df[col], errors="coerce").to_numpy(), index=pd.DatetimeIndex(ts), name=col).sort_index()
    return s[~s.index.duplicated(keep="last")]


def solar_shift_hours(s: pd.Series) -> int:
    """把日均功率曲线的峰值小时平移到 12 点：返回整数小时偏移"""
    prof = s.groupby(s.index.hour).mean()
    peak_h = int(prof.idxmax())
    return 12 - peak_h


def load_pvanalytics_series(key: str) -> Dict[str, pd.DataFrame]:
    spec = PVANALYTICS_SERIES[key]
    d = _data_dir()
    raw = _read_series(d / spec["file"], spec["col"])
    shift = solar_shift_hours(raw)
    s = raw.copy()
    s.index = s.index + pd.Timedelta(hours=shift)
    s = s.clip(lower=0) * spec["scale_kw"]                      # kW
    station_id = f"NREL-{key}"
    inv_id = f"{station_id}-INV01"
    sid = f"{inv_id}-pv1"
    df = pd.DataFrame({"ts": s.index, "k": "x", "p": s.values})
    p5 = upsample_5min(df, "k", ["p"])
    samples_inverters = pd.DataFrame({"inverter_id": inv_id, "ts": p5["ts"], "pac_kw": p5["p"].round(4), "status": None})
    samples_strings = pd.DataFrame({"string_id": sid, "ts": p5["ts"], "current_a": (p5["p"] * 1000.0 / V_DC).round(4), "voltage_v": V_DC})
    if spec.get("strings") is False:
        samples_strings = samples_strings.iloc[0:0]
    rated = spec.get("rated_kw")
    stations = pd.DataFrame([{"station_id": station_id, "name": f"NREL {key}（pvanalytics 样例）", "platform": "csv", "capacity_kwp": None,
                              "region": "美国", "lat": None, "lon": None, "tariff_json": "{}",
                              "extra_json": f'{{"source": "pvanalytics", "file": "{spec["file"]}", "solar_shift_hours": {shift}}}'}])
    inverters = pd.DataFrame([{"inverter_id": inv_id, "station_id": station_id, "name": inv_id, "model": "", "rated_ac_kw": rated,
                               "dc_kwp": None, "extra_json": "{}"}])
    strings = pd.DataFrame([{"string_id": sid, "station_id": station_id, "inverter_id": inv_id, "mppt": 1, "pv_index": 1,
                             "n_modules": None, "module_model": "NREL-inverter-as-string", "module_wp": None, "module_vmp": None,
                             "module_imp": None, "azimuth": 180, "tilt": 15, "roof_area": "", "known_shading": 0,
                             "shading_expected_dev": None, "shading_note": "", "notes": spec["file"]}])
    # ---- 参考标注（按天，只看 8:00-17:00 当地时）
    ref = pd.DataFrame({"date": sorted({t.date().isoformat() for t in s.index})}).set_index("date")
    for name, (fname, mcol) in spec.get("masks", {}).items():
        m = pd.read_csv(d / fname)
        mts = pd.to_datetime(m.iloc[:, 0], utc=True).dt.tz_convert(None) + pd.Timedelta(hours=shift)
        mask = pd.Series(m[mcol].astype(bool).to_numpy(), index=pd.DatetimeIndex(mts))
        day = mask[(mask.index.hour >= 8) & (mask.index.hour < 17)]
        if name == "clip":
            pts = day.groupby(day.index.date).sum()
            ref["clip_points"] = pd.Series({d_.isoformat(): int(v) for d_, v in pts.items()})
        else:
            share = day.groupby(day.index.date).mean()
            ref[f"{name}_share"] = pd.Series({d_.isoformat(): float(v) for d_, v in share.items()})
    return {"stations": stations, "inverters": inverters, "strings": strings, "samples_strings": samples_strings,
            "samples_inverters": samples_inverters, "irradiance": None, "reference_daily": ref.reset_index()}
