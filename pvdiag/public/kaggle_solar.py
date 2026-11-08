"""Kaggle《Solar Power Generation Data》→ pvdiag 结构。

数据特点（两座印度电站，2020-05-15 ~ 2020-06-17，15 分钟）：
- 每站 22 台逆变器，只有逆变器级 DC/AC 功率，没有组串电流 → 把 **整站当作一台"虚拟逆变器"，22 台真实逆变器当作 22 路"组串"**，
  pvdiag 的"同机各串互比"在这里就是"同站各逆变器互比"（口径一致：中午窗口中位数互比、晴好天过滤、持续度）
- Plant 1 的 DC_POWER 比 AC_POWER 大 10 倍（AC/DC≈0.098，公认的数据单位瑕疵）→ 这里把 Plant 1 的 DC 除以 10 复原为 kW
- Plant 2 多台逆变器白天频繁出现"整段 0 输出"（掉线/停机），不同日期轮换 → 真实世界的"间歇停机"签名，合成数据里没有
- 自带气象站：辐照（kW/m²）、环境/组件温度 → 作为 station_daily.irradiance_kwh_m2，可独立校验晴好天判定

电流换算：I = P_dc(kW)·1000 / 600V（固定 600V 虚拟电压，能量守恒：I·V·Δt = DC 电量），互比只看比值，不受此影响。
15 分钟 → 5 分钟：日内线性插值（限两格），缺测段保持缺测（缺数据 ≠ 零发电）。

参考标签（独立于 pvdiag 口径，由原始数据统计得出，供基准评测用）：
- underperform：在"全站干净日"（无逆变器掉线、辐照充足）上，该逆变器 DC 日电量 / 全站中位 的中位数 ≤ 0.95
- mild：同上 ≤ 0.975
- intermittent_outage：白天（8:00-17:00）零输出占比 ≥20% 的天数 ≥ 5 天
- normal：其余
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

V_DC = 600.0                 # 虚拟直流电压（V）
PLANT_IDS = {1: 4135001, 2: 4136001}


def _read_generation(root: Path, plant: int) -> pd.DataFrame:
    g = pd.read_csv(root / f"Plant_{plant}_Generation_Data.csv")
    # Plant 1 时间格式 dd-mm-yyyy HH:MM；Plant 2 为 ISO
    g["ts"] = pd.to_datetime(g["DATE_TIME"], dayfirst=(plant == 1), format="mixed")
    g["dc_kw"] = pd.to_numeric(g["DC_POWER"], errors="coerce") / (10.0 if plant == 1 else 1.0)
    g["ac_kw"] = pd.to_numeric(g["AC_POWER"], errors="coerce")
    return g[["ts", "SOURCE_KEY", "dc_kw", "ac_kw", "DAILY_YIELD"]].sort_values(["SOURCE_KEY", "ts"])


def _read_weather(root: Path, plant: int) -> pd.DataFrame:
    w = pd.read_csv(root / f"Plant_{plant}_Weather_Sensor_Data.csv")
    w["ts"] = pd.to_datetime(w["DATE_TIME"])
    w["ghi_w_m2"] = pd.to_numeric(w["IRRADIATION"], errors="coerce") * 1000.0
    return w[["ts", "ghi_w_m2", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]].sort_values("ts")


def upsample_5min(df: pd.DataFrame, key: str, value_cols: List[str]) -> pd.DataFrame:
    """15 分钟 → 5 分钟：按日线性插值（只在已有数据之间补，最多补两格），不跨缺测段。"""
    out = []
    for k, g in df.groupby(key, sort=False):
        s = g.set_index("ts")[value_cols].sort_index()
        s = s[~s.index.duplicated(keep="last")]
        for d, part in s.groupby(s.index.date):
            r = part.resample("5min").asfreq()
            r = r.interpolate(method="linear", limit=2, limit_area="inside")
            r = r.dropna(how="all")
            r[key] = k
            out.append(r.reset_index())
    if not out:
        return pd.DataFrame(columns=["ts", key, *value_cols])
    return pd.concat(out, ignore_index=True)


def reference_labels(g: pd.DataFrame, w: pd.DataFrame, string_ids: Dict[str, str]) -> pd.DataFrame:
    """独立于 pvdiag 的参考标签（见模块说明）。string_ids: SOURCE_KEY → string_id"""
    g = g.copy()
    g["d"] = g["ts"].dt.date
    mins = g["ts"].dt.hour * 60 + g["ts"].dt.minute
    day = g[(mins >= 8 * 60) & (mins < 17 * 60)]
    energy = g.groupby(["d", "SOURCE_KEY"])["dc_kw"].sum().unstack() * 0.25
    zero_frac = day.assign(z=day["dc_kw"] <= 0).groupby(["d", "SOURCE_KEY"])["z"].mean().unstack()
    n_day = day.groupby(["d", "SOURCE_KEY"]).size().unstack()
    expected = n_day.max().max()
    w = w.copy()
    w["d"] = w["ts"].dt.date
    irr = w.groupby("d")["ghi_w_m2"].sum() * 0.25 / 1000.0          # kWh/m²
    # "干净"的逆变器·日：该机白天无零输出（<5%）、样本完整（≥70%）、当日辐照充足（≥ 75% 的 90 分位）；
    # 全站中位数只用当日干净的机器算（Plant 2 几乎每天都有机器掉线，不能要求全站都干净）
    ok = (zero_frac.fillna(1) < 0.05) & (n_day.fillna(0) >= 0.7 * expected)
    sunny = pd.Series({d: (d in irr.index and irr[d] >= 0.75 * irr.quantile(0.9)) for d in energy.index})
    ok = ok[sunny.values]
    e_ok = energy.loc[ok.index].where(ok)
    med_ok = e_ok.median(axis=1)
    n_ok = ok.sum(axis=1)
    ratio_ok = e_ok.div(med_ok, axis=0)[n_ok >= 8]            # 当日至少 8 台干净机器才算有效中位数
    rows = []
    for key in energy.columns:
        r_clean = ratio_ok[key].dropna()
        med = float(r_clean.median()) if len(r_clean) >= 3 else np.nan
        dropout_days = int((zero_frac[key].fillna(0) >= 0.20).sum())
        gap_days = int((n_day[key].fillna(0) < 0.5 * expected).sum())
        if not np.isnan(med) and med <= 0.95:
            label = "underperform"
        elif not np.isnan(med) and med <= 0.975:
            label = "mild"
        else:
            label = "normal"
        rows.append({"string_id": string_ids[key], "source_key": key, "median_clean_ratio": round(med, 4) if not np.isnan(med) else None,
                     "n_clean_days": int(len(r_clean)), "dropout_days": dropout_days, "gap_days": gap_days,
                     "label": label, "intermittent_outage": int(dropout_days >= 5)})
    return pd.DataFrame(rows).sort_values("median_clean_ratio")


def load_kaggle_plant(root: Path | str, plant: int, station_id: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """返回与 sim.simulate() 同形的 bundle：stations, inverters, strings, samples_strings, samples_inverters, irradiance, reference"""
    root = Path(root)
    station_id = station_id or f"KP{plant}"
    inv_id = f"{station_id}-INV01"
    g = _read_generation(root, plant)
    w = _read_weather(root, plant)
    keys = sorted(g["SOURCE_KEY"].unique())
    string_ids = {k: f"{inv_id}-pv{i + 1}" for i, k in enumerate(keys)}

    # ---- 档案
    strings = pd.DataFrame([{
        "string_id": string_ids[k], "station_id": station_id, "inverter_id": inv_id, "mppt": (i // 2) + 1, "pv_index": i + 1,
        "n_modules": None, "module_model": "Kaggle-inverter-as-string", "module_wp": None, "module_vmp": None, "module_imp": None,
        "azimuth": 180, "tilt": 15, "roof_area": f"{station_id}-区{i // 6 + 1}", "known_shading": 0,
        "shading_expected_dev": None, "shading_note": "", "notes": f"SOURCE_KEY={k}",
    } for i, k in enumerate(keys)])
    peak_ac = float(g.groupby("ts")["ac_kw"].sum().quantile(0.999))
    inverters = pd.DataFrame([{"inverter_id": inv_id, "station_id": station_id, "name": f"Plant {plant} 全站（虚拟逆变器）",
                               "model": "virtual", "rated_ac_kw": None, "dc_kwp": None,
                               "extra_json": f'{{"n_real_inverters": {len(keys)}, "peak_ac_kw_observed": {peak_ac:.1f}}}'}])
    stations = pd.DataFrame([{"station_id": station_id, "name": f"Kaggle Plant {plant}（PLANT_ID {PLANT_IDS[plant]}）", "platform": "csv",
                              "capacity_kwp": None, "region": "印度", "lat": None, "lon": None, "tariff_json": "{}",
                              "extra_json": '{"source": "kaggle_solar", "note": "22 台逆变器映射为 22 路组串"}'}])

    # ---- 5 分钟明细
    s5 = upsample_5min(g, "SOURCE_KEY", ["dc_kw", "ac_kw"])
    samples_strings = pd.DataFrame({
        "string_id": s5["SOURCE_KEY"].map(string_ids), "ts": s5["ts"],
        "current_a": (s5["dc_kw"].clip(lower=0) * 1000.0 / V_DC).round(3), "voltage_v": V_DC,
    })
    # 虚拟逆变器功率 = 各逆变器 AC 之和；某时刻缺报的逆变器按已报均值补齐（避免缺报被当成云影）
    per_ts = s5.groupby("ts")["ac_kw"].agg(["sum", "count"])
    per_ts = per_ts[per_ts["count"] >= 0.5 * len(keys)]
    pac = per_ts["sum"] * (len(keys) / per_ts["count"])
    samples_inverters = pd.DataFrame({"inverter_id": inv_id, "ts": pac.index, "pac_kw": pac.values.round(3), "status": None})
    w5 = upsample_5min(w.assign(k="w"), "k", ["ghi_w_m2"])[["ts", "ghi_w_m2"]]
    return {
        "stations": stations, "inverters": inverters, "strings": strings,
        "samples_strings": samples_strings, "samples_inverters": samples_inverters, "irradiance": w5,
        "reference": reference_labels(g, w, string_ids),
        "weather_raw": w,
    }


def irradiance_clear_days(w: pd.DataFrame, noon=("11:30", "13:30")) -> pd.DataFrame:
    """用气象站辐照独立判晴好天：日辐照 ≥ 75% 的 90 分位，且中午窗口辐照变异系数 ≤0.15。供校验 pvdiag 的 weather_tag。"""
    w = w.copy()
    w["d"] = w["ts"].dt.date
    mins = w["ts"].dt.hour * 60 + w["ts"].dt.minute
    n0 = int(noon[0][:2]) * 60 + int(noon[0][3:])
    n1 = int(noon[1][:2]) * 60 + int(noon[1][3:])
    daily = w.groupby("d")["ghi_w_m2"].sum() * 0.25 / 1000.0
    noonw = w[(mins >= n0) & (mins < n1)].groupby("d")["ghi_w_m2"].agg(["mean", "std"])
    cv = (noonw["std"] / noonw["mean"]).reindex(daily.index)
    ref = daily.quantile(0.9)
    tag = np.where(daily >= 0.75 * ref, np.where(cv.fillna(0) <= 0.15, "clear", "cloudy"), "overcast")
    return pd.DataFrame({"date": [d.isoformat() for d in daily.index], "irradiation_kwh_m2": daily.values.round(3),
                         "noon_cv": cv.values, "ref_tag": tag})
