"""日特征库计算：把 5 分钟明细压成"每串每日一行"的特征，供基线/诊断智能体消费。

每串每日：中午窗口中位数（电流/电压）、全天最大电流、能量、日间电流标准差、全天 0 电流、电流钉死、
          半小时粒度的全天曲线向量（电流/电压）—— 曲线签名分型的输入
每机每日：中午功率中位数/峰值/变异系数、限发平顶时段、离线标记、半小时功率曲线
每站每日：中午功率、晴天峰值参考、变异系数、天气标记（clear/cloudy/overcast/nodata）
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import BaselineCfg, Thresholds, hhmm_to_minutes

SLOTS_PER_DAY = 48  # 30 分钟槽


def _minutes(ts: pd.Series) -> pd.Series:
    return ts.dt.hour * 60 + ts.dt.minute


def slot_label(slot: int, slot_minutes: int = 30) -> str:
    m = slot * slot_minutes
    return f"{m // 60:02d}:{m % 60:02d}"


def profile_vector(ts: pd.Series, values: pd.Series, slot_minutes: int = 30) -> List[Optional[float]]:
    """按半小时槽求均值 → 长度 48 的列表（缺测为 None）"""
    n = (24 * 60) // slot_minutes
    slots = (_minutes(ts) // slot_minutes).to_numpy()
    vals = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out: List[Optional[float]] = [None] * n
    df = pd.DataFrame({"s": slots, "v": vals}).dropna()
    if df.empty:
        return out
    g = df.groupby("s")["v"].mean()
    for s, v in g.items():
        if 0 <= int(s) < n:
            out[int(s)] = round(float(v), 4)
    return out


def stale_run_minutes(values: np.ndarray, zero_a: float, step_min: int = 5, rtol: float = 1e-4) -> int:
    """日间（9-16 点）电流"钉死不动"的最长连续时长（分钟）：相邻样本相对变化 < rtol 且电流 > 零值阈。
    与 pvanalytics.quality.gaps.stale_values_diff 的思想一致，但按天聚合成"最长钉死分钟数"，
    用来抓"半天钉死"（全天 std 很大但其中几小时冻结）这类 daytime_i_std 抓不到的采集故障。"""
    v = values[~np.isnan(values)]
    if len(v) < 3:
        return 0
    same = (np.abs(np.diff(v)) <= rtol * np.maximum(np.abs(v[1:]), 1e-9)) & (v[1:] > zero_a)
    best = cur = 0
    for s in same:
        cur = cur + 1 if s else 0
        best = max(best, cur)
    return int((best + 1) * step_min) if best else 0


# ---------------------------------------------------------------------------- 组串
def compute_string_daily(samples: pd.DataFrame, day: date, cfg: BaselineCfg, th: Thresholds) -> pd.DataFrame:
    """samples: string_id, ts, current_a, voltage_v（单日、单站）"""
    if samples.empty:
        return pd.DataFrame()
    df = samples.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[df["ts"].dt.date == day]
    if df.empty:
        return pd.DataFrame()
    mins = _minutes(df["ts"])
    n0, n1 = hhmm_to_minutes(cfg.noon_start), hhmm_to_minutes(cfg.noon_end)
    d0, d1 = hhmm_to_minutes(cfg.daylight_start), hhmm_to_minutes(cfg.daylight_end)
    expected_daytime = (d1 - d0) // 5
    df["is_noon"] = (mins >= n0) & (mins < n1)
    df["is_day"] = (mins >= d0) & (mins < d1)
    df["is_core"] = (mins >= 9 * 60) & (mins < 16 * 60)          # 钉死检测窗口
    df["i"] = pd.to_numeric(df["current_a"], errors="coerce")
    df["v"] = pd.to_numeric(df["voltage_v"], errors="coerce")
    df["i_clean"] = df["i"].where(df["i"] >= th.zero_current_a, 0.0)   # <0.5A 按未接/夜间 → 0

    rows = []
    for sid, g in df.groupby("string_id", sort=False):
        noon = g[g["is_noon"]]
        day_g = g[g["is_day"] & g["i"].notna()]
        core = g[g["is_core"] & g["i"].notna()]
        noon_valid = noon[noon["i"] >= th.zero_current_a]
        day_max_i = float(np.nanmax(g["i"].to_numpy())) if g["i"].notna().any() else np.nan
        energy = float(np.nansum((g["i_clean"] * g["v"].fillna(0)).to_numpy()) * (5 / 60) / 1000.0)
        core_std = float(core["i"].std()) if len(core) >= 20 else np.nan
        core_mean = float(core["i"].mean()) if len(core) >= 20 else np.nan
        stale_run = stale_run_minutes(core.sort_values("ts")["i"].to_numpy(dtype=float), th.zero_current_a) if len(core) else 0
        rows.append({
            "string_id": sid, "date": day.isoformat(),
            "n_samples": int(len(g)), "n_daytime_expected": int(expected_daytime),
            "n_missing": int(max(0, expected_daytime - len(day_g))),
            "noon_median_i": float(noon["i_clean"].median()) if len(noon) else np.nan,
            "noon_median_v": float(noon_valid["v"].median()) if len(noon_valid) else np.nan,
            "noon_n": int(len(noon)),
            "noon_zero_frac": float((noon["i_clean"] <= 0).mean()) if len(noon) else np.nan,
            "day_max_i": day_max_i,
            "day_energy_kwh": round(energy, 4),
            "daytime_i_std": core_std,
            "zero_all_day": int(np.isnan(day_max_i) or day_max_i < th.zero_current_a),
            "stuck": int((not np.isnan(core_std)) and core_std < th.stuck_std_a and core_mean > th.zero_current_a),
            "stale_run_min": int(stale_run),
            "i_profile_json": json.dumps(profile_vector(g["ts"], g["i_clean"], cfg.slot_minutes)),
            "v_profile_json": json.dumps(profile_vector(g["ts"], g["v"], cfg.slot_minutes)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- 逆变器
def detect_flat_top(ts: pd.Series, pac: pd.Series, cfg: BaselineCfg, rated_ac_kw: Optional[float]) -> Tuple[bool, List[int]]:
    """限发"平顶"签名：功率钉在固定值不动。返回 (limited_flag, 受影响半小时槽列表)。"""
    s = pd.DataFrame({"ts": ts, "p": pd.to_numeric(pac, errors="coerce")}).dropna().sort_values("ts")
    if len(s) < cfg.limited_run_slots + 1:
        return False, []
    mins = _minutes(s["ts"]).to_numpy()
    p = s["p"].to_numpy(dtype=float)
    peak = float(np.nanmax(p)) if len(p) else 0.0
    if peak <= 0:
        return False, []
    core = (mins >= 9 * 60 + 30) & (mins < 15 * 60)
    rel = np.abs(np.diff(p)) / np.maximum(p[1:], 1e-6)
    flat = np.concatenate([[False], rel <= cfg.limited_flat_tol]) & core & (p >= 0.85 * peak)
    # 找最长连续平段
    best_len, best_idx, cur_len, cur_start = 0, (0, 0), 0, 0
    for k, f in enumerate(flat):
        if f:
            if cur_len == 0:
                cur_start = k
            cur_len += 1
            if cur_len > best_len:
                best_len, best_idx = cur_len, (cur_start, k)
        else:
            cur_len = 0
    if best_len == 0:
        return False, []
    level = float(np.median(p[best_idx[0]:best_idx[1] + 1]))
    at_rated = rated_ac_kw is not None and rated_ac_kw > 0 and level >= 0.97 * rated_ac_kw
    limited = (at_rated and best_len >= cfg.limited_run_slots) or (best_len >= 2 * cfg.limited_run_slots)
    if not limited:
        return False, []
    slots = sorted({int(m // cfg.slot_minutes) for m, f in zip(mins, flat) if f})
    return True, slots


def compute_inverter_daily(inv_samples: pd.DataFrame, day: date, cfg: BaselineCfg,
                           rated_map: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    if inv_samples.empty:
        return pd.DataFrame()
    df = inv_samples.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[df["ts"].dt.date == day]
    if df.empty:
        return pd.DataFrame()
    mins = _minutes(df["ts"])
    n0, n1 = hhmm_to_minutes(cfg.noon_start), hhmm_to_minutes(cfg.noon_end)
    d0, d1 = hhmm_to_minutes(cfg.daylight_start), hhmm_to_minutes(cfg.daylight_end)
    df["is_noon"] = (mins >= n0) & (mins < n1)
    df["is_day"] = (mins >= d0) & (mins < d1)
    df["p"] = pd.to_numeric(df["pac_kw"], errors="coerce")
    expected_daytime = (d1 - d0) // 5
    rows = []
    for inv, g in df.groupby("inverter_id", sort=False):
        noon = g[g["is_noon"]]["p"].dropna()
        day_g = g[g["is_day"]]["p"].dropna()
        offline = int(len(day_g) < 0.5 * expected_daytime or (day_g > 0).sum() < 0.3 * expected_daytime)
        limited, slots = detect_flat_top(g["ts"], g["p"], cfg, (rated_map or {}).get(inv))
        rows.append({
            "inverter_id": inv, "date": day.isoformat(),
            "pac_noon_median": float(noon.median()) if len(noon) else np.nan,
            "pac_peak": float(day_g.max()) if len(day_g) else np.nan,
            "pac_noon_cv": float(noon.std() / noon.mean()) if len(noon) > 2 and noon.mean() > 0 else np.nan,
            "limited_flag": int(limited), "limited_slots_json": json.dumps(slots),
            "offline_flag": offline,
            "pac_profile_json": json.dumps(profile_vector(g["ts"], g["p"], cfg.slot_minutes)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- 电站 / 天气
def station_noon_cv(inv_samples: pd.DataFrame, day: date, cfg: BaselineCfg) -> float:
    if inv_samples.empty:
        return np.nan
    df = inv_samples.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[df["ts"].dt.date == day]
    mins = _minutes(df["ts"])
    n0, n1 = hhmm_to_minutes(cfg.noon_start), hhmm_to_minutes(cfg.noon_end)
    noon = df[(mins >= n0) & (mins < n1)]
    tot = noon.groupby("ts")["pac_kw"].sum(min_count=1).dropna()
    if len(tot) < 3 or tot.mean() <= 0:
        return np.nan
    return float(tot.std() / tot.mean())


def tag_weather(noon_power: float, noon_cv: float, peak_ref: float, cfg: BaselineCfg) -> str:
    if noon_power is None or np.isnan(noon_power) or peak_ref is None or np.isnan(peak_ref) or peak_ref <= 0:
        return "nodata"
    if noon_power >= cfg.clear_day_ratio * peak_ref:
        if np.isnan(noon_cv) or noon_cv <= cfg.clear_cv_max:
            return "clear"
        return "cloudy"
    return "overcast"


def compute_station_daily(station_id: str, day: date, inv_daily: pd.DataFrame, noon_cv: float,
                          history: pd.DataFrame, capacity_kwp: Optional[float], cfg: BaselineCfg,
                          irradiance_kwh_m2: Optional[float] = None, irradiance_source: str = "measured") -> Dict:
    """history: 该站此前 station_daily（含 noon_power 列），用于晴天峰值参考。"""
    online = inv_daily[inv_daily["offline_flag"] == 0] if not inv_daily.empty else inv_daily
    noon_power = float(online["pac_noon_median"].sum()) if not online.empty else np.nan
    hist = history[history["date"] < day.isoformat()].tail(cfg.peak_lookback_days) if not history.empty else history
    if not hist.empty and hist["noon_power"].notna().sum() >= 10:
        peak_ref = float(np.nanpercentile(hist["noon_power"].to_numpy(dtype=float), 95))
    elif capacity_kwp:
        peak_ref = capacity_kwp * cfg.peak_fallback_capacity_ratio
    elif not hist.empty and hist["noon_power"].notna().any():
        peak_ref = float(np.nanmax(hist["noon_power"]))
    else:
        peak_ref = noon_power if not np.isnan(noon_power) else np.nan
    # 冷启动早期（<10 天）：若当天明显高于历史峰值，用当天值刷新参考，避免把第一批晴天都判成"异常高"
    if not np.isnan(noon_power) and not np.isnan(peak_ref) and noon_power > peak_ref:
        peak_ref = max(peak_ref, noon_power) if (hist.empty or len(hist) < 10) else peak_ref
    tag = tag_weather(noon_power, noon_cv, peak_ref, cfg)
    # 有气象站辐照时用它校正：日辐照 < 50% 近期晴天参考 → 阴雨；< 75% 且判了 clear → 降为 cloudy（仅在有 ≥10 天辐照历史时启用）
    # 晴空模型给的是理论上限，不能拿来判阴天；只有实测/再分析辐照参与天气校正
    if irradiance_source in ("measured", "openmeteo") and irradiance_kwh_m2 is not None and not np.isnan(irradiance_kwh_m2) and not hist.empty \
            and "irradiance_kwh_m2" in hist.columns and hist["irradiance_kwh_m2"].notna().sum() >= 10:
        irr_ref = float(np.nanpercentile(hist["irradiance_kwh_m2"].to_numpy(dtype=float), 95))
        if irr_ref > 0 and tag != "nodata":
            if irradiance_kwh_m2 < 0.5 * irr_ref:
                tag = "overcast"
            elif irradiance_kwh_m2 < 0.75 * irr_ref and tag == "clear":
                tag = "cloudy"
    rain = int(tag == "overcast" and (np.isnan(noon_power) or noon_power < 0.2 * peak_ref))
    return {"station_id": station_id, "date": day.isoformat(), "noon_power": noon_power, "peak_ref": peak_ref,
            "noon_cv": noon_cv, "weather_tag": tag, "irradiance_kwh_m2": irradiance_kwh_m2, "rain": rain,
            "irradiance_source": irradiance_source if irradiance_kwh_m2 is not None else None}
