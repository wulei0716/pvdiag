"""简化的太阳几何 / 晴空倾斜面辐照（POA）模型——够用即可，不追求气象精度。

用途：
1. 季节性朝向期望：两路朝向/倾角不同的组串，中午窗口的"物理期望电流比"随季节变化（冬天南向 vs 东向差 20%，夏天只差几个点）。
   把它从互比偏差里扣掉，剩下的才是真正的异常 → 朝向差异（C 级）不再靠"阴天消失"的经验判断，而是对着物理期望校核档案。
2. 跟踪支架期望：单轴跟踪的清晨/傍晚增益曲线，用来识别"卡滞"（该串呈固定倾角的钟形，而同机其它串是跟踪的平顶形）。

模型：晴空直射 DNI ≈ 950·(sin h)^0.15，水平散射 DHI ≈ 0.12·GHI_clear（各向同性），POA = DNI·cos(θ) + DHI·(1+cos β)/2 + 地面反射 0.2·GHI·(1−cos β)/2。
只用比值，绝对值不重要。
"""
from __future__ import annotations

import math
from datetime import date
from typing import Iterable, Optional, Tuple

import numpy as np


def declination(doy: int) -> float:
    return math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365.0)


def sun_position(lat_deg: float, doy: int, minutes: np.ndarray, lon_offset_min: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (太阳高度角 h, 太阳方位角 A)，弧度；A 以正南为 180°、正东 90°（与档案 azimuth 约定一致）。minutes 为当地太阳时分钟。"""
    lat = math.radians(lat_deg)
    dec = declination(doy)
    ha = np.radians(((minutes - lon_offset_min) / 60.0 - 12.0) * 15.0)
    sin_h = math.sin(lat) * math.sin(dec) + math.cos(lat) * math.cos(dec) * np.cos(ha)
    h = np.arcsin(np.clip(sin_h, -1, 1))
    cos_h = np.cos(h)
    # 方位角（从正北顺时针）
    sin_az = -np.sin(ha) * math.cos(dec) / np.maximum(cos_h, 1e-6)
    cos_az = (math.sin(dec) - math.sin(lat) * sin_h) / np.maximum(math.cos(lat) * cos_h, 1e-6)
    az = np.degrees(np.arctan2(sin_az, cos_az)) % 360.0
    return h, np.radians(az)


def lon_offset_minutes(lon_deg: Optional[float], tz_meridian_deg: float = 120.0) -> float:
    """时钟时间 − 当地太阳时（分钟）：经度每偏离时区中央经线 1° 差 4 分钟。lon 缺失 → 0（按中央经线）"""
    if lon_deg is None or (isinstance(lon_deg, float) and math.isnan(lon_deg)):
        return 0.0
    return float((tz_meridian_deg - lon_deg) * 4.0)


def clear_sky_poa(lat_deg: float, doy: int, minutes: np.ndarray, azimuth_deg: float, tilt_deg: float, albedo: float = 0.2,
                  lon_offset_min: float = 0.0) -> np.ndarray:
    """晴空下倾斜面辐照（相对单位），夜间为 0；minutes 为时钟分钟，lon_offset_min 把它换算到太阳时"""
    h, az = sun_position(lat_deg, doy, minutes, lon_offset_min)
    up = h > 0
    sin_h = np.clip(np.sin(h), 0, 1)
    dni = np.where(up, 950.0 * np.power(sin_h, 0.15), 0.0)
    ghi = dni * sin_h
    dhi = 0.12 * ghi
    beta = math.radians(tilt_deg)
    gamma = math.radians(azimuth_deg)
    cos_theta = np.sin(h) * math.cos(beta) + np.cos(h) * math.sin(beta) * np.cos(az - gamma)
    poa = dni * np.clip(cos_theta, 0, None) + dhi * (1 + math.cos(beta)) / 2 + albedo * ghi * (1 - math.cos(beta)) / 2
    return np.where(up, poa, 0.0)


def tracker_poa(lat_deg: float, doy: int, minutes: np.ndarray, axis_tilt_deg: float = 0.0, max_angle_deg: float = 60.0,
                lon_offset_min: float = 0.0) -> np.ndarray:
    """水平南北轴单轴跟踪（东西转动）的 POA 近似：跟踪角 = clip(太阳方位在东西向的投影角, ±max)，再按对应倾斜面算"""
    h, az = sun_position(lat_deg, doy, minutes, lon_offset_min)
    up = h > 0
    sin_h = np.clip(np.sin(h), 0, 1)
    dni = np.where(up, 950.0 * np.power(sin_h, 0.15), 0.0)
    ghi = dni * sin_h
    dhi = 0.12 * ghi
    # 太阳在东西-垂直平面内的投影角（在东为正 → 面朝东；在西为负 → 面朝西）
    sx = np.cos(h) * np.sin(az)          # 东西分量（az 从北顺时针：东=90°→ sin=1）
    sz = np.sin(h)
    rot = np.clip(np.degrees(np.arctan2(sx, np.maximum(sz, 1e-6))), -max_angle_deg, max_angle_deg)
    beta = np.radians(np.abs(rot))
    gamma = np.where(rot >= 0, math.radians(90.0), math.radians(270.0))      # 太阳在东（sx>0）→ 面朝东；在西 → 面朝西
    cos_theta = np.sin(h) * np.cos(beta) + np.cos(h) * np.sin(beta) * np.cos(az - gamma)
    poa = dni * np.clip(cos_theta, 0, None) + dhi * (1 + np.cos(beta)) / 2 + 0.2 * ghi * (1 - np.cos(beta)) / 2
    return np.where(up, poa, 0.0)


def dual_tracker_poa(lat_deg: float, doy: int, minutes: np.ndarray, max_tilt_deg: float = 85.0, lon_offset_min: float = 0.0) -> np.ndarray:
    """双轴跟踪的 POA 近似：板面始终正对太阳（倾角 = 90°−h，受 max_tilt 限制；方位 = 太阳方位），
    POA = DNI·cos θ + 各向同性散射 + 地面反射。晴天全天近似平顶，比单轴清晨/傍晚更高。"""
    h, az = sun_position(lat_deg, doy, minutes, lon_offset_min)
    up = h > 0
    sin_h = np.clip(np.sin(h), 0, 1)
    dni = np.where(up, 950.0 * np.power(sin_h, 0.15), 0.0)
    ghi = dni * sin_h
    dhi = 0.12 * ghi
    beta = np.clip(np.pi / 2 - h, 0, math.radians(max_tilt_deg))
    cos_theta = np.sin(h) * np.cos(beta) + np.cos(h) * np.sin(beta)        # 方位已对准：cos(az−γ)=1
    poa = dni * np.clip(cos_theta, 0, None) + dhi * (1 + np.cos(beta)) / 2 + 0.2 * ghi * (1 - np.cos(beta)) / 2
    return np.where(up, poa, 0.0)


TRACKER_MOUNTS = ("tracker", "tracker_dual")


def mount_poa(mount: str, lat_deg: float, doy: int, minutes: np.ndarray, azimuth_deg: float = 180.0, tilt_deg: float = 20.0,
              lon_offset_min: float = 0.0) -> np.ndarray:
    """按支架类型给出晴空 POA：fixed → 固定朝向；tracker → 水平单轴；tracker_dual → 双轴"""
    if mount == "tracker":
        return tracker_poa(lat_deg, doy, minutes, lon_offset_min=lon_offset_min)
    if mount == "tracker_dual":
        return dual_tracker_poa(lat_deg, doy, minutes, lon_offset_min=lon_offset_min)
    return clear_sky_poa(lat_deg, doy, minutes, azimuth_deg, tilt_deg, lon_offset_min=lon_offset_min)


def tracker_stuck_fit(mount: str, lat_deg: float, d: date, ratio_curve: np.ndarray, slot_minutes: int = 30,
                      stuck_tilts: Iterable[float] = (0, 10, 20, 30), stuck_azimuth: float = 180.0,
                      lon_offset_min: float = 0.0) -> Optional[Tuple[float, float, float]]:
    """跟踪电站：把该串相对群组（都在跟踪）的全天比值曲线，与"卡在固定倾角"的物理期望曲线（固定 POA / 跟踪 POA）比形状。
    返回 (最佳 rmse, 平直假设 rmse, 最像的卡滞倾角)；曲线有效槽不足返回 None。
    单轴卡滞：清晨/傍晚丢增益、中午 ≈1；双轴卡滞：全天钟形且冬季中午也明显偏低——两者都由同一个期望曲线覆盖。"""
    if mount not in TRACKER_MOUNTS:
        return None
    valid = ~np.isnan(ratio_curve)
    if valid.sum() < 8:
        return None
    mins = np.arange(0, 24 * 60, 5)
    doy = d.timetuple().tm_yday
    track = mount_poa(mount, lat_deg, doy, mins, lon_offset_min=lon_offset_min)
    n = (24 * 60) // slot_minutes
    slot = mins // slot_minutes
    flat_rmse = float(np.sqrt(np.mean((ratio_curve[valid] - np.median(ratio_curve[valid])) ** 2)))
    best = None
    for tilt in stuck_tilts:
        fixed = clear_sky_poa(lat_deg, doy, mins, stuck_azimuth, float(tilt), lon_offset_min=lon_offset_min)
        exp = np.full(n, np.nan)
        for s in range(n):
            sel = slot == s
            b = track[sel].mean()
            exp[s] = fixed[sel].mean() / b if b > 1.0 else np.nan
        m = valid & ~np.isnan(exp)
        if m.sum() < 8:
            continue
        scale = float(np.clip(np.median(ratio_curve[m] / exp[m]), 0.95, 1.05))
        rmse = float(np.sqrt(np.mean((ratio_curve[m] - scale * exp[m]) ** 2)))
        if best is None or rmse < best[0]:
            best = (rmse, float(tilt))
    if best is None:
        return None
    return (best[0], flat_rmse, best[1])


def noon_window_minutes(start: str = "11:30", end: str = "13:30", step: int = 5) -> np.ndarray:
    h0, m0 = map(int, start.split(":")); h1, m1 = map(int, end.split(":"))
    return np.arange(h0 * 60 + m0, h1 * 60 + m1, step)


def expected_noon_ratio(lat_deg: float, d: date, az_a: float, tilt_a: float, az_b: float, tilt_b: float,
                        noon: Tuple[str, str] = ("11:30", "13:30"), lon_offset_min: float = 0.0) -> float:
    """组串 A 相对参考朝向 B 在中午窗口的期望电流比（晴空）。朝向/倾角缺失时返回 1.0。"""
    for v in (az_a, tilt_a, az_b, tilt_b):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return 1.0
    mins = noon_window_minutes(*noon)
    doy = d.timetuple().tm_yday
    pa = clear_sky_poa(lat_deg, doy, mins, az_a, tilt_a, lon_offset_min=lon_offset_min).mean()
    pb = clear_sky_poa(lat_deg, doy, mins, az_b, tilt_b, lon_offset_min=lon_offset_min).mean()
    if pb <= 0:
        return 1.0
    return float(pa / pb)


def expected_profile_ratio(lat_deg: float, d: date, az_a: float, tilt_a: float, az_b: float, tilt_b: float,
                           slot_minutes: int = 30, lon_offset_min: float = 0.0) -> np.ndarray:
    """A 相对 B 的全天（按半小时槽）期望比值曲线，用于把朝向差异从曲线签名里扣掉"""
    mins = np.arange(0, 24 * 60, 5)
    doy = d.timetuple().tm_yday
    pa = clear_sky_poa(lat_deg, doy, mins, az_a, tilt_a, lon_offset_min=lon_offset_min)
    pb = clear_sky_poa(lat_deg, doy, mins, az_b, tilt_b, lon_offset_min=lon_offset_min)
    n = (24 * 60) // slot_minutes
    out = np.full(n, np.nan)
    for s in range(n):
        sel = (mins // slot_minutes) == s
        b = pb[sel].mean()
        out[s] = pa[sel].mean() / b if b > 1.0 else np.nan
    return out


def best_fit_orientation(lat_deg: float, d: date, ratio_curve: np.ndarray, ref_az: float, ref_tilt: float,
                         slot_minutes: int = 30, azimuths: Iterable[float] = (90, 120, 150, 180, 210, 240, 270),
                         tilts: Iterable[float] = (5, 10, 15, 20, 25, 30, 40), lon_offset_min: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
    """给定该串相对群组的全天比值曲线（48 槽，NaN 为无效），在朝向/倾角网格里找最像的一组 → (azimuth, tilt, rmse)。
    参考朝向本身也在网格里：若参考朝向最像，说明不是朝向问题。"""
    valid = ~np.isnan(ratio_curve)
    if valid.sum() < 8:
        return None
    best = None
    ref_rmse = None
    grid = [(a, t) for a in azimuths for t in tilts]
    if (ref_az, ref_tilt) not in grid:
        grid.append((ref_az, ref_tilt))
    for az, tilt in grid:
        if True:
            exp = expected_profile_ratio(lat_deg, d, az, tilt, ref_az, ref_tilt, slot_minutes, lon_offset_min)
            m = valid & ~np.isnan(exp)
            if m.sum() < 8:
                continue
            # 允许一个整体标定系数（组串固有效率差 ±5% 内），比形状不比绝对值
            scale = float(np.clip(np.median(ratio_curve[m] / exp[m]), 0.95, 1.05))
            rmse = float(np.sqrt(np.mean((ratio_curve[m] - scale * exp[m]) ** 2)))
            if az == ref_az and tilt == ref_tilt:
                ref_rmse = rmse
            if best is None or rmse < best[2]:
                best = (float(az), float(tilt), rmse)
    if best is None:
        return None
    return (best[0], best[1], best[2], ref_rmse if ref_rmse is not None else best[2])
