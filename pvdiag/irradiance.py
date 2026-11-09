"""辐照源（可插拔）：竖向基线 Y=E/H、站级 PR、站间互比都需要"今天有多少太阳"。

提供方按 irradiance.providers 顺序回退：
  measured   平台/CSV 带来的实测（adapter FetchResult.irradiance：ts, ghi_w_m2；Meteonorm 导出的小时文件 G_Gh 也走这里）
  nasa_power NASA POWER（https://power.larc.nasa.gov）日 ALLSKY_SFC_SW_DWN，kWh/m²/天，免费、无 key，滞后数天
  meteonorm  Meteonorm REST API（Bearer token，小时 G_Gh W/m² → 日 kWh/m²）
  openmeteo  Open-Meteo ERA5 历史再分析（免费、无 key），daily shortwave_radiation_sum MJ/m² → kWh/m²
  clearsky   晴空模型（solar 水平面积分）—— 永远可用，但只是"理论上限"：不能判阴天，只在晴好天作分母

每天的来源记在 station_daily.irradiance_source，下游据此决定能不能用它判天气、站间互比要求同一天同一来源。
外部提供方的日值缓存在 platform_state（irr:<provider>:<station>），一天只拉一次。
"""
from __future__ import annotations

import math
import os
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import Settings
from .solar import lon_offset_minutes, sun_position

EXTERNAL = ("nasa_power", "meteonorm", "openmeteo")


# ---------------------------------------------------------------------------- 晴空
def clearsky_daily_kwh_m2(lat_deg: float, lon_deg: Optional[float], d: date, tz_meridian_deg: float = 120.0) -> float:
    """晴空水平面日辐照量（kWh/m²）：GHI = DNI·sin h（+12% 各向同性散射），按 5 分钟积分"""
    mins = np.arange(0, 24 * 60, 5)
    h, _ = sun_position(lat_deg, d.timetuple().tm_yday, mins, lon_offset_minutes(lon_deg, tz_meridian_deg))
    sin_h = np.clip(np.sin(h), 0, 1)
    dni = np.where(h > 0, 950.0 * np.power(sin_h, 0.15), 0.0)
    return float((dni * sin_h * 1.12).sum() * (5 / 60) / 1000.0)


# ---------------------------------------------------------------------------- 外部提供方
class NasaPower:
    """NASA POWER 日值：ALLSKY_SFC_SW_DWN（全天空地表短波下行，kWh/m²/天），-999 为缺测"""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url, self.timeout = base_url, timeout

    def daily_ghi(self, lat: float, lon: float, start: date, end: date) -> Dict[str, float]:
        import requests
        r = requests.get(self.base_url, params={"parameters": "ALLSKY_SFC_SW_DWN", "community": "RE", "longitude": lon, "latitude": lat,
                                                "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON"},
                         timeout=self.timeout)
        r.raise_for_status()
        vals = r.json().get("properties", {}).get("parameter", {}).get("ALLSKY_SFC_SW_DWN", {})
        out = {}
        for k, v in vals.items():
            if v is None or float(v) < 0:          # -999 缺测
                continue
            out[f"{k[:4]}-{k[4:6]}-{k[6:8]}"] = float(v)
        return out


class Meteonorm:
    """Meteonorm REST API（v1）：按小时取 G_Gh（W/m²）聚合为日 kWh/m²。
    端点与字段以 Meteonorm 官方文档为准，这里默认 /realtime，可用 meteonorm_base_url 覆盖；鉴权 Bearer token（环境变量 METEONORM_TOKEN）。"""

    def __init__(self, base_url: str, token: str, timeout: int = 30):
        self.base_url, self.token, self.timeout = base_url.rstrip("/"), token, timeout

    def daily_ghi(self, lat: float, lon: float, start: date, end: date) -> Dict[str, float]:
        import requests
        r = requests.get(f"{self.base_url}/realtime", params={"lat": lat, "lon": lon, "start": start.isoformat(), "end": (end + timedelta(days=1)).isoformat(),
                                                             "parameters": "G_Gh", "frequency": "1_hour"},
                         headers={"Authorization": f"Bearer {self.token}"}, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        return hourly_to_daily(body.get("values", body.get("data", [])))


def hourly_to_daily(records) -> Dict[str, float]:
    """[{time|ts|datetime, G_Gh|ghi}] 小时 W/m² → {日期: kWh/m²}"""
    if not records:
        return {}
    df = pd.DataFrame(records)
    tcol = next((c for c in ("time", "ts", "datetime", "period_start") if c in df.columns), None)
    vcol = next((c for c in ("G_Gh", "ghi", "GHI", "ghi_w_m2") if c in df.columns), None)
    if tcol is None or vcol is None:
        return {}
    ts = pd.to_datetime(df[tcol], utc=True, errors="coerce").dt.tz_convert(None)
    v = pd.to_numeric(df[vcol], errors="coerce")
    daily = pd.Series(v.to_numpy(), index=ts).dropna().groupby(lambda t: t.date()).sum() / 1000.0
    return {d.isoformat(): float(x) for d, x in daily.items()}


def meteonorm_csv_to_ghi(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Meteonorm 导出的小时 CSV（列 y,m,dm,h,G_Gh,…）→ (ts, ghi_w_m2)；不是该格式返回 None"""
    cols = {c.lower(): c for c in df.columns}
    if "g_gh" not in cols:
        return None
    if all(k in cols for k in ("y", "m", "dm", "h")):
        ts = pd.to_datetime(dict(year=df[cols["y"]], month=df[cols["m"]], day=df[cols["dm"]], hour=df[cols["h"]].clip(upper=23)), errors="coerce")
    elif "time" in cols:
        ts = pd.to_datetime(df[cols["time"]], errors="coerce")
    else:
        return None
    return pd.DataFrame({"ts": ts, "ghi_w_m2": pd.to_numeric(df[cols["g_gh"]], errors="coerce")}).dropna()


class OpenMeteo:
    def __init__(self, base_url: str = "https://archive-api.open-meteo.com", timeout: int = 30):
        self.base_url, self.timeout = base_url.rstrip("/"), timeout

    def daily_ghi(self, lat: float, lon: float, start: date, end: date) -> Dict[str, float]:
        import requests
        r = requests.get(f"{self.base_url}/v1/archive",
                         params={"latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
                                 "daily": "shortwave_radiation_sum", "timezone": "auto"}, timeout=self.timeout)
        r.raise_for_status()
        body = r.json().get("daily", {})
        return {d_: float(v) / 3.6 for d_, v in zip(body.get("time", []), body.get("shortwave_radiation_sum", [])) if v is not None}


def build_provider(name: str, cfg) -> Optional[object]:
    if name == "nasa_power" and cfg.nasa_power:
        return NasaPower(cfg.nasa_power_base_url, cfg.timeout_seconds)
    if name == "meteonorm" and cfg.meteonorm:
        tok = os.environ.get(cfg.meteonorm_token_env, "")
        return Meteonorm(cfg.meteonorm_base_url, tok, cfg.timeout_seconds) if tok else None
    if name == "openmeteo" and cfg.openmeteo:
        return OpenMeteo(cfg.openmeteo_base_url, cfg.timeout_seconds)
    return None


# ---------------------------------------------------------------------------- 解析
def resolve_daily_irradiance(settings: Settings, store, station_id: str, days: List[date],
                             measured: Optional[Dict[date, float]] = None) -> Dict[date, Tuple[Optional[float], str]]:
    """返回 {日期: (kWh/m², 来源)}，按 providers 顺序回退；外部提供方按站缓存、失败告警后继续回退。"""
    cfg = settings.irradiance
    measured = measured or {}
    out: Dict[date, Tuple[Optional[float], str]] = {}
    if cfg.source == "none":
        return {d: (None, "none") for d in days}
    st = store.query("SELECT lat, lon FROM stations WHERE station_id=?", (station_id,))
    lat = float(st["lat"].iloc[0]) if not st.empty and pd.notna(st["lat"].iloc[0]) else None
    lon = float(st["lon"].iloc[0]) if not st.empty and pd.notna(st["lon"].iloc[0]) else None
    pending = list(days)
    for name in cfg.providers:
        if not pending:
            break
        if name == "measured":
            for d in list(pending):
                v = measured.get(d)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    out[d] = (float(v), "measured"); pending.remove(d)
            continue
        if name == "clearsky":
            if lat is None:
                continue
            for d in list(pending):
                out[d] = (clearsky_daily_kwh_m2(lat, lon, d, settings.baseline.tz_meridian_deg), "clearsky"); pending.remove(d)
            continue
        if name in EXTERNAL:
            prov = build_provider(name, cfg)
            if prov is None or lat is None or lon is None:
                continue
            key = f"irr:{name}:{station_id}"
            cache: Dict[str, float] = store.get_state(key) or {}
            missing = [d for d in pending if d.isoformat() not in cache and d < date.today()]
            if missing:
                try:
                    cache.update(prov.daily_ghi(lat, lon, min(missing), max(missing)))
                    cutoff = (date.today() - timedelta(days=cfg.cache_days)).isoformat()
                    cache = {k: v for k, v in cache.items() if k >= cutoff}
                    store.set_state(key, cache)
                except Exception as e:
                    store.alert(days[-1], "warn", "irradiance", f"{station_id} {name} 拉取失败，回退下一辐照源: {e}")
            for d in list(pending):
                if d.isoformat() in cache:
                    out[d] = (float(cache[d.isoformat()]), name); pending.remove(d)
    for d in pending:
        out[d] = (None, "none")
    return out
