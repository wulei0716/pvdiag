"""合成电站与故障注入——用来验证整条流水线的召回率/误报率（没有真实数据也能跑通）。

物理模型（够用即可，不追求气象精度）：
- 晴空辐照 G = 1000·sin(太阳高度角)^1.15，天气：clear / cloudy(云影抖动) / overcast(阴雨，散射光)
- 组件温度 T = T_amb + 0.03·G；电流 I = Imp·G/1000·(1+0.0005(T-25))·串固有系数(≈N(1,1%))
- MPPT 电压钳定：同一逆变器各串电压相同，V = n·Vmp·(1-0.0029(T-25))·f(G)
- 逆变器 pac = Σ I·V·η，受额定交流功率限幅 → "平顶"限发

可注入故障（string 级）：open_circuit / diode_bypass / shading_fixed / orientation / dust_local / stuck /
                        wrong_profile(串长档案错) / archived_shading(已备案遮挡) / degradation_step /
                        outage_intermittent(间歇停机，Kaggle Plant 2 真实签名) / stale_partial(半天钉死，NREL 标注签名) /
                        azimuth_actual(实际朝向不同；archive_correct 决定档案是否如实) / tracker_stuck(跟踪支架卡滞，需 mount=tracker)
（inverter 级）：mppt_fault（整机低发）/ limit（限发）
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .. import solar as _solar


@dataclass
class Fault:
    scope: str            # string | inverter
    target: str
    kind: str
    start_day: int        # 第几天开始（0-based）
    params: Dict = field(default_factory=dict)
    end_day: Optional[int] = None   # 修复日（含）之后恢复正常；None = 持续


@dataclass
class SimConfig:
    station_id: str = "ST01"
    station_name: str = "合成电站01"
    n_inverters: int = 3
    strings_per_inverter: int = 12
    n_modules: int = 22
    module_wp: float = 550.0
    module_vmp: float = 41.5
    module_imp: float = 13.25
    rated_ac_kw: float = 110.0          # 额定交流（DC 约 145kWp → 晴天会限幅，制造平顶）
    lat_deg: float = 31.2
    start: date = date(2026, 9, 1)
    days: int = 60
    seed: int = 7
    weather_probs: Tuple[float, float, float] = (0.55, 0.28, 0.17)   # clear, cloudy, overcast
    faults: List[Fault] = field(default_factory=list)
    roof_areas: Tuple[str, ...] = ("A区屋面", "B区屋面", "C区屋面")
    mount: str = "fixed"                # fixed | tracker（水平单轴跟踪：清晨/傍晚增益，曲线平顶）
    azimuth: float = 180.0
    tilt: float = 15.0


def default_faults(cfg: SimConfig) -> List[Fault]:
    """一组覆盖主要签名的默认故障（第 35 天起注入，前 30+ 天留给基线）"""
    inv = lambda k: f"{cfg.station_id}-INV{k:02d}"
    s = lambda k, n: f"{inv(k)}-pv{n}"
    return [
        Fault("string", s(1, 3), "open_circuit", 36),
        Fault("string", s(1, 5), "diode_bypass", 35, {"start": "10:00", "end": "13:00", "depth": 0.33}),
        Fault("string", s(1, 8), "shading_fixed", 34, {"start": "14:30", "end": "17:30", "depth": 0.5}),
        Fault("string", s(2, 2), "orientation", 0, {"clear_factor": 0.94}),
        Fault("string", s(2, 6), "dust_local", 20, {"rate_per_day": 0.006, "floor": 0.80, "rain_recover": 0.5}),
        Fault("string", s(2, 9), "stuck", 38, {"value": 8.0}),
        Fault("string", s(2, 11), "wrong_profile", 0, {"n_modules": 20}),
        Fault("string", s(3, 4), "archived_shading", 0, {"start": "09:30", "end": "11:00", "depth": 0.3}),
        Fault("string", s(3, 7), "degradation_step", 40, {"factor": 0.955}),   # 全天均匀 -4.5%（D 级边界）
        Fault("string", s(3, 10), "outage_intermittent", 37, {"prob": 0.6, "hours": 2.0}),
        Fault("string", s(1, 11), "stale_partial", 39, {"start": "10:30", "hours": 3.0}),
        Fault("inverter", inv(3), "mppt_fault", 42, {"factor": 0.92}),
    ]


# ---------------------------------------------------------------------------- 物理
def _solar_elevation(lat_deg: float, doy: int, minutes: np.ndarray) -> np.ndarray:
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365.0)
    lat = math.radians(lat_deg)
    hour_angle = np.radians((minutes / 60.0 - 12.0) * 15.0)
    sin_el = math.sin(lat) * math.sin(decl) + math.cos(lat) * math.cos(decl) * np.cos(hour_angle)
    return np.arcsin(np.clip(sin_el, -1, 1))


def _hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _day_irradiance(rng: np.random.Generator, weather: str, elev: np.ndarray) -> np.ndarray:
    clear = 1000.0 * np.power(np.clip(np.sin(elev), 0, None), 1.15)
    n = len(clear)
    if weather == "clear":
        return clear * (1 + rng.normal(0, 0.004, n))
    if weather == "cloudy":
        # 慢变随机游走 + 云影脉冲
        rw = np.cumsum(rng.normal(0, 0.03, n))
        base = 0.55 + 0.3 * np.sin(rw)
        pulses = rng.random(n) < 0.08
        base = np.where(pulses, base * rng.uniform(0.3, 0.7, n), base)
        return clear * np.clip(base, 0.15, 1.05)
    # overcast：散射光，平滑、低
    return clear * np.clip(0.18 + 0.06 * np.sin(np.linspace(0, 3, n)) + rng.normal(0, 0.01, n), 0.08, 0.3)


# ---------------------------------------------------------------------------- 主函数
def simulate(cfg: SimConfig) -> Dict[str, pd.DataFrame]:
    """返回 dict: stations, inverters, strings(档案), events, samples_strings, samples_inverters, weather, truth"""
    rng = np.random.default_rng(cfg.seed)
    faults = cfg.faults or default_faults(cfg)
    inv_ids = [f"{cfg.station_id}-INV{k:02d}" for k in range(1, cfg.n_inverters + 1)]
    minutes = np.arange(0, 24 * 60, 5)
    n_t = len(minutes)

    # ---- 档案
    strings_rows, string_factor, string_nmod, inv_strings = [], {}, {}, {}
    sf_by_string: Dict[str, List[Fault]] = {}
    for f in faults:
        if f.scope == "string":
            sf_by_string.setdefault(f.target, []).append(f)
    for k, inv in enumerate(inv_ids):
        inv_strings[inv] = []
        for n in range(1, cfg.strings_per_inverter + 1):
            sid = f"{inv}-pv{n}"
            inv_strings[inv].append(sid)
            string_factor[sid] = float(rng.normal(1.0, 0.01))
            nm = cfg.n_modules
            known, exp_dev, note = 0, None, ""
            az_doc, tilt_doc = cfg.azimuth, cfg.tilt
            for f in sf_by_string.get(sid, []):
                if f.kind == "wrong_profile":
                    nm = int(f.params.get("n_modules", nm))        # 真实串长少两块，但档案仍记 cfg.n_modules
                if f.kind == "archived_shading":
                    known, exp_dev, note = 1, -round(f.params.get("depth", 0.3) * 0.25, 3), "备案：东侧广告牌上午遮挡"
                if f.kind == "azimuth_actual" and f.params.get("archive_correct", True):
                    az_doc, tilt_doc = f.params.get("azimuth", 90.0), f.params.get("tilt", cfg.tilt)   # 档案如实登记 → 系统应不报警
            string_nmod[sid] = nm
            strings_rows.append({
                "string_id": sid, "station_id": cfg.station_id, "inverter_id": inv, "mppt": (n - 1) // 2 + 1,
                "pv_index": n, "n_modules": cfg.n_modules, "module_model": "SIM-550M", "module_wp": cfg.module_wp,
                "module_vmp": cfg.module_vmp, "module_imp": cfg.module_imp, "azimuth": az_doc, "tilt": tilt_doc,
                "roof_area": cfg.roof_areas[k % len(cfg.roof_areas)], "known_shading": known,
                "shading_expected_dev": exp_dev, "shading_note": note, "notes": "",
            })
    strings_df = pd.DataFrame(strings_rows)
    dc_kwp = cfg.strings_per_inverter * cfg.n_modules * cfg.module_wp / 1000.0
    inverters_df = pd.DataFrame([{"inverter_id": inv, "station_id": cfg.station_id, "name": inv, "model": "SIM-110K",
                                  "rated_ac_kw": cfg.rated_ac_kw, "dc_kwp": dc_kwp} for inv in inv_ids])
    stations_df = pd.DataFrame([{"station_id": cfg.station_id, "name": cfg.station_name, "platform": "csv",
                                 "capacity_kwp": dc_kwp * cfg.n_inverters, "region": "华东", "lat": cfg.lat_deg, "lon": 121.4,
                                 "extra_json": json.dumps({"mount": cfg.mount}, ensure_ascii=False)}])

    # ---- 天气序列
    weathers = rng.choice(["clear", "cloudy", "overcast"], size=cfg.days, p=list(cfg.weather_probs))
    rain_days = (weathers == "overcast") & (rng.random(cfg.days) < 0.6)

    s_frames, i_frames, w_rows, truth = [], [], [], []
    days_since_rain = 0
    soil: Dict[Tuple[str, str], float] = {}
    for di in range(cfg.days):
        d = cfg.start + timedelta(days=di)
        ts = pd.to_datetime([datetime.combine(d, datetime.min.time()) + timedelta(minutes=int(m)) for m in minutes])
        elev = _solar_elevation(cfg.lat_deg, d.timetuple().tm_yday, minutes)
        G = _day_irradiance(rng, weathers[di], elev)
        G = np.where(elev > 0, G, 0.0)
        doy = d.timetuple().tm_yday
        # 几何因子：相对"参考朝向固定倾角"的倾斜面辐照比；阴天散射为主无方向性（因子 1），多云取半
        geo_w = {"clear": 1.0, "cloudy": 0.5, "overcast": 0.0}[weathers[di]]
        ref_poa = _solar.clear_sky_poa(cfg.lat_deg, doy, minutes, cfg.azimuth, cfg.tilt)
        def geo_factor(az, tilt):
            poa = _solar.clear_sky_poa(cfg.lat_deg, doy, minutes, az, tilt)
            r = np.where(ref_poa > 1, poa / np.maximum(ref_poa, 1), 1.0)
            return 1.0 + geo_w * (r - 1.0)
        tracker_gain = None
        if cfg.mount == "tracker":
            tp = _solar.tracker_poa(cfg.lat_deg, doy, minutes)
            r = np.where(ref_poa > 1, tp / np.maximum(ref_poa, 1), 1.0)
            tracker_gain = np.clip(1.0 + geo_w * (r - 1.0), 0.2, 3.0)
        t_amb = 18 + 10 * math.sin(2 * math.pi * (d.timetuple().tm_yday - 100) / 365) + 6 * np.sin((minutes / 60 - 9) / 24 * 2 * math.pi)
        t_mod = t_amb + 0.03 * G
        days_since_rain = 0 if rain_days[di] else days_since_rain + 1
        w_rows.append({"date": d.isoformat(), "weather": weathers[di], "rain": int(rain_days[di])})

        for inv in inv_ids:
            inv_faults = [f for f in faults if f.scope == "inverter" and f.target == inv and di >= f.start_day
                          and (f.end_day is None or di < f.end_day)]
            inv_factor = 1.0
            for f in inv_faults:
                if f.kind == "mppt_fault":
                    inv_factor *= f.params.get("factor", 0.92)
            # 钳定电压（取标准串长）
            v_clamp = cfg.n_modules * cfg.module_vmp * (1 - 0.0029 * (t_mod - 25)) * np.clip(0.86 + 0.14 * G / 1000.0, 0, 1)
            v_clamp = np.where(G > 15, v_clamp, 0.0)
            currents: Dict[str, np.ndarray] = {}
            voltages: Dict[str, np.ndarray] = {}
            for sid in inv_strings[inv]:
                I = cfg.module_imp * G / 1000.0 * (1 + 0.0005 * (t_mod - 25)) * string_factor[sid] * inv_factor
                I = I * (1 + rng.normal(0, 0.004, n_t))
                if tracker_gain is not None:
                    I = I * tracker_gain
                V = v_clamp.copy()
                for f in sf_by_string.get(sid, []):
                    if di < f.start_day or (f.end_day is not None and di >= f.end_day):
                        continue
                    p = f.params
                    if f.kind == "azimuth_actual":
                        # 该串实际朝向/倾角与参考不同（物理事实）；档案对不对由 archive_correct 决定
                        I = I * geo_factor(p.get("azimuth", 90.0), p.get("tilt", cfg.tilt))
                    elif f.kind == "tracker_stuck":
                        # 跟踪支架卡在水平位：失去清晨/傍晚增益，曲线回到固定倾角的钟形
                        if tracker_gain is not None:
                            I = I / tracker_gain * geo_factor(cfg.azimuth, p.get("stuck_tilt", 0.0))
                    elif f.kind == "open_circuit":
                        I = np.zeros(n_t); V = np.where(G > 15, v_clamp * 1.18, 0.0)
                    elif f.kind in ("diode_bypass", "shading_fixed", "archived_shading"):
                        band = (minutes >= _hhmm(p["start"])) & (minutes < _hhmm(p["end"]))
                        depth = p.get("depth", 0.33) if weathers[di] != "overcast" else p.get("depth", 0.33) * 0.15
                        I = np.where(band, I * (1 - depth), I)
                    elif f.kind == "orientation":
                        fac = {"clear": p.get("clear_factor", 0.94), "cloudy": 0.97, "overcast": 1.0}[weathers[di]]
                        I = I * fac
                    elif f.kind == "dust_local":
                        # 积灰逐日累积，雨天冲掉一部分（之后从冲洗后的水平继续累积）
                        key = ("dust", sid)
                        acc = soil.get(key, 0.0)
                        if di == f.start_day:
                            acc = 0.0
                        acc = min(acc + p.get("rate_per_day", 0.004), 1 - p.get("floor", 0.85))
                        if rain_days[di]:
                            acc *= (1 - p.get("rain_recover", 0.7))
                        soil[key] = acc
                        I = I * (1 - acc)
                    elif f.kind == "stuck":
                        I = np.where(G > 15, p.get("value", 8.0), 0.0)
                    elif f.kind == "wrong_profile":
                        # 实际少两块组件：电压低 ~9%，电流略偏（偏离 MPP）
                        V = v_clamp * (string_nmod[sid] / cfg.n_modules); I = I * 0.985
                    elif f.kind == "degradation_step":
                        I = I * p.get("factor", 0.955)
                    elif f.kind == "outage_intermittent":
                        # 间歇停机/掉线：每天以 prob 概率在白天随机时段整段掉到 0（时长 hours），其余时间正常
                        if rng.random() < p.get("prob", 0.6):
                            start_h = float(rng.uniform(p.get("earliest_h", 9.0), p.get("latest_h", 14.0)))
                            band = (minutes >= start_h * 60) & (minutes < (start_h + p.get("hours", 2.0)) * 60)
                            I = np.where(band, 0.0, I)
                    elif f.kind == "stale_partial":
                        # 半天钉死：从 start 时刻起电流冻结在当时值（采集器冻结），hours 小时后恢复
                        k0 = int(_hhmm(p.get("start", "10:30")) // 5)
                        k1 = min(n_t, k0 + int(p.get("hours", 3.0) * 12))
                        I = I.copy(); I[k0:k1] = I[k0]
                currents[sid] = np.clip(I, 0, None)
                voltages[sid] = V
            # 逆变器功率 + 限幅（平顶）
            p_dc = sum(currents[s] * voltages[s] for s in inv_strings[inv]) / 1000.0
            pac = p_dc * 0.985
            clip = pac > cfg.rated_ac_kw
            if clip.any():
                scale = np.where(clip, cfg.rated_ac_kw / np.maximum(pac, 1e-6), 1.0)
                pac = np.minimum(pac, cfg.rated_ac_kw)
                for s in inv_strings[inv]:          # 限幅时逆变器偏离 MPP：电流下降、电压上升
                    currents[s] = currents[s] * scale
                    voltages[s] = voltages[s] * np.where(clip, 1.04, 1.0)
            for s in inv_strings[inv]:
                s_frames.append(pd.DataFrame({"string_id": s, "ts": ts,
                                              "current_a": np.round(currents[s], 3), "voltage_v": np.round(voltages[s], 1)}))
            i_frames.append(pd.DataFrame({"inverter_id": inv, "ts": ts, "pac_kw": np.round(pac, 3),
                                          "status": np.where(G > 15, "running", "standby")}))

    for f in faults:
        truth.append({"scope": f.scope, "target": f.target, "kind": f.kind,
                      "start_date": (cfg.start + timedelta(days=f.start_day)).isoformat(),
                      "end_date": (cfg.start + timedelta(days=f.end_day)).isoformat() if f.end_day is not None else None,
                      "params": str(f.params)})
    return {
        "stations": stations_df, "inverters": inverters_df, "strings": strings_df,
        "events": pd.DataFrame(columns=["scope", "target_id", "event_type", "event_date", "notes"]),
        "samples_strings": pd.concat(s_frames, ignore_index=True),
        "samples_inverters": pd.concat(i_frames, ignore_index=True),
        "weather": pd.DataFrame(w_rows), "truth": pd.DataFrame(truth),
    }


def write_csv_bundle(out_dir, data: Dict[str, pd.DataFrame], station_id: str) -> None:
    """按 CsvAdapter 目录约定落盘，方便演示 pvdiag ingest --source csv"""
    from pathlib import Path
    root = Path(out_dir)
    (root / station_id / "strings").mkdir(parents=True, exist_ok=True)
    (root / station_id / "inverters").mkdir(parents=True, exist_ok=True)
    ss = data["samples_strings"]
    for d, part in ss.groupby(ss["ts"].dt.date):
        part.to_parquet(root / station_id / "strings" / f"{d}.parquet", index=False)
    si = data["samples_inverters"]
    for d, part in si.groupby(si["ts"].dt.date):
        part.to_parquet(root / station_id / "inverters" / f"{d}.parquet", index=False)
    data["stations"].to_csv(root / "stations.csv", index=False)
    data["inverters"].to_csv(root / "inverters.csv", index=False)
    data["strings"].to_csv(root / "strings.csv", index=False)
    data["truth"].to_csv(root / "truth.csv", index=False)
    data["weather"].to_csv(root / "weather_truth.csv", index=False)
