"""日电量模式：只有"每串每日发电量（kWh）"时也能跑诊断。

真实运维里最容易拿到的不是 5 分钟电流曲线，而是平台导出/巡检报告里的"每串每日 kWh"宽表
（行=逆变器×通道，列=日期，另有 状态 列：接入/零电流/未接）。本模块：
  1. read_report_xlsx  读这种宽表（含"逐日发电概况"的全站中午功率/占晴峰%/天气），或 read_long_csv 读长表
  2. load_into_store   档案（stations/inverters/strings）+ daily_features.day_energy_kwh + station_daily（天气标签）
  3. diagnose          同机日电量互比（晴好天）：比值=本串/同机接入串中位数（有档案按组件数归一），
                       连续低发天数分级（证据比曲线弱：互比最高到 B；零电流且档案确认接线才到 A），
                       通道三态：接入 / 空输入（同 MPPT 另一路在发电：华为 2 路共 1 MPPT，不是故障）/ 零电流待核实 / 未接
                       疑双串并联（比值 ≥1.6 且档案无并联数）→ 不当异常、也不进中位数
     结果写 suspects（signature_json.notes 标注"日电量模式"），后面的复核/工单/派单链路照常接。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .models import Cause, Grade, Signature, SuspectStatus

WEATHER_MAP = {"晴好": "clear", "晴": "clear", "多云": "cloudy", "阴雨": "overcast", "阴": "overcast", "雨": "overcast"}
STATUS_MAP = {"接入": "connected", "零电流": "zero_current", "未接": "unconnected", "—": "absent", "-": "absent", "": "absent"}
INPUTS_PER_MPPT_DEFAULT = 2        # 华为 SUN2000-100/110KTL、阳光 SG110CX 都是每 MPPT 两路输入


@dataclass
class DailyEnergyBundle:
    station: Dict
    inverters: pd.DataFrame
    strings: pd.DataFrame            # string_id, station_id, inverter_id, mppt, pv_index, status
    energy: pd.DataFrame             # string_id, date(ISO), energy_kwh
    station_daily: pd.DataFrame      # date, noon_power, peak_ref, weather_tag, energy_kwh
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- 读取
def _year_from_title(text: str, default: int) -> int:
    m = re.search(r"(20\d{2})-\d{1,2}-\d{1,2}", text or "")
    return int(m.group(1)) if m else default


def _parse_md(s, year: int) -> Optional[str]:
    """'10-07' / '2026-10-07' / datetime → ISO 日期"""
    if isinstance(s, (datetime, pd.Timestamp)):
        return s.date().isoformat()
    s = str(s).strip()
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", s)
    if m:
        return date(year, int(m.group(1)), int(m.group(2))).isoformat()
    m = re.fullmatch(r"(20\d{2})-(\d{1,2})-(\d{1,2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    return None


def _norm_inv(name: str) -> str:
    """'逆变器2' → 'INV02'；'NB01' 原样"""
    s = str(name).strip()
    m = re.fullmatch(r"逆变器\s*(\d+)", s)
    return f"INV{int(m.group(1)):02d}" if m else s


def _norm_ch(name: str) -> Tuple[str, Optional[int]]:
    """'MPPT1' / 'pv16' / 'PV16' → ('pv16', 16)（MPPT 通道按通道号当 pv_index）"""
    s = str(name).strip()
    m = re.fullmatch(r"(?:MPPT|mppt|pv|PV)\s*(\d+)", s)
    return (f"pv{int(m.group(1))}", int(m.group(1))) if m else (s, None)


def read_report_xlsx(path, station_id: str, platform: str = "", inputs_per_mppt: int = INPUTS_PER_MPPT_DEFAULT,
                     channel_is_mppt: Optional[bool] = None) -> DailyEnergyBundle:
    """读巡检报告工作簿：宽表 sheet（列：逆变器/设备, 组串/通道, 状态, [偏差%], 日期..., 合计, 日均）+ 概况表（日期, 中午功率kW, 占晴峰%, 天气, 日发电量kWh）"""
    xl = pd.ExcelFile(path)
    year = date.today().year
    wide = None; daily = None; title = ""; cap = None; name = station_id
    for sh in xl.sheet_names:
        df = xl.parse(sh, header=None)
        if df.empty:
            continue
        t0 = str(df.iloc[0, 0]) if pd.notna(df.iloc[0, 0]) else ""
        year = _year_from_title(t0, year)
        if not title:
            title = t0
            m = re.search(r"(.+?)\s*[·•]", t0)
            name = m.group(1).strip() if m else name
            m = re.search(r"([\d.]+)\s*(MW|kW)", t0)
            if m:
                cap = float(m.group(1)) * (1000 if m.group(2) == "MW" else 1)
        for i in range(min(len(df), 8)):
            row = [str(x).strip() for x in df.iloc[i].tolist()]
            if row[0] in ("逆变器", "设备") and row[1] in ("组串", "通道") and "状态" in row:
                wide = (df.iloc[i + 1:].reset_index(drop=True), row)
                break
        for i in range(len(df)):
            row = [str(x).strip() for x in df.iloc[i].tolist()]
            if row[0] == "日期" and any("中午功率" in c for c in row):
                j = i + 1; rows = []
                while j < len(df) and _parse_md(df.iloc[j, 0], year):
                    r = df.iloc[j].tolist(); rows.append(r); j += 1
                cols = row
                daily = pd.DataFrame(rows, columns=cols[:len(rows[0])] if rows else cols)
                break
    if wide is None:
        raise ValueError(f"{path}: 没找到每串每日发电量宽表（需要 逆变器/设备, 组串/通道, 状态, 日期... 列）")
    body, header = wide
    date_cols = [(k, _parse_md(h, year)) for k, h in enumerate(header)]
    date_cols = [(k, d) for k, d in date_cols if d]
    if channel_is_mppt is None:
        channel_is_mppt = header[1] == "通道" or body.iloc[:, 1].astype(str).str.upper().str.startswith("MPPT").any()
    s_rows, e_rows = [], []
    for _, r in body.iterrows():
        if pd.isna(r.iloc[0]) or str(r.iloc[0]).strip() in ("全站合计", "合计", "nan", ""):
            continue
        inv = _norm_inv(r.iloc[0]); ch, idx = _norm_ch(r.iloc[1])
        status = STATUS_MAP.get(str(r.iloc[2]).strip() if pd.notna(r.iloc[2]) else "", "absent")
        sid = f"{station_id}-{inv}-{ch}"
        mppt = idx if channel_is_mppt else (None if idx is None else (idx - 1) // inputs_per_mppt + 1)
        s_rows.append({"string_id": sid, "station_id": station_id, "inverter_id": f"{station_id}-{inv}", "mppt": mppt, "pv_index": idx,
                       "status": status, "channel_is_mppt": bool(channel_is_mppt)})
        for k, dd in date_cols:
            v = r.iloc[k]
            if pd.notna(v) and str(v).strip() not in ("", "nan"):
                try:
                    e_rows.append({"string_id": sid, "date": dd, "energy_kwh": float(v)})
                except ValueError:
                    pass
    strings = pd.DataFrame(s_rows)
    energy = pd.DataFrame(e_rows)
    inv_ids = strings["inverter_id"].unique().tolist()
    inverters = pd.DataFrame([{"inverter_id": i, "station_id": station_id, "name": i.split("-", 1)[1], "model": None} for i in inv_ids])
    sd_rows = []
    if daily is not None:
        cols = list(daily.columns)
        c_noon = next((c for c in cols if "中午功率" in c), None)
        c_pct = next((c for c in cols if "占晴峰" in c), None)
        c_w = next((c for c in cols if c.startswith("天气")), None)
        c_e = next((c for c in cols if "日发电量" in c), None)
        for _, r in daily.iterrows():
            dd = _parse_md(r["日期"], year)
            if not dd:
                continue
            noon = pd.to_numeric(r.get(c_noon), errors="coerce") if c_noon else np.nan
            pct = pd.to_numeric(r.get(c_pct), errors="coerce") if c_pct else np.nan
            peak = noon / (pct / 100.0) if pd.notna(noon) and pd.notna(pct) and pct > 0 else np.nan
            wtxt = str(r.get(c_w) or "") if c_w else ""
            tag = next((v for k, v in WEATHER_MAP.items() if k in wtxt), None)
            if tag is None:
                tag = "clear" if pd.notna(pct) and pct >= 35 else ("overcast" if pd.notna(pct) else "nodata")
            sd_rows.append({"date": dd, "noon_power": noon, "peak_ref": peak, "weather_tag": tag,
                            "energy_kwh": pd.to_numeric(r.get(c_e), errors="coerce") if c_e else np.nan})
    station_daily = pd.DataFrame(sd_rows)
    if station_daily.empty and not energy.empty:
        # 没有概况表：用全站日电量相对期内最大值判天气（≥35% 晴好）
        tot = energy.groupby("date")["energy_kwh"].sum()
        station_daily = pd.DataFrame({"date": tot.index, "noon_power": np.nan, "peak_ref": np.nan,
                                      "weather_tag": np.where(tot.values >= 0.35 * tot.max(), "clear", "overcast"), "energy_kwh": tot.values})
    notes = [f"来源文件 {Path(path).name}", f"标题 {title[:60]}"]
    return DailyEnergyBundle(station={"station_id": station_id, "name": name, "platform": platform, "capacity_kwp": cap},
                             inverters=inverters, strings=strings, energy=energy, station_daily=station_daily, notes=notes)


def read_long_csv(path, station_id: str, platform: str = "", inputs_per_mppt: int = INPUTS_PER_MPPT_DEFAULT) -> DailyEnergyBundle:
    """长表 CSV：inverter_id, string_id|pv_index, date, energy_kwh[, status]；天气按全站日电量判"""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    inv = df[cols["inverter_id"]].astype(str).map(_norm_inv)
    if "string_id" in cols:
        ch = df[cols["string_id"]].astype(str).map(lambda s: _norm_ch(s.rsplit("-", 1)[-1]))
    else:
        ch = df[cols["pv_index"]].map(lambda i: (f"pv{int(i)}", int(i)))
    sid = [f"{station_id}-{a}-{b[0]}" for a, b in zip(inv, ch)]
    energy = pd.DataFrame({"string_id": sid, "date": pd.to_datetime(df[cols["date"]]).dt.date.astype(str),
                           "energy_kwh": pd.to_numeric(df[cols["energy_kwh"]], errors="coerce")}).dropna()
    st = pd.DataFrame({"string_id": sid, "inverter_id": [f"{station_id}-{a}" for a in inv], "pv_index": [b[1] for b in ch],
                       "status": df[cols["status"]].map(lambda s: STATUS_MAP.get(str(s), "connected")) if "status" in cols else "connected"})
    st = st.drop_duplicates("string_id")
    st["station_id"] = station_id
    st["mppt"] = st["pv_index"].map(lambda i: None if pd.isna(i) else (int(i) - 1) // inputs_per_mppt + 1)
    st["channel_is_mppt"] = False
    inverters = pd.DataFrame([{"inverter_id": i, "station_id": station_id, "name": i.split("-", 1)[1], "model": None} for i in st["inverter_id"].unique()])
    tot = energy.groupby("date")["energy_kwh"].sum()
    station_daily = pd.DataFrame({"date": tot.index, "noon_power": np.nan, "peak_ref": np.nan,
                                  "weather_tag": np.where(tot.values >= 0.35 * tot.max(), "clear", "overcast"), "energy_kwh": tot.values})
    return DailyEnergyBundle(station={"station_id": station_id, "name": station_id, "platform": platform, "capacity_kwp": None},
                             inverters=inverters, strings=st, energy=energy, station_daily=station_daily)


# ---------------------------------------------------------------------------- 落库
def load_into_store(store, b: DailyEnergyBundle, keep_profile: bool = True) -> Dict[str, int]:
    """档案 + 日电量 + 天气落库。keep_profile=True 时不覆盖已有 strings 行的档案字段（n_modules/朝向/并联数等），只补状态"""
    from .agents.profile import ProfileAgent
    st = dict(b.station); st["extra_json"] = json.dumps({"mode": "daily_energy"}, ensure_ascii=False)
    store.upsert_df("stations", pd.DataFrame([st]))
    store.upsert_df("inverters", b.inverters)
    existing = store.strings(b.station["station_id"])
    have = set(existing["string_id"]) if not existing.empty else set()
    s = b.strings.copy()
    s["notes"] = s["status"].map(lambda x: f"daily_energy:{x}")
    s = s.drop(columns=["status", "channel_is_mppt"])
    new = s[~s["string_id"].isin(have)] if keep_profile else s
    n_new = ProfileAgent.import_strings(store, new) if not new.empty else 0
    if keep_profile and have:
        for sid, note in s[s["string_id"].isin(have)][["string_id", "notes"]].itertuples(index=False):
            store.execute("UPDATE strings SET notes=? WHERE string_id=?", (note, sid))
    feats = b.energy.copy()
    feats["zero_all_day"] = (feats["energy_kwh"] <= 0).astype(int)
    feats = feats.rename(columns={"energy_kwh": "day_energy_kwh"})
    feats["n_samples"] = 1; feats["stuck"] = 0
    store.upsert_df("daily_features", feats)
    sd = b.station_daily.copy(); sd["station_id"] = b.station["station_id"]
    sd["irradiance_source"] = "none"
    store.upsert_df("station_daily", sd[["station_id", "date", "noon_power", "peak_ref", "weather_tag", "irradiance_source"]])
    return {"strings": int(len(s)), "new_strings": int(n_new), "energy_rows": int(len(feats)), "days": int(sd["date"].nunique())}


# ---------------------------------------------------------------------------- 诊断
def _status_of(row: pd.Series) -> str:
    n = str(row.get("notes") or "")
    return n.split("daily_energy:", 1)[1] if "daily_energy:" in n else "connected"


def diagnose(store, station_id: str, start: date, end: date, th, parallel_ratio: float = 1.6, severe_floor: float = -0.20,
             min_group: int = 3, group_mode: str = "auto", min_clear_days: int = 5, write: bool = True) -> Dict:
    """日电量互比诊断。返回 {suspects, strings(逐串统计), channel_states, clear_days}"""
    strings = store.strings(station_id)
    if strings.empty:
        return {"suspects": [], "strings": pd.DataFrame(), "channel_states": {}, "clear_days": []}
    strings = strings.set_index("string_id")
    sd = store.station_daily(station_id, start, end)
    clear_days = sorted(sd.loc[sd["weather_tag"] == "clear", "date"].tolist())
    feats = store.daily_features(station_id, start, end)[["string_id", "date", "day_energy_kwh"]]
    feats = feats[feats["string_id"].isin(strings.index)]
    E = feats.pivot(index="string_id", columns="date", values="day_energy_kwh").reindex(strings.index)
    status = strings.apply(_status_of, axis=1)
    inv_of = strings["inverter_id"]
    # 归一：有组件数按组件数（双串并联档案 n_modules 翻倍即可自然归一）；再有 n_parallel 列也乘上
    norm = pd.Series(1.0, index=strings.index)
    if "n_modules" in strings.columns:
        nm = pd.to_numeric(strings["n_modules"], errors="coerce")
        if nm.notna().any():
            norm = norm * (nm / nm[nm > 0].median()).fillna(1.0).replace(0, 1.0)
    if "n_parallel" in strings.columns:
        npar = pd.to_numeric(strings["n_parallel"], errors="coerce").fillna(1.0).replace(0, 1.0)
        norm = norm * npar
    En = E.div(norm, axis=0)

    # ---- 通道三态（同 MPPT 另一路在发电 → 空输入，不是故障；华为/阳光每 MPPT 两路）
    produced = (E.fillna(0) > 0).any(axis=1)
    # 全站某路编号在任何一台逆变器上都没发过电 → 平台多报的虚拟通道/整站未启用的输入，不当零电流
    idx_series = pd.to_numeric(strings["pv_index"], errors="coerce") if "pv_index" in strings.columns else pd.Series(np.nan, index=strings.index)
    idx_ever = produced.groupby(idx_series).any()
    idx_count = idx_series.value_counts()                        # 该路编号在多少台逆变器上出现
    channel_states: Dict[str, str] = {}
    for sid in strings.index:
        st_ = status[sid]
        if st_ in ("absent", "unconnected"):
            channel_states[sid] = st_; continue
        if produced[sid]:
            channel_states[sid] = "connected"; continue
        ix = idx_series[sid]
        if pd.notna(ix) and ix in idx_ever.index and not idx_ever[ix] and idx_count.get(ix, 0) >= 3:
            channel_states[sid] = "absent"; continue
        # 零电流：看同机同 MPPT 的兄弟路
        mp = strings.at[sid, "mppt"] if "mppt" in strings.columns else None
        sib = strings[(inv_of == inv_of[sid]) & (strings["mppt"] == mp)].index.difference([sid]) if mp is not None and pd.notna(mp) else []
        sib_prod = any(produced.get(x, False) for x in sib)
        nmod = pd.to_numeric(strings.at[sid, "n_modules"], errors="coerce") if "n_modules" in strings.columns else np.nan
        if sib_prod and not (pd.notna(nmod) and nmod > 0):
            channel_states[sid] = "empty_input"          # 共 MPPT 另一路有电：大概率这一路没接线，档案也没说接了
        elif pd.notna(nmod) and nmod > 0:
            channel_states[sid] = "zero_current_wired"   # 档案说接了却 0 电量 → 断线/保险丝 A 级
        else:
            channel_states[sid] = "zero_current"         # 待核实
    # ---- 互比分组：默认同机；同机能互比的串太少（小站/双串单串混接）→ 改"同编号通道跨设备"互比
    connected = [s_ for s_ in strings.index if channel_states.get(s_) == "connected"]
    overcast_days = sorted(sd.loc[sd["weather_tag"] == "overcast", "date"].tolist())
    all_days = [d for d in clear_days + overcast_days if d in En.columns]

    def _groups(mode: str) -> Dict[str, List[str]]:
        g: Dict[str, List[str]] = {}
        key = inv_of[connected] if mode == "inverter" else idx_series[connected]
        for k, grp in strings.loc[connected].groupby(key):
            g[f"{'inv' if mode == 'inverter' else 'ch'}:{k if mode == 'inverter' else int(k)}"] = list(grp.index)
        return g

    def _dev(groups: Dict[str, List[str]]):
        """每天每组：剔除 ≥parallel_ratio× 中位的疑并联串后，比值−1；同时给出该组当天的稳健标准差（噪声）"""
        dev = pd.DataFrame(np.nan, index=strings.index, columns=all_days)
        noise = pd.DataFrame(np.nan, index=strings.index, columns=all_days)
        par: Dict[str, int] = {}
        for g, members in groups.items():
            for d in all_days:
                v = En.loc[members, d].dropna(); v = v[v > 0]
                if len(v) < min_group:
                    continue
                med = float(v.median())
                big = v[v >= parallel_ratio * med]
                for s_ in big.index:
                    par[s_] = par.get(s_, 0) + 1
                v2 = v.drop(big.index)
                if len(v2) < min_group:
                    continue
                med2 = float(v2.median())
                if med2 <= 0:
                    continue
                dv_ = v2 / med2 - 1.0
                dev.loc[v2.index, d] = dv_
                # 群组稳健标准差只在成员 ≥6 时才有意义（小组的 MAD 被个别异常串撑大，会把真低发门控掉）
                noise.loc[v2.index, d] = float(1.4826 * np.median(np.abs(dv_ - np.median(dv_)))) if len(dv_) >= 6 else np.nan
        return dev, noise, par

    group_mode_used = "inverter" if group_mode != "channel" else "channel"
    if group_mode == "auto" and connected and idx_series[connected].notna().any():
        # 同机组太小（典型 <5 串：小站、或双串/单串混接剔除后所剩无几）而"同编号通道跨设备"组更大 → 用通道组
        inv_size = float(np.median([len(m) for m in _groups("inverter").values()]))
        ch_size = float(np.median([len(m) for m in _groups("channel").values()]))
        if inv_size < 5 and ch_size >= max(min_group, inv_size):
            group_mode_used = "channel"
    groups = _groups(group_mode_used)
    dev, noise, parallel_suspect = _dev(groups)
    if group_mode == "auto" and connected and group_mode_used == "inverter":
        covered = int(dev.loc[connected].notna().any(axis=1).sum())
        if covered < 0.5 * len(connected) and idx_series[connected].notna().any():
            groups = _groups("channel"); group_mode_used = "channel"
            dev, noise, parallel_suspect = _dev(groups)
    group_of = {s_: g for g, m in groups.items() for s_ in m}
    dev_clear = dev[[d for d in clear_days if d in dev.columns]]
    noise_clear = noise[[d for d in clear_days if d in dev.columns]]
    dev_over = dev[[d for d in overcast_days if d in dev.columns]]
    # ---- 逐串统计与分级
    rows, suspects = [], []
    n_clear = len([d for d in clear_days if d in En.columns])
    for sid in strings.index:
        stt = channel_states[sid]
        dv = dev_clear.loc[sid].dropna()
        nz = noise_clear.loc[sid].reindex(dv.index).fillna(0.0)
        ov = dev_over.loc[sid].dropna()
        over_dev = float(ov.mean()) if len(ov) >= 1 else np.nan
        dev_std = float(dv.std()) if len(dv) >= 3 else np.nan
        g_noise = float(nz.median()) if len(nz) else 0.0
        # 群组噪声门控（同主流水线）：一次"低发"要同时 ≤ fault_floor 且 ≤ −k×当日群组稳健标准差
        sig_low = (dv <= th.fault_floor) & (dv <= -th.noise_k * nz)
        low_days = int(sig_low.sum()); n_eval = int(len(dv))
        mean_dev = float(dv.mean()) if n_eval else np.nan
        streak = 0
        for x in sig_low.values[::-1]:
            if x:
                streak += 1
            else:
                break
        grade, cause, notes = None, None, [f"日电量模式（无曲线签名，证据弱于 5 分钟互比；分组={group_mode_used}，群组噪声 {g_noise:.1%}）"]
        is_par = parallel_suspect.get(sid, 0) >= max(1, n_clear // 2)
        constant = pd.notna(dev_std) and dev_std <= 0.02 and pd.notna(over_dev) and abs(over_dev - mean_dev) <= 0.04
        if stt == "zero_current_wired":
            grade, cause = Grade.A.value, Cause.OPEN_CIRCUIT.value
            notes.append("档案已登记组件但全期 0 电量，同机其它串正常 → 断线/保险丝/未启用")
        elif stt == "empty_input":
            notes.append("同 MPPT 另一路在发电、档案未登记组件 → 判为空输入（共压读到电压、电流 0），不报故障；若现场确有接线请补档案")
        elif stt == "zero_current":
            notes.append("全期 0 电量且无档案、同 MPPT 两路都为 0 → 大概率整个 MPPT 未用；待现场核实接线，不单独报故障（通道三态表里列出）")
        elif is_par:
            notes.append(f"晴好天电量 ≥{parallel_ratio:.1f}× 群组中位（{parallel_suspect[sid]}/{n_clear} 天）→ 疑双串并联/档案缺并联数，不作异常，不进中位数")
        elif n_eval >= 1 and mean_dev <= severe_floor and low_days >= max(th.fault_persist_days, int(math.ceil(0.8 * n_eval))):
            if -0.62 <= mean_dev <= -0.40:
                grade, cause = Grade.B.value, Cause.BYPASS_DIODE.value
                notes.append(f"晴好天群组互比均值 {mean_dev:+.1%}，{low_days}/{n_eval} 天低发，≈−50% 疑半串不发电（旁路二极管/半串断线/大面积遮挡）")
            elif constant and mean_dev > -0.40:
                grade, cause = Grade.C.value, Cause.PROFILE_ERROR.value
                notes.append(f"恒定偏低 {mean_dev:+.1%}（离散 {dev_std:.1%}，阴天 {over_dev:+.1%} 一样）→ 与天气无关的固定比例：疑串长/组件数不同或若干组件被旁路（电压型偏差，电流互比看不到）；核对档案与串电压")
            else:
                grade, cause = Grade.B.value, Cause.SHADING.value       # 日电量只有量没有形状：最高给 B，派单前请人工复核
                notes.append(f"晴好天群组互比均值 {mean_dev:+.1%}，{low_days}/{n_eval} 天低发，持续严重低发（遮挡/组件故障）")
        elif streak >= th.fault_persist_days and mean_dev <= th.fault_floor:
            if pd.notna(over_dev) and abs(over_dev) <= max(th.orientation_overcast_abs, 0.03) and pd.notna(dev_std) and dev_std <= 0.03:
                grade, cause = Grade.C.value, Cause.ORIENTATION.value      # 阴天散射光无方向性 → 偏差消失 = 朝向/倾角
                notes.append(f"晴好天恒定偏低 {mean_dev:+.1%}（离散 {dev_std:.1%}），阴天偏差 {over_dev:+.1%} 基本消失 → 疑朝向/倾角不同，核对档案")
            elif constant:
                grade, cause = Grade.C.value, Cause.PROFILE_ERROR.value
                notes.append(f"恒定偏低 {mean_dev:+.1%}（离散 {dev_std:.1%}，阴天 {over_dev:+.1%} 一样）→ 固定比例、与天气无关：疑串长/组件数不同或组件被旁路；核对档案与串电压")
            else:
                grade, cause = Grade.B.value, Cause.UNKNOWN.value
                notes.append(f"连续 {streak} 个晴好天显著低于群组中位（均值 {mean_dev:+.1%}" +
                             (f"，阴天 {over_dev:+.1%}" if pd.notna(over_dev) else "") + "，随天气变化 → 遮挡/积灰/组件故障）")
        elif n_eval >= th.fault_persist_days and mean_dev <= 2 * th.fault_floor and float((dv <= th.fault_floor).mean()) >= 0.8:
            # 群组噪声大（同机有一簇结构性低发串）时门控会把 −10%~−20% 的持续低发压成 D；这里按"幅度大且几乎每天都低"兜底
            if constant:
                grade, cause = Grade.C.value, Cause.PROFILE_ERROR.value
                notes.append(f"持续偏低 {mean_dev:+.1%} 且与天气无关（离散 {dev_std:.1%}，阴天 {over_dev:+.1%}）→ 疑串长/组件数不同或组件被旁路；核对档案与串电压（群组噪声 {g_noise:.1%}）")
            else:
                grade, cause = Grade.B.value, Cause.UNKNOWN.value
                notes.append(f"{int((dv <= th.fault_floor).sum())}/{n_eval} 个晴好天低于群组中位 ≥{-th.fault_floor:.0%}（均值 {mean_dev:+.1%}；群组噪声 {g_noise:.1%} 偏大，同机可能有一簇结构性低发串）")
        elif n_eval and mean_dev <= th.normal_floor and mean_dev <= -2.0 * g_noise and int((dv <= th.normal_floor).sum()) >= th.observe_persist_days:
            grade, cause = Grade.D.value, Cause.UNKNOWN.value
            notes.append(f"轻度偏低（均值 {mean_dev:+.1%}，群组噪声 {g_noise:.1%}），观察")
        if grade in (Grade.B.value, Grade.C.value) and n_eval < min_clear_days:
            notes.append(f"晴好天样本仅 {n_eval} 天（<{min_clear_days}），冷启动：分级上限 {th.cold_start_cap_grade}，等样本够了再定")
            grade = th.cold_start_cap_grade
        lost = 0.0
        if n_eval and sid in En.index:
            for d in dv.index:
                members = [m for m in groups.get(group_of.get(sid, ""), []) if not parallel_suspect.get(m)]
                med = float(En.loc[members, d].dropna().median()) if members else np.nan
                if pd.notna(med) and pd.notna(En.at[sid, d]):
                    lost += max(0.0, (med - En.at[sid, d]) * norm[sid])
        rows.append({"string_id": sid, "inverter_id": inv_of[sid], "group": group_of.get(sid), "state": stt, "n_clear_eval": n_eval, "mean_dev": mean_dev, "over_dev": over_dev, "dev_std": dev_std, "group_noise": g_noise,
                     "low_days": low_days, "streak": streak, "grade": grade, "cause": cause, "lost_kwh": round(lost, 1),
                     "energy_kwh": float(E.loc[sid].sum(skipna=True)) if sid in E.index else 0.0})
        if grade:
            first = next((d for d, x in dv.items() if x <= th.fault_floor), end.isoformat()) if stt == "connected" else (clear_days[0] if clear_days else end.isoformat())
            sig = Signature(ratio_mean=(1 + mean_dev) if pd.notna(mean_dev) else None, zero_all_day=stt.startswith("zero"), notes=notes)
            suspects.append({"suspect_id": f"{sid}@{first}", "string_id": sid, "inverter_id": inv_of[sid], "station_id": station_id,
                             "date": end.isoformat(), "first_date": first, "scope": "string", "grade": grade, "rule_cause": cause,
                             "dev_today": None if pd.isna(mean_dev) else float(mean_dev), "persist_days": int(max(streak, low_days)),
                             "signature_json": sig.model_dump_json(), "low_confidence": 1, "archived_shading": 0,
                             "status": SuspectStatus.OPEN.value, "created_at": pd.Timestamp.now().isoformat()})
    if write and suspects:
        store.upsert_df("suspects", pd.DataFrame(suspects))
    stats = pd.DataFrame(rows)
    return {"suspects": suspects, "strings": stats, "channel_states": channel_states, "clear_days": clear_days, "group_mode": group_mode_used,
            "state_counts": pd.Series(list(channel_states.values())).value_counts().to_dict()}
