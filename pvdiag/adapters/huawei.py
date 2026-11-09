"""华为 FusionSolar 北向接口适配器（组串级 5 分钟数据）。

按 2026-10 现网实测（SmartPVMS 25.x，中国区，20 台 SUN2000-100/110KTL）校准：
  POST /thirdData/login                {userName, systemCode} → 响应头 Set-Cookie XSRF-TOKEN（也有 xsrf-token 头），30 分钟有效，登录计入配额
  POST /thirdData/getStationList       {pageNo, pageSize(必填)}
  POST /thirdData/getDevList           {stationCodes:"NE=xxx,NE=yyy"} → [{id, devName, devTypeId, esnCode, ...}]；devTypeId 1 组串逆变器、10 环境监测仪
  POST /thirdData/getDevFiveMinutes    {devIds(≤100 台), devTypeId:1, collectTime(该天任意时刻 ms)} → 该天全天，一行 = 一台 × 一个 5 分钟点
       dataItemMap: pv1_i/pv1_u … pv36_i/pv36_u（机型没有的路为 null，有但没接的路为 0.0）、active_power(kW)、inverter_state、
                    mppt_1_cap … mppt_10_cap（各 MPPT 直流容量 W，可反推每 MPPT 的组件数/并联数）、mppt_total_cap、temperature
  POST /thirdData/getDevHistoryKpi     部分账号 401 无权限 → 默认走 getDevFiveMinutes，可用 history_api: historykpi 切回
  域名：cn.fusionsolar.huawei.com（中国区）/ intl.fusionsolar.huawei.com 可用；uni01cn 会 302 到 SSO 不能用
  限流：同一接口有每分钟频次限制（实测两次数据调用间隔 ≥65 s 安全）；每日总配额 = Σ⌈每类设备数/100⌉ + 24（20 台 = 25 次/日），超限 failCode 407；
        305 = token 过期（重登一次）
  时间：collectTime 为 UTC 毫秒 → 换算电站本地时（默认 +8h）

命名：inverter_id = "<station_id>-<devName>"（如 XWZ3-NB01），string_id = "<inverter_id>-pv<N>"，与日电量模式/档案一致；
      devId 只在调用时用（state 里保存 name→devId 映射）。id_style: devid 可改回用 devId。
凭证：system_code 建议不落明文——支持 system_code_env（环境变量名）。
"""
from __future__ import annotations

import math
import os
import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from .base import FetchResult, PlatformAdapter, STRING_COLS, INVERTER_COLS


class QuotaPlanner:
    """配额先算后拉：给定设备数与待回补天数，规划每日可拉的批次。"""

    def __init__(self, n_devices: int, base_quota: int = 24, devices_per_unit: int = 100,
                 batch_days: int = 1, login_cost: int = 1):
        self.n_devices = n_devices
        self.units = max(1, math.ceil(n_devices / devices_per_unit))   # 每次历史查询需要的调用数
        self.daily_quota = self.units + base_quota
        self.batch_days = batch_days
        self.login_cost = login_cost

    def calls_per_batch(self) -> int:
        return self.units

    def batches_per_day(self, used_today: int = 0, reserve_for_daily: int = 1) -> int:
        """今天还能拉多少个 batch（预留每日增量 1 批 + 登录）"""
        remain = self.daily_quota - used_today - self.login_cost - reserve_for_daily * self.units
        return max(0, remain // self.units)

    def plan_backfill(self, start: date, end: date, used_today: int = 0) -> Dict[str, Any]:
        total_days = (end - start).days + 1
        n_batches = math.ceil(total_days / self.batch_days)
        per_day = self.batches_per_day(used_today)
        days_needed = math.ceil(n_batches / per_day) if per_day else math.inf
        return {"total_days": total_days, "n_batches": n_batches, "batches_per_day": per_day,
                "calendar_days_needed": days_needed, "daily_quota": self.daily_quota}


def _safe_name(s: str) -> str:
    s = re.sub(r"\s+", "", str(s or "")).strip()
    return re.sub(r"[^0-9A-Za-z_一-鿿-]", "", s) or "DEV"


class HuaweiAdapter(PlatformAdapter):
    platform = "huawei"

    def __init__(self, station_id: str, config: Dict[str, Any], state_get=None, state_set=None):
        super().__init__(station_id, config, state_get, state_set)
        self.base_url = (self.config.get("base_url") or self.config.get("base") or "https://cn.fusionsolar.huawei.com").rstrip("/")
        self.user = self.config.get("user_name") or self.config.get("userName")
        env = self.config.get("system_code_env")
        self.system_code = (os.environ.get(env) if env else None) or self.config.get("system_code") or self.config.get("systemCode")
        self.station_code = self.config.get("station_code") or self.config.get("stationCode") or station_id
        self.history_api = self.config.get("history_api", "fivemin")           # fivemin | historykpi
        self.batch_days = int(self.config.get("batch_days", 1 if self.history_api == "fivemin" else 4))
        self.token_ttl = timedelta(minutes=int(self.config.get("token_ttl_minutes", 30)))
        self.timeout = int(self.config.get("timeout", 60))
        self.tz_offset = timedelta(hours=float(self.config.get("tz_offset_hours", 8)))
        self.min_interval = float(self.config.get("min_interval_seconds", 65))     # 数据接口两次调用间隔
        self.id_style = self.config.get("id_style", "name")                        # name | devid
        self.dev_types = [int(x) for x in self.config.get("dev_types", [1])]        # 1 组串逆变器
        self.session = requests.Session()
        self._last_call_at: Optional[float] = None
        self._dev_map: Dict[str, str] = {}      # inverter_id → devId

    # ------------------------------------------------------------------ 配额与 token 状态
    def _quota_key(self) -> str:
        return f"huawei:{self.user}:calls:{date.today().isoformat()}"

    def calls_used_today(self) -> int:
        return int(self._state_get(self._quota_key(), 0) or 0)

    def _count_call(self, n: int = 1) -> None:
        self._state_set(self._quota_key(), self.calls_used_today() + n)

    def _token(self) -> str:
        st = self._state_get(f"huawei:{self.user}:token") or {}
        if st and datetime.fromisoformat(st["expires"]) > datetime.now():
            self.session.headers["XSRF-TOKEN"] = st["token"]
            self.session.cookies.set("XSRF-TOKEN", st["token"])
            return st["token"]
        if not self.user or not self.system_code:
            raise RuntimeError("华为北向凭证缺失：platform_config 需要 user_name / system_code（或 system_code_env）")
        r = self.session.post(f"{self.base_url}/thirdData/login",
                              json={"userName": self.user, "systemCode": self.system_code}, timeout=self.timeout)
        self._count_call(1)
        r.raise_for_status()
        body = r.json()
        if not body.get("success"):
            raise RuntimeError(f"华为登录失败: {body.get('failCode')} {body.get('message')} —— 20400/305 需重新登录或核对北向账号；407 为频次超限")
        tok = self.session.cookies.get("XSRF-TOKEN") or r.headers.get("xsrf-token") or r.headers.get("XSRF-TOKEN")
        if not tok:
            raise RuntimeError("华为登录成功但没有 XSRF-TOKEN（Cookie/响应头都没有）")
        self.session.headers["XSRF-TOKEN"] = tok
        self._state_set(f"huawei:{self.user}:token",
                        {"token": tok, "expires": (datetime.now() + self.token_ttl - timedelta(minutes=2)).isoformat()})
        return tok

    def _pace(self) -> None:
        """数据接口两次调用之间至少隔 min_interval 秒（同一接口每分钟频次限制）"""
        if self._last_call_at is not None and self.min_interval > 0:
            wait = self.min_interval - (time.monotonic() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
        self._last_call_at = time.monotonic()

    def _post(self, path: str, payload: Dict[str, Any], paced: bool = False) -> Dict[str, Any]:
        self._token()
        if paced:
            self._pace()
        r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        self._count_call(1)
        r.raise_for_status()
        body = r.json()
        if not body.get("success"):
            code = body.get("failCode")
            if code == 305:                  # token 过期 → 清状态重登一次
                self._state_set(f"huawei:{self.user}:token", None)
                self._token()
                r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
                self._count_call(1)
                body = r.json()
                if body.get("success"):
                    return body
                code = body.get("failCode")
            if code == 407:
                raise RuntimeError("华为接口 407：超出调用频次/每日配额，今日停止拉取，明日续拉")
            if code == 401:
                raise RuntimeError(f"华为接口 401：该北向账号无 {path} 权限（getDevHistoryKpi 常见；改 history_api: fivemin）")
            raise RuntimeError(f"华为接口失败 {path}: {code} {body.get('message')}")
        return body

    # ------------------------------------------------------------------
    def list_stations(self) -> pd.DataFrame:
        body = self._post("/thirdData/getStationList", {"pageNo": 1, "pageSize": 100})
        rows = [{"station_code": s.get("stationCode"), "name": s.get("stationName"), "capacity": s.get("capacity"), "addr": s.get("stationAddr")}
                for s in (body.get("data") or {}).get("list", [])]
        return pd.DataFrame(rows)

    def list_devices(self) -> pd.DataFrame:
        body = self._post("/thirdData/getDevList", {"stationCodes": self.station_code})
        rows = []
        self._dev_map = {}
        for d in body.get("data", []) or []:
            if int(d.get("devTypeId", 0)) not in self.dev_types:       # 只取组串式逆变器（可配 10 环境监测仪）
                continue
            dev_id = str(d.get("id"))
            name = _safe_name(d.get("devName") or dev_id)
            inv_id = f"{self.station_id}-{name}" if self.id_style == "name" else f"{self.station_id}-{dev_id}"
            self._dev_map[inv_id] = dev_id
            rows.append({"inverter_id": inv_id, "name": d.get("devName", ""), "model": d.get("model", "") or d.get("devTypeName", ""),
                         "rated_ac_kw": None, "n_strings": None, "esn": d.get("esnCode", ""), "dev_id": dev_id,
                         "dev_type": int(d.get("devTypeId", 0))})
        self._state_set(f"huawei:{self.station_id}:devmap", self._dev_map)
        return pd.DataFrame(rows)

    def _collect_time_ms(self, d: date) -> int:
        """该天电站本地时 12:00 对应的 UTC 毫秒（collectTime 按设备时区归属日期）"""
        local_noon = datetime.combine(d, datetime.min.time()) + timedelta(hours=12)
        return int(((local_noon - self.tz_offset) - datetime(1970, 1, 1)).total_seconds() * 1000)

    def fetch(self, start: date, end: date) -> FetchResult:
        res = FetchResult()
        devs = self.list_devices()
        inv = devs[devs["dev_type"] == 1] if "dev_type" in devs.columns else devs
        if inv.empty:
            res.notes.append("无组串式逆变器设备")
            return res
        id_by_dev = {r.dev_id: r.inverter_id for r in inv.itertuples()}
        dev_ids = list(id_by_dev.keys())
        planner = QuotaPlanner(len(dev_ids), batch_days=self.batch_days)
        s_rows: List[pd.DataFrame] = []
        i_rows: List[pd.DataFrame] = []
        hints: List[pd.DataFrame] = []
        for (b0, b1) in self.date_batches(start, end, self.batch_days):
            if planner.batches_per_day(self.calls_used_today(), reserve_for_daily=0) <= 0:
                res.notes.append(f"配额不足，停止于 {b0}（明日续拉）")
                break
            for k in range(0, len(dev_ids), 100):
                chunk = dev_ids[k:k + 100]
                if self.history_api == "historykpi":
                    payload = {"devIds": ",".join(chunk), "devTypeId": 1,
                               "startTime": self._collect_time_ms(b0) - 12 * 3600 * 1000,
                               "endTime": self._collect_time_ms(b1) + 12 * 3600 * 1000 - 1}
                    body = self._post("/thirdData/getDevHistoryKpi", payload, paced=True)
                else:
                    body = self._post("/thirdData/getDevFiveMinutes",
                                      {"devIds": ",".join(chunk), "devTypeId": 1, "collectTime": self._collect_time_ms(b0)}, paced=True)
                res.calls_used += 1
                s, i, h = self._parse_history(body.get("data", []) or [], id_by_dev, self.tz_offset)
                s_rows.append(s); i_rows.append(i); hints.append(h)
        if s_rows:
            res.strings = self.normalize_strings(pd.concat(s_rows, ignore_index=True))
        if i_rows:
            res.inverters = self.normalize_inverters(pd.concat(i_rows, ignore_index=True))
        if hints:
            h = pd.concat(hints, ignore_index=True)
            if not h.empty:
                res.notes.append(f"MPPT 容量提示 {len(h)} 条（pvdiag platform hints 可生成档案建议）")
                self._state_set(f"huawei:{self.station_id}:mppt_caps", h.drop_duplicates(["inverter_id", "mppt"], keep="last").to_dict("records"))
        return res

    @staticmethod
    def _parse_history(data: List[Dict[str, Any]], id_by_dev: Dict[str, str], tz_offset: timedelta = timedelta(hours=8)
                       ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """扁平记录 → (组串明细, 逆变器明细, MPPT 容量提示)。pvN 为 null 的路 = 机型没有这路（不输出）；0.0 = 有这路但没电流（保留）"""
        s_rows, i_rows, caps = [], [], {}
        for rec in data:
            dev = str(rec.get("devId"))
            inv = id_by_dev.get(dev, dev)
            ts = pd.to_datetime(rec.get("collectTime"), unit="ms") + tz_offset
            m = rec.get("dataItemMap", {}) or {}
            i_rows.append({"inverter_id": inv, "ts": ts, "pac_kw": m.get("active_power"), "status": m.get("inverter_state")})
            for n in range(1, 37):
                ci, cu = f"pv{n}_i", f"pv{n}_u"
                if m.get(ci) is not None:
                    s_rows.append({"string_id": f"{inv}-pv{n}", "ts": ts, "current_a": m.get(ci), "voltage_v": m.get(cu)})
            if inv not in caps:
                c = {k: v for k, v in m.items() if k.startswith("mppt_") and k.endswith("_cap") and k != "mppt_total_cap" and v is not None}
                if c:
                    caps[inv] = c
        h_rows = [{"inverter_id": inv, "mppt": int(k.split("_")[1]), "cap_w": float(v)} for inv, c in caps.items() for k, v in c.items()]
        return (pd.DataFrame(s_rows, columns=STRING_COLS), pd.DataFrame(i_rows, columns=INVERTER_COLS),
                pd.DataFrame(h_rows, columns=["inverter_id", "mppt", "cap_w"]))

    # ------------------------------------------------------------------ 档案提示
    @staticmethod
    def profile_hints(strings_5min: pd.DataFrame, mppt_caps: pd.DataFrame, module_wp: Optional[float] = None,
                      inputs_per_mppt: int = 2, typical_n_modules: Optional[int] = None) -> pd.DataFrame:
        """用 MPPT 直流容量 + 哪些路有电流，反推每路的组件数/并联数建议：
        - 一个 MPPT 的容量 C 由该 MPPT 上有电流的路分担；只有一路有电流 → 这一路 n_modules ≈ C/wp（若 C/wp 明显大于典型串长 → 并联 n_parallel=round(C/wp/典型串长)）
        - 有电压无电流且兄弟路有电流 → 空输入（n_modules 空，notes=empty_input）
        - 电流电压都 ~0 或该路 null → 未接
        返回列：string_id, inverter_id, pv_index, mppt, cap_w, state, n_modules, n_parallel, note"""
        if strings_5min.empty:
            return pd.DataFrame()
        s = strings_5min.copy()
        s["pv_index"] = s["string_id"].str.extract(r"pv(\d+)$").astype(float)
        s["inverter_id"] = s["string_id"].str.replace(r"-pv\d+$", "", regex=True)
        agg = s.groupby(["inverter_id", "string_id", "pv_index"]).agg(i_max=("current_a", "max"), v_med=("voltage_v", "median")).reset_index()
        agg["mppt"] = ((agg["pv_index"] - 1) // inputs_per_mppt + 1).astype(int)
        cap = mppt_caps.set_index(["inverter_id", "mppt"])["cap_w"].to_dict() if not mppt_caps.empty else {}
        out = []
        for (inv, mp), g in agg.groupby(["inverter_id", "mppt"]):
            C = cap.get((inv, mp))
            producing = g[g["i_max"].fillna(0) >= 0.5]
            n_prod = len(producing)
            for r in g.itertuples():
                state, nm, npar, note = "connected", None, 1, ""
                if r.i_max is None or pd.isna(r.i_max) or r.i_max < 0.5:
                    if n_prod > 0 and (pd.notna(r.v_med) and r.v_med > 100):
                        state, note = "empty_input", "同 MPPT 另一路有电流、本路电压随之但电流 0 → 空输入（档案 n_modules 留空）"
                    elif C and C > 0 and n_prod == 0:
                        state, note = "zero_current", f"MPPT 有容量 {C:.0f}W 但两路都无电流 → 待核实（断线/未启用）"
                    else:
                        state, note = "unconnected", "无容量/无电压 → 未接"
                elif C and module_wp:
                    share = C / n_prod
                    k = share / module_wp
                    if typical_n_modules and k > 1.5 * typical_n_modules:
                        npar = int(round(k / typical_n_modules)); nm = int(round(k / npar))
                        note = f"MPPT 容量 {C:.0f}W/{n_prod} 路 ≈ {k:.1f} 块 → 疑 {npar} 串并联 × {nm} 块"
                    else:
                        nm = int(round(k)); note = f"MPPT 容量 {C:.0f}W/{n_prod} 路 ≈ {k:.1f} 块"
                out.append({"string_id": r.string_id, "inverter_id": inv, "pv_index": int(r.pv_index), "mppt": mp, "cap_w": C,
                            "state": state, "n_modules": nm, "n_parallel": npar, "i_max": r.i_max, "v_med": r.v_med, "note": note})
        return pd.DataFrame(out).sort_values(["inverter_id", "pv_index"]).reset_index(drop=True)
