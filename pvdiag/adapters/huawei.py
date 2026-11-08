"""华为 FusionSolar 北向接口适配器（组串级 5 分钟数据）。

已有资产：F:\\Hermes\\华为北向API（pull_range.py 断点续拉、build_report_v2.py 两 sheet 报告）。
本适配器复刻其核心调用链，供流水线统一调度；现网联调时以旧工具包为准核对字段名。

接口链：
  POST /thirdData/login              → 响应头 xsrf-token（30 分钟时效，登录本身计入配额）
  POST /thirdData/getDevList         → 设备清单（devTypeId=1 组串式逆变器）
  POST /thirdData/getDevHistoryKpi   → 5 分钟历史，devIds ≤100 台/批，时间跨度 ≤ batch_days
       dataItemMap: pv1_u, pv1_i … pv36_u, pv36_i, active_power(kW), inverter_state

配额口径（需求表）：日限额 = Σ⌈设备数/100⌉ + 24，登录也计入；回补历史会吃光当日配额 → 跨天分批。
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .base import FetchResult, PlatformAdapter, STRING_COLS, INVERTER_COLS


class QuotaPlanner:
    """配额先算后拉：给定设备数与待回补天数，规划每日可拉的批次。"""

    def __init__(self, n_devices: int, base_quota: int = 24, devices_per_unit: int = 100,
                 batch_days: int = 4, login_cost: int = 1):
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


class HuaweiAdapter(PlatformAdapter):
    platform = "huawei"

    def __init__(self, station_id: str, config: Dict[str, Any], state_get=None, state_set=None):
        super().__init__(station_id, config, state_get, state_set)
        self.base_url = self.config.get("base_url", "https://intl.fusionsolar.huawei.com")
        self.user = self.config.get("user_name")
        self.system_code = self.config.get("system_code")       # 建议从 DPAPI/EFS/钥匙串解出后注入，不落明文
        self.station_code = self.config.get("station_code", station_id)
        self.batch_days = int(self.config.get("batch_days", 4))
        self.token_ttl = timedelta(minutes=int(self.config.get("token_ttl_minutes", 30)))
        self.timeout = int(self.config.get("timeout", 60))
        self.session = requests.Session()

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
            return st["token"]
        r = self.session.post(f"{self.base_url}/thirdData/login",
                              json={"userName": self.user, "systemCode": self.system_code}, timeout=self.timeout)
        self._count_call(1)
        r.raise_for_status()
        body = r.json()
        if not body.get("success"):
            raise RuntimeError(f"华为登录失败: {body.get('failCode')} {body.get('message')} —— 若为 20400/305 需业主重新授权")
        tok = r.headers.get("xsrf-token") or r.headers.get("XSRF-TOKEN")
        self.session.headers["XSRF-TOKEN"] = tok
        self._state_set(f"huawei:{self.user}:token",
                        {"token": tok, "expires": (datetime.now() + self.token_ttl - timedelta(minutes=2)).isoformat()})
        return tok

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._token()
        r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
        self._count_call(1)
        r.raise_for_status()
        body = r.json()
        if not body.get("success"):
            code = body.get("failCode")
            if code in (305, 401):           # token 失效 → 清状态重登一次
                self._state_set(f"huawei:{self.user}:token", None)
                self._token()
                r = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout)
                self._count_call(1)
                body = r.json()
                if body.get("success"):
                    return body
            if code == 407:
                raise RuntimeError("华为接口 407：超出调用频次/配额，今日停止拉取，明日续拉")
            raise RuntimeError(f"华为接口失败 {path}: {code} {body.get('message')}")
        return body

    # ------------------------------------------------------------------
    def list_devices(self) -> pd.DataFrame:
        body = self._post("/thirdData/getDevList", {"stationCodes": self.station_code})
        rows = []
        for d in body.get("data", []):
            if int(d.get("devTypeId", 0)) != 1:       # 只取组串式逆变器
                continue
            rows.append({"inverter_id": str(d.get("id")), "name": d.get("devName", ""),
                         "model": d.get("model", ""), "rated_ac_kw": None, "n_strings": None,
                         "esn": d.get("esnCode", "")})
        return pd.DataFrame(rows)

    def fetch(self, start: date, end: date) -> FetchResult:
        res = FetchResult()
        devs = self.list_devices()
        if devs.empty:
            res.notes.append("无组串式逆变器设备")
            return res
        dev_ids = devs["inverter_id"].tolist()
        planner = QuotaPlanner(len(dev_ids), batch_days=self.batch_days)
        s_rows: List[pd.DataFrame] = []
        i_rows: List[pd.DataFrame] = []
        for (b0, b1) in self.date_batches(start, end, self.batch_days):
            if planner.batches_per_day(self.calls_used_today(), reserve_for_daily=0) <= 0:
                res.notes.append(f"配额不足，停止于 {b0}（明日续拉）")
                break
            for k in range(0, len(dev_ids), 100):
                chunk = dev_ids[k:k + 100]
                payload = {"devIds": ",".join(chunk), "devTypeId": 1,
                           "startTime": int(datetime.combine(b0, datetime.min.time()).timestamp() * 1000),
                           "endTime": int(datetime.combine(b1 + timedelta(days=1), datetime.min.time()).timestamp() * 1000) - 1}
                body = self._post("/thirdData/getDevHistoryKpi", payload)
                res.calls_used += 1
                s, i = self._parse_history(body.get("data", []))
                s_rows.append(s)
                i_rows.append(i)
        if s_rows:
            res.strings = self.normalize_strings(pd.concat(s_rows, ignore_index=True))
        if i_rows:
            res.inverters = self.normalize_inverters(pd.concat(i_rows, ignore_index=True))
        return res

    @staticmethod
    def _parse_history(data: List[Dict[str, Any]]):
        s_rows, i_rows = [], []
        for rec in data:
            dev = str(rec.get("devId"))
            ts = pd.to_datetime(rec.get("collectTime"), unit="ms")
            m = rec.get("dataItemMap", {}) or {}
            i_rows.append({"inverter_id": dev, "ts": ts, "pac_kw": m.get("active_power"),
                           "status": m.get("inverter_state")})
            for n in range(1, 37):
                ci, cu = f"pv{n}_i", f"pv{n}_u"
                if ci in m and m[ci] is not None:
                    s_rows.append({"string_id": f"{dev}-pv{n}", "ts": ts,
                                   "current_a": m.get(ci), "voltage_v": m.get(cu)})
        return (pd.DataFrame(s_rows, columns=STRING_COLS), pd.DataFrame(i_rows, columns=INVERTER_COLS))
