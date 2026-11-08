"""阳光电源 iSolarCloud OpenAPI 适配器。

要点（需求表）：token 48 小时滚动失效（失效需业主重新授权）；不返回当天数据 → 每日任务拉 T-1。
接口链：/openapi/login → token；/openapi/getDeviceList；/openapi/getDevicePointMinuteDataList（5 分钟测点）。
组串电流/电压测点编号因机型而异，通过 config.points_map 配置，例如：
    points_map:
      pac: "p24"            # 有功功率 (kW)
      strings:
        1: {"i": "p5", "u": "p6"}
        2: {"i": "p7", "u": "p8"}
现网联调时以 F:\\Hermes 阳光工具包核对。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
import requests

from .base import FetchResult, PlatformAdapter, STRING_COLS, INVERTER_COLS


class SungrowAdapter(PlatformAdapter):
    platform = "sungrow"

    def __init__(self, station_id: str, config: Dict[str, Any], state_get=None, state_set=None):
        super().__init__(station_id, config, state_get, state_set)
        self.base_url = self.config.get("base_url", "https://gateway.isolarcloud.com")
        self.appkey = self.config.get("appkey")
        self.access_key = self.config.get("access_key")
        self.account = self.config.get("user_account")
        self.password = self.config.get("user_password")
        self.ps_id = self.config.get("ps_id", station_id)
        self.points_map: Dict[str, Any] = self.config.get("points_map", {})
        self.token_ttl = timedelta(hours=int(self.config.get("token_ttl_hours", 48)))
        self.timeout = int(self.config.get("timeout", 60))

    def _headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json", "x-access-key": self.access_key or "", "sys_code": "901"}

    def _token(self) -> str:
        st = self._state_get(f"sungrow:{self.account}:token") or {}
        if st and datetime.fromisoformat(st["expires"]) > datetime.now():
            return st["token"]
        r = requests.post(f"{self.base_url}/openapi/login", headers=self._headers(),
                          json={"appkey": self.appkey, "user_account": self.account,
                                "user_password": self.password, "login_type": "1"}, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if str(body.get("result_code")) != "1":
            raise RuntimeError(f"阳光登录失败: {body.get('result_code')} {body.get('result_msg')} —— token 失效需业主重新授权")
        tok = body["result_data"]["token"]
        self._state_set(f"sungrow:{self.account}:token",
                        {"token": tok, "expires": (datetime.now() + self.token_ttl - timedelta(hours=1)).isoformat()})
        return tok

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = {**payload, "appkey": self.appkey, "token": self._token()}
        r = requests.post(f"{self.base_url}{path}", headers=self._headers(), json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if str(body.get("result_code")) != "1":
            if str(body.get("result_code")) in ("E00003", "401"):
                self._state_set(f"sungrow:{self.account}:token", None)
            raise RuntimeError(f"阳光接口失败 {path}: {body.get('result_code')} {body.get('result_msg')}")
        return body

    def list_devices(self) -> pd.DataFrame:
        body = self._post("/openapi/getDeviceList", {"ps_id": self.ps_id, "curPage": 1, "size": 200})
        rows = []
        for d in (body.get("result_data", {}) or {}).get("pageList", []):
            if int(d.get("device_type", 0)) != 1:        # 1 = 逆变器
                continue
            rows.append({"inverter_id": str(d.get("ps_key")), "name": d.get("device_name", ""),
                         "model": d.get("device_model_code", ""), "rated_ac_kw": None, "n_strings": None})
        return pd.DataFrame(rows)

    def fetch(self, start: date, end: date) -> FetchResult:
        res = FetchResult()
        if end >= date.today():                     # 阳光不返回当天数据
            end = date.today() - timedelta(days=1)
            res.notes.append("阳光平台不返回当天数据，截止日期回退到 T-1")
        if start > end:
            return res
        devs = self.list_devices()
        if devs.empty or not self.points_map.get("strings"):
            res.notes.append("缺少设备或 points_map 配置，无法拉取组串测点")
            return res
        pac_pt = self.points_map.get("pac", "p24")
        smap: Dict[int, Dict[str, str]] = {int(k): v for k, v in self.points_map["strings"].items()}
        points = [pac_pt] + [p for m in smap.values() for p in (m["i"], m["u"])]
        s_rows: List[Dict[str, Any]] = []
        i_rows: List[Dict[str, Any]] = []
        for (b0, b1) in self.date_batches(start, end, 1):
            body = self._post("/openapi/getDevicePointMinuteDataList", {
                "ps_key_list": devs["inverter_id"].tolist(), "points": ",".join(points),
                "start_time_stamp": b0.strftime("%Y%m%d000000"), "end_time_stamp": b1.strftime("%Y%m%d235959"),
                "minute_interval": 5})
            res.calls_used += 1
            for rec in body.get("result_data", []) or []:
                dev = str(rec.get("ps_key"))
                ts = pd.to_datetime(str(rec.get("time_stamp")), format="%Y%m%d%H%M%S", errors="coerce")
                i_rows.append({"inverter_id": dev, "ts": ts, "pac_kw": rec.get(pac_pt), "status": None})
                for n, m in smap.items():
                    if rec.get(m["i"]) is not None:
                        s_rows.append({"string_id": f"{dev}-pv{n}", "ts": ts,
                                       "current_a": rec.get(m["i"]), "voltage_v": rec.get(m["u"])})
        res.strings = self.normalize_strings(pd.DataFrame(s_rows, columns=STRING_COLS))
        res.inverters = self.normalize_inverters(pd.DataFrame(i_rows, columns=INVERTER_COLS))
        return res
