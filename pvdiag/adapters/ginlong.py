"""锦浪 SolisCloud 开放平台适配器。

要点（需求表）：pac 单位是 W → ÷1000 转 kW；按天查询 inverterDay 返回 5 分钟点；
字段 iPv1~iPv32 / uPv1~uPv32 为组串电流/电压。
鉴权：HMAC-SHA1 签名（Content-MD5 + Date + Authorization: API keyId:sign）。
现网联调时以 F:\\Hermes 锦浪工具包核对 base_url 与字段名。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
import requests

from .base import FetchResult, PlatformAdapter, STRING_COLS, INVERTER_COLS


class GinlongAdapter(PlatformAdapter):
    platform = "ginlong"

    def __init__(self, station_id: str, config: Dict[str, Any], state_get=None, state_set=None):
        super().__init__(station_id, config, state_get, state_set)
        self.base_url = self.config.get("base_url", "https://www.soliscloud.com:13333")
        self.key_id = self.config.get("key_id")
        self.key_secret = (self.config.get("key_secret") or "").encode()
        self.plant_id = self.config.get("plant_id", station_id)
        self.pac_divisor = float(self.config.get("pac_divisor", 1000.0))
        self.timeout = int(self.config.get("timeout", 60))
        self.max_strings = int(self.config.get("max_strings", 32))

    # ------------------------------------------------------------------ 签名
    def _headers(self, path: str, body: str) -> Dict[str, str]:
        md5 = base64.b64encode(hashlib.md5(body.encode()).digest()).decode()
        gmt = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        ctype = "application/json"
        to_sign = f"POST\n{md5}\n{ctype}\n{gmt}\n{path}"
        sign = base64.b64encode(hmac.new(self.key_secret, to_sign.encode(), hashlib.sha1).digest()).decode()
        return {"Content-MD5": md5, "Content-Type": ctype, "Date": gmt, "Authorization": f"API {self.key_id}:{sign}"}

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload, separators=(",", ":"))
        r = requests.post(f"{self.base_url}{path}", data=body, headers=self._headers(path, body), timeout=self.timeout)
        r.raise_for_status()
        res = r.json()
        if not res.get("success", True) and res.get("code") not in ("0", 0, None):
            raise RuntimeError(f"锦浪接口失败 {path}: {res.get('code')} {res.get('msg')}")
        return res

    # ------------------------------------------------------------------
    def list_devices(self) -> pd.DataFrame:
        res = self._post("/v1/api/inverterList", {"pageNo": 1, "pageSize": 100, "stationId": self.plant_id})
        rows = []
        for d in (res.get("data", {}) or {}).get("page", {}).get("records", []):
            rows.append({"inverter_id": str(d.get("sn")), "name": d.get("name", d.get("sn")),
                         "model": d.get("productModel", ""), "rated_ac_kw": d.get("power"), "n_strings": None})
        return pd.DataFrame(rows)

    def fetch(self, start: date, end: date) -> FetchResult:
        res = FetchResult()
        devs = self.list_devices()
        s_rows: List[Dict[str, Any]] = []
        i_rows: List[Dict[str, Any]] = []
        d = start
        while d <= end:
            for sn in devs["inverter_id"].tolist():
                body = self._post("/v1/api/inverterDay",
                                  {"sn": sn, "money": "CNY", "time": d.isoformat(), "timeZone": 8})
                res.calls_used += 1
                for rec in body.get("data", []) or []:
                    ts = pd.to_datetime(int(rec.get("dataTimestamp")), unit="ms") + timedelta(hours=8)
                    pac = rec.get("pac")
                    i_rows.append({"inverter_id": sn, "ts": ts,
                                   "pac_kw": (float(pac) / self.pac_divisor) if pac is not None else None,
                                   "status": rec.get("state")})
                    for n in range(1, self.max_strings + 1):
                        ci, cu = f"iPv{n}", f"uPv{n}"
                        if ci in rec and rec[ci] is not None:
                            s_rows.append({"string_id": f"{sn}-pv{n}", "ts": ts,
                                           "current_a": rec[ci], "voltage_v": rec.get(cu)})
            d += timedelta(days=1)
        res.strings = self.normalize_strings(pd.DataFrame(s_rows, columns=STRING_COLS))
        res.inverters = self.normalize_inverters(pd.DataFrame(i_rows, columns=INVERTER_COLS))
        return res
