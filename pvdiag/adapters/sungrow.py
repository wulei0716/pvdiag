"""阳光电源 iSolarCloud 北向 OpenAPI 适配器（按 2026-10 现网实测校准：熙盛分布式项目 · 免费套餐 · SG110CX）。

调用约定：
  全部 POST，Content-Type: application/json;charset=UTF-8；请求头 x-access-key: <Secret Key>；Authorization: Bearer <access_token>；
  请求体必带 appkey（缺了 er_invalid_appkey）；成功 result_code == "1"，数据在 result_data
  鉴权是 OAuth 授权码：业主在阳光云授权页点"同意授权" → 回调 ?code=XXXX（一次性）
     POST /openapi/apiManage/token         {appkey, grant_type:"authorization_code", code, redirect_uri, token_scope_mode:"global"}
     POST /openapi/apiManage/refreshToken  {appkey, refresh_token}   access_token 48h，滚动续期；长期不用失效需业主再授权
  电站列表 POST /openapi/platform/queryPowerStationList   {page, size, valid_flag:"1,3"} → result_data.pageList[{ps_id, ps_name, online_status}]
  设备列表 POST /openapi/platform/getDeviceListByPsId     {ps_id, is_virtual_unit:"0", device_type_list:[1], page, size} → pageList[{ps_key, device_name, device_sn, device_type, device_model_code}]
  分钟数据 POST /openapi/platform/getDevicePointMinuteDataList
           {ps_key_list(≤50), points:"p5,p6,…"(<50 个), start_time_stamp/end_time_stamp yyyyMMddHHmmss(电站时区, 窗口 ≤3h), minute_interval:"5", is_get_point_dict:"1"}
           → result_data = {ps_key: [{time_stamp, p5, p6, …}], point_dict:[…]}   ※ 不返回当天数据 → 每日任务拉 T-1
  组串级测点（SG110CX，9 路 MPPT，电压奇数/电流偶数）：MPPT1-9 电压 p5,p7,p9,p45,p47,p49,p51,p53,p55；电流 p6,p8,p10,p46,…,p56；逆变器总有功功率 p24（W → ÷1000 kW）
  通道口径：阳光给的是 MPPT 级（一个 MPPT 两路输入并在一起）→ string_id = <inv>-pv<MPPT 号>，档案 mppt=pv_index，channel_is_mppt
           晴天中午电压 <400V = 空通道（未接，非故障）；电压正常 + 电流恒 <0.1A = 断线

令牌状态存 platform_state（sungrow:<appkey>:token = {access_token, refresh_token, expire_at}）；首次用 `pvdiag platform auth sungrow --station X --code XXXX` 换取。
命名：inverter_id = "<station_id>-INV<nn>"（device_name "逆变器2" → INV02；其它名字原样清洗），与日电量模式一致。
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .base import FetchResult, PlatformAdapter, STRING_COLS, INVERTER_COLS

# SG110CX（-P2）默认测点：MPPT1..9 电压 / 电流；其它机型用 points_map 覆盖
SG110CX_POINTS = {"pac": "p24",
                  "strings": {m + 1: {"u": u, "i": i} for m, (u, i) in enumerate(zip(
                      ["p5", "p7", "p9", "p45", "p47", "p49", "p51", "p53", "p55"],
                      ["p6", "p8", "p10", "p46", "p48", "p50", "p52", "p54", "p56"]))}}


def _norm_inv_name(name: str) -> str:
    s = str(name or "").strip()
    m = re.fullmatch(r"逆变器\s*(\d+)", s)
    if m:
        return f"INV{int(m.group(1)):02d}"
    s = re.sub(r"\s+", "", s)
    return re.sub(r"[^0-9A-Za-z_一-鿿-]", "", s) or "INV"


class SungrowAdapter(PlatformAdapter):
    platform = "sungrow"

    def __init__(self, station_id: str, config: Dict[str, Any], state_get=None, state_set=None):
        super().__init__(station_id, config, state_get, state_set)
        self.base_url = (self.config.get("base_url") or self.config.get("gateway") or "https://gateway.isolarcloud.com").rstrip("/")
        self.appkey = self.config.get("appkey")
        self.secret = self.config.get("secret_key") or self.config.get("access_key")
        self.ps_id = str(self.config.get("ps_id", station_id))
        self.redirect_uri = self.config.get("redirect_uri", "https://www.baidu.com/")
        pm = self.config.get("points_map") or {}
        self.custom_strings = bool(pm.get("strings"))
        self.points_map: Dict[str, Any] = {"pac": pm.get("pac", SG110CX_POINTS["pac"]),
                                           "strings": {int(k): v for k, v in (pm.get("strings") or SG110CX_POINTS["strings"]).items()}}
        self.window_hours = int(self.config.get("window_hours", 3))
        self.min_interval = float(self.config.get("min_interval_seconds", 0.5))
        self.timeout = int(self.config.get("timeout", 60))
        self.same_day = bool(self.config.get("same_day_available", False))
        self.pac_divisor = float(self.config.get("pac_divisor", 1000.0))      # p24 逆变器总有功功率单位 W → kW
        self._last_call_at: Optional[float] = None

    # ------------------------------------------------------------------ 令牌
    def _tok_key(self) -> str:
        return f"sungrow:{self.appkey}:token"

    def _headers(self, auth: bool = True) -> Dict[str, str]:
        h = {"Content-Type": "application/json;charset=UTF-8", "x-access-key": self.secret or ""}
        if auth:
            h["Authorization"] = f"Bearer {self._access_token()}"
        return h

    def _save_token(self, d: Dict[str, Any]) -> Dict[str, Any]:
        t = {"access_token": d.get("access_token"), "refresh_token": d.get("refresh_token"),
             "expire_at": time.time() + int(d.get("expires_in", 172799)) - 600, "auth_ps_list": d.get("auth_ps_list"),
             "saved_at": datetime.now().isoformat(timespec="seconds")}
        self._state_set(self._tok_key(), t)
        return t

    def token_by_code(self, code: str) -> Dict[str, Any]:
        """业主授权回调里的 code（一次性）换令牌"""
        r = requests.post(f"{self.base_url}/openapi/apiManage/token", headers=self._headers(auth=False),
                          json={"appkey": self.appkey, "grant_type": "authorization_code", "code": code,
                                "redirect_uri": self.redirect_uri, "token_scope_mode": "global"}, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        d = body.get("result_data") or {}
        if not d.get("access_token"):
            raise RuntimeError(f"阳光换取令牌失败: {body.get('result_code')} {body.get('result_msg')}")
        return self._save_token(d)

    def _refresh(self) -> str:
        tok = self._state_get(self._tok_key()) or {}
        if not tok.get("refresh_token"):
            raise RuntimeError("阳光云没有可用 refresh_token：需业主在阳光云授权页点“同意授权”，把回调 ?code=XXXX 交给 "
                               "`pvdiag platform auth sungrow --station <站> --code XXXX`")
        r = requests.post(f"{self.base_url}/openapi/apiManage/refreshToken", headers=self._headers(auth=False),
                          json={"appkey": self.appkey, "refresh_token": tok["refresh_token"]}, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        d = body.get("result_data") or {}
        if not d.get("access_token"):
            self._state_set(self._tok_key(), {**tok, "refresh_failed_at": datetime.now().isoformat(timespec="seconds")})
            raise RuntimeError(f"阳光刷新令牌失败（refresh_token 失效，需业主重新授权）: {body.get('result_code')} {body.get('result_msg')}")
        return self._save_token(d)["access_token"]

    def _access_token(self) -> str:
        tok = self._state_get(self._tok_key()) or {}
        if tok.get("access_token") and float(tok.get("expire_at", 0)) > time.time() + 300:
            return tok["access_token"]
        return self._refresh()

    def token_status(self) -> Dict[str, Any]:
        tok = self._state_get(self._tok_key()) or {}
        return {"has_token": bool(tok.get("access_token")), "has_refresh": bool(tok.get("refresh_token")),
                "expires_in_h": round((float(tok.get("expire_at", 0)) - time.time()) / 3600, 1) if tok else None,
                "auth_ps_list": tok.get("auth_ps_list"), "saved_at": tok.get("saved_at")}

    # ------------------------------------------------------------------ 调用
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._last_call_at is not None and self.min_interval > 0:
            wait = self.min_interval - (time.monotonic() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
        self._last_call_at = time.monotonic()
        body_json = {"appkey": self.appkey, "lang": "_zh_CN", **payload}
        r = requests.post(f"{self.base_url}{path}", headers=self._headers(), json=body_json, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        if str(body.get("result_code")) != "1":
            code = str(body.get("result_code"))
            if "token" in code.lower() or code in ("E00003", "401", "invalid_token"):
                tok = self._state_get(self._tok_key()) or {}
                self._state_set(self._tok_key(), {**tok, "expire_at": 0})       # 强制下次刷新
            raise RuntimeError(f"阳光接口失败 {path}: {code} {body.get('result_msg')}")
        return body.get("result_data") or {}

    def list_stations(self) -> pd.DataFrame:
        d = self._post("/openapi/platform/queryPowerStationList", {"page": 1, "size": 100, "valid_flag": "1,3"})
        return pd.DataFrame([{"ps_id": str(p.get("ps_id")), "name": p.get("ps_name"), "online": p.get("online_status")} for p in d.get("pageList") or []])

    def list_devices(self) -> pd.DataFrame:
        d = self._post("/openapi/platform/getDeviceListByPsId",
                       {"ps_id": self.ps_id, "is_virtual_unit": "0", "device_type_list": [1], "page": 1, "size": 200})
        rows = []
        for x in d.get("pageList") or []:
            if int(x.get("device_type", 1)) != 1:
                continue
            model = x.get("device_model_code") or ""
            n_strings = 4 if model == "SG40CX-P3-CN" and not self.custom_strings else len(self.points_map["strings"])
            rows.append({"inverter_id": f"{self.station_id}-{_norm_inv_name(x.get('device_name'))}", "name": x.get("device_name", ""),
                         "model": x.get("device_model_code", ""), "rated_ac_kw": None, "n_strings": n_strings,
                         "ps_key": str(x.get("ps_key")), "sn": x.get("device_sn", "")})
        return pd.DataFrame(rows)

    def fetch(self, start: date, end: date) -> FetchResult:
        res = FetchResult()
        if not self.same_day and end >= date.today():                 # 阳光不返回当天数据
            end = date.today() - timedelta(days=1)
            res.notes.append("阳光平台不返回当天数据，截止日期回退到 T-1")
        if start > end:
            return res
        devs = self.list_devices()
        if devs.empty:
            res.notes.append("电站下没有逆变器设备")
            return res
        pac_pt = self.points_map["pac"]
        smap: Dict[int, Dict[str, str]] = self.points_map["strings"]
        device_groups = []
        is_four_input = devs["model"].eq("SG40CX-P3-CN") & (not self.custom_strings)
        for four_input, devices in devs.groupby(is_four_input):
            channel_map = {n: v for n, v in smap.items() if not four_input or n <= 4}
            points = [pac_pt] + [point for pair in channel_map.values() for point in (pair["u"], pair["i"])]
            device_groups.append((dict(zip(devices["ps_key"], devices["inverter_id"])), points, channel_map))
        s_rows: List[Dict[str, Any]] = []
        i_rows: List[Dict[str, Any]] = []
        d = start
        while d <= end:
            for h in range(0, 24, self.window_hours):
                h1 = min(h + self.window_hours, 24)
                st_ = f"{d:%Y%m%d}{h:02d}0000"
                en_ = f"{d:%Y%m%d}{h1:02d}0000" if h1 < 24 else f"{d:%Y%m%d}235959"
                for inv_by_key, points, channel_map in device_groups:
                    keys = list(inv_by_key)
                    for k in range(0, len(keys), 50):
                        data = self._post("/openapi/platform/getDevicePointMinuteDataList",
                                          {"ps_key_list": keys[k:k + 50], "points": ",".join(points), "start_time_stamp": st_,
                                           "end_time_stamp": en_, "minute_interval": "5", "is_get_point_dict": "0"})
                        res.calls_used += 1
                        self._collect(data, inv_by_key, pac_pt, channel_map, s_rows, i_rows, self.pac_divisor)
            d += timedelta(days=1)
        res.strings = self.normalize_strings(pd.DataFrame(s_rows, columns=STRING_COLS))
        res.inverters = self.normalize_inverters(pd.DataFrame(i_rows, columns=INVERTER_COLS))
        return res

    @staticmethod
    def _collect(data: Dict[str, Any], inv_by_key: Dict[str, str], pac_pt: str, smap: Dict[int, Dict[str, str]],
                 s_rows: List[Dict[str, Any]], i_rows: List[Dict[str, Any]], pac_divisor: float = 1000.0) -> None:
        """result_data = {ps_key: [records]}（另有 point_dict 键要跳过）"""
        for key, recs in (data or {}).items():
            if key not in inv_by_key or not isinstance(recs, list):
                continue
            inv = inv_by_key[key]
            for rec in recs:
                ts = pd.to_datetime(str(rec.get("time_stamp")), format="%Y%m%d%H%M%S", errors="coerce")
                if pd.isna(ts):
                    continue
                pac = rec.get(pac_pt)
                pv = _num(pac)
                i_rows.append({"inverter_id": inv, "ts": ts, "pac_kw": None if pv is None else pv / pac_divisor, "status": None})
                for n, m in smap.items():
                    u, i = rec.get(m["u"]), rec.get(m["i"])
                    if u is None and i is None:
                        continue
                    s_rows.append({"string_id": f"{inv}-pv{n}", "ts": ts, "current_a": _num(i), "voltage_v": _num(u)})


def _num(v):
    try:
        return None if v is None or v == "" or v == "--" else float(v)
    except (TypeError, ValueError):
        return None
