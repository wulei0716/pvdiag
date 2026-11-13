"""云平台适配器：按现网实测的响应结构（合成样例）校验解析、时区、命名、空输入/未接判别。不发任何真实请求。"""
from datetime import date, timedelta

import pandas as pd

from pvdiag.adapters.huawei import HuaweiAdapter, QuotaPlanner
from pvdiag.adapters.sungrow import SungrowAdapter, SG110CX_POINTS, _norm_inv_name


def _hw_record(dev_id, ts_ms, pv_i, pv_u, caps, active_kw):
    m = {"active_power": active_kw, "inverter_state": 512.0, "temperature": 40.0, "mppt_total_cap": sum(caps.values()), "run_state": 1}
    for n in range(1, 37):
        m[f"pv{n}_i"] = pv_i.get(n)           # 没有的路 → None（机型只有 20/24 路）
        m[f"pv{n}_u"] = pv_u.get(n)
    for k, v in caps.items():
        m[f"mppt_{k}_cap"] = v
    return {"devId": dev_id, "sn": f"SN{dev_id}", "collectTime": ts_ms, "dataItemMap": m}


def test_huawei_fivemin_parse_timezone_naming_and_hints():
    # 北京时间 2026-10-28 12:00 = UTC 04:00
    ts_ms = int((pd.Timestamp("2026-10-28 04:00:00") - pd.Timestamp("1970-01-01")).total_seconds() * 1000)
    # 20 路：奇数路有电流；偶数路电压随兄弟路、电流 0（空输入）；pv21~24 电压电流都 0（未接）；pv25+ None
    pv_i = {n: (8.6 if n % 2 == 1 else 0.0) for n in range(1, 21)}; pv_u = {n: 649.0 for n in range(1, 21)}
    for n in range(21, 25):
        pv_i[n] = 0.0; pv_u[n] = 0.0
    caps = {k: 10952.0 for k in range(1, 10)}; caps[10] = 18391.0
    data = [_hw_record(1000000354233295, ts_ms, pv_i, pv_u, caps, 57.5), _hw_record(1000000354233295, ts_ms + 300000, pv_i, pv_u, caps, 58.0)]
    s, i, h = HuaweiAdapter._parse_history(data, {"1000000354233295": "XWZ3-NB01"}, timedelta(hours=8))
    assert i["ts"].iloc[0] == pd.Timestamp("2026-10-28 12:00:00") and i["inverter_id"].iloc[0] == "XWZ3-NB01"
    assert set(s["string_id"].str.extract(r"pv(\d+)$")[0].astype(int)) == set(range(1, 25))       # None 的路不输出
    assert s["string_id"].str.startswith("XWZ3-NB01-pv").all()
    assert len(h) == 10 and h.loc[h["mppt"] == 10, "cap_w"].iloc[0] == 18391.0
    # 档案建议：奇数路 = 20 块（10952/550）；偶数路空输入；MPPT10 容量 18391 → 疑 2 并联×17 块；pv21~24 未接
    hints = HuaweiAdapter.profile_hints(s, h, module_wp=550, inputs_per_mppt=2, typical_n_modules=20)
    hi = hints.set_index("string_id")
    assert hi.at["XWZ3-NB01-pv1", "n_modules"] == 20 and hi.at["XWZ3-NB01-pv1", "state"] == "connected"
    assert hi.at["XWZ3-NB01-pv2", "state"] == "empty_input" and pd.isna(hi.at["XWZ3-NB01-pv2", "n_modules"])
    assert hi.at["XWZ3-NB01-pv19", "n_parallel"] == 2 and hi.at["XWZ3-NB01-pv19", "n_modules"] == 17
    assert hi.at["XWZ3-NB01-pv21", "state"] == "unconnected"
    # 配额：20 台 → 25 次/日；登录 1 + 每日增量 1 → 还能回补 23 天
    q = QuotaPlanner(20)
    assert q.daily_quota == 25 and q.batches_per_day(used_today=0) == 23


def test_huawei_collect_time_is_local_noon():
    a = HuaweiAdapter("XWZ3", {"user_name": "u", "system_code": "x", "station_code": "NE=1"})
    ms = a._collect_time_ms(date(2026, 10, 26))
    assert pd.Timestamp(ms, unit="ms") == pd.Timestamp("2026-10-26 04:00:00")      # 北京 12:00 = UTC 04:00
    assert a.history_api == "fivemin" and a.batch_days == 1 and a.min_interval == 65


def test_sungrow_minute_data_parse_and_points():
    assert _norm_inv_name("逆变器2") == "INV02" and _norm_inv_name("NB01") == "NB01"
    pm = SG110CX_POINTS["strings"]
    assert pm[1] == {"u": "p5", "i": "p6"} and pm[4] == {"u": "p45", "i": "p46"} and pm[9] == {"u": "p55", "i": "p56"}
    data = {"5119111_1_1_1": [{"time_stamp": "20261025120000", "p24": "85600", "p5": "612.3", "p6": "13.7", "p7": "610.0", "p8": "7.0", "p9": "110.2", "p10": "0"},
                              {"time_stamp": "20261025120500", "p24": "86000", "p5": "612.0", "p6": "13.8", "p7": "609.5", "p8": "7.1", "p9": "109.9", "p10": "0"}],
            "point_dict": [{"point_id": "p5", "point_name": "MPPT1 电压"}]}
    s_rows, i_rows = [], []
    SungrowAdapter._collect(data, {"5119111_1_1_1": "LH-INV02"}, "p24", pm, s_rows, i_rows, 1000.0)
    s = pd.DataFrame(s_rows); i = pd.DataFrame(i_rows)
    assert i["pac_kw"].iloc[0] == 85.6 and i["ts"].iloc[0] == pd.Timestamp("2026-10-25 12:00:00")
    assert set(s["string_id"]) == {"LH-INV02-pv1", "LH-INV02-pv2", "LH-INV02-pv3"}       # 只出现有测点的 MPPT
    r = s[s["string_id"] == "LH-INV02-pv3"].iloc[0]
    assert r["voltage_v"] < 400 and r["current_a"] == 0                                 # 空通道口径：电压浮空 <400V


class _FakeResp:
    def __init__(self, body, cookies=None, headers=None):
        self._b = body; self.cookies = cookies or {}; self.headers = headers or {}; self.status_code = 200
    def raise_for_status(self): pass
    def json(self): return self._b


def test_huawei_login_cookie_token_and_fetch_flow(monkeypatch, tmp_path):
    """用假 Session 走完 登录(Cookie XSRF-TOKEN) → getDevList → getDevFiveMinutes(按天, collectTime=北京正午) 的完整链路，并入库成 5 分钟样本"""
    import requests
    calls = []
    ts0 = int((pd.Timestamp("2026-10-28 00:00:00") - pd.Timestamp("1970-01-01")).total_seconds() * 1000)   # 北京 08:00
    pv_i = {n: (8.0 if n % 2 else 0.0) for n in range(1, 21)}; pv_u = {n: 640.0 for n in range(1, 21)}
    caps = {k: 11000.0 for k in range(1, 11)}

    def fake_post(self, url, json=None, timeout=None, headers=None, **kw):
        path = url.split("huawei.com")[1]
        calls.append((path, json))
        if path == "/thirdData/login":
            self.cookies.set("XSRF-TOKEN", "tok-abc")
            return _FakeResp({"success": True, "failCode": 0}, headers={"xsrf-token": "tok-abc"})
        assert self.headers.get("XSRF-TOKEN") == "tok-abc"
        if path == "/thirdData/getDevList":
            return _FakeResp({"success": True, "data": [{"id": 1000000354233295, "devName": "NB01", "devTypeId": 1, "esnCode": "SN1"},
                                                        {"id": 1000000354233296, "devName": "NB02", "devTypeId": 1, "esnCode": "SN2"},
                                                        {"id": 99, "devName": "表", "devTypeId": 17}]})
        if path == "/thirdData/getDevFiveMinutes":
            assert json["devTypeId"] == 1 and set(json["devIds"].split(",")) == {"1000000354233295", "1000000354233296"}
            day_ms = json["collectTime"]
            assert pd.Timestamp(day_ms, unit="ms").hour == 4             # 北京正午 = UTC 04:00
            recs = []
            for dev in (1000000354233295, 1000000354233296):
                for k in range(0, 288, 24):                               # 每 2 小时一个点，够用
                    t = day_ms - 12 * 3600 * 1000 + k * 300000                 # 该天北京 00:00 起
                    recs.append(_hw_record(dev, t, pv_i, pv_u, caps, 50.0))
            return _FakeResp({"success": True, "data": recs})
        raise AssertionError(path)

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    state = {}
    a = HuaweiAdapter("XWZ3", {"user_name": "u", "system_code": "x", "station_code": "NE=1", "min_interval_seconds": 65},
                      state_get=lambda k, d=None: state.get(k, d), state_set=lambda k, v: state.__setitem__(k, v))
    res = a.fetch(date(2026, 10, 28), date(2026, 10, 29))
    paths = [p for p, _ in calls]
    assert paths.count("/thirdData/login") == 1 and paths.count("/thirdData/getDevFiveMinutes") == 2        # token 复用；一天一次
    assert res.calls_used == 2 and a.calls_used_today() == 4                                                  # 登录 1 + 设备 1 + 数据 2
    assert set(res.inverters["inverter_id"]) == {"XWZ3-NB01", "XWZ3-NB02"}
    assert res.strings["string_id"].str.match(r"^XWZ3-NB0[12]-pv\d+$").all()
    assert res.strings["ts"].dt.date.nunique() == 2 and res.strings["ts"].min() >= pd.Timestamp("2026-10-28")
    assert state["huawei:XWZ3:mppt_caps"] and state["huawei:XWZ3:devmap"]["XWZ3-NB01"] == "1000000354233295"


def test_huawei_station_end_to_end_with_auto_profile(monkeypatch, tmp_path):
    """新华为站：只登记凭证 → 流水线拉数 → 自动档案骨架 → 基线/诊断能跑；空输入（偶数路电压随兄弟路、电流 0）不报断线 A"""
    import numpy as np
    import requests
    from pvdiag.config import Settings
    from pvdiag.storage import Store
    from pvdiag.agents.profile import ProfileAgent
    from pvdiag.orchestrator import DailyPipeline

    caps = {k: 11000.0 for k in range(1, 11)}
    minutes = np.arange(0, 1440, 5)
    elev = np.clip(np.sin((minutes / 60 - 6) / 12 * np.pi), 0, None)      # 6:00~18:00 钟形

    def fake_post(self, url, json=None, timeout=None, headers=None, **kw):
        path = url.split("huawei.com")[1]
        if path == "/thirdData/login":
            self.cookies.set("XSRF-TOKEN", "tok")
            return _FakeResp({"success": True})
        if path == "/thirdData/getDevList":
            return _FakeResp({"success": True, "data": [{"id": 101, "devName": "NB01", "devTypeId": 1}, {"id": 102, "devName": "NB02", "devTypeId": 1}]})
        if path == "/thirdData/getDevFiveMinutes":
            day0 = json["collectTime"] - 12 * 3600 * 1000
            rng = np.random.default_rng(json["collectTime"] % 1000)
            recs = []
            for dev in (101, 102):
                for k, m in enumerate(minutes):
                    g = elev[k]
                    pv_i = {n: (float(9.5 * g * (1 + rng.normal(0, 0.005))) if n % 2 else 0.0) for n in range(1, 21)}
                    if dev == 102:
                        pv_i[7] *= 0.6                                               # NB02-pv7 严重低发
                    pv_u = {n: (650.0 if g > 0.02 else 0.0) for n in range(1, 21)}
                    recs.append(_hw_record(dev, day0 + int(m) * 60000, pv_i, pv_u, caps, float(sum(pv_i.values()) * 650 / 1000)))
            return _FakeResp({"success": True, "data": recs})
        raise AssertionError(path)

    monkeypatch.setattr(requests.Session, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("PVDIAG_HW_TEST", "secret")
    settings = Settings(data_dir=tmp_path / "data"); settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    ProfileAgent.import_stations(store, pd.DataFrame([{"station_id": "HW1", "name": "HW1", "platform": "huawei", "capacity_kwp": 220.0, "lat": 30.2, "lon": 119.7,
                                                       "extra_json": '{"platform_config": {"user_name": "u", "system_code_env": "PVDIAG_HW_TEST", "station_code": "NE=1"}}'}]))
    end = date(2026, 10, 29)
    pipe = DailyPipeline(settings, store, ingest=True, backfill_days=4)
    rep = pipe.run(end)
    assert rep["agents"]["ingest"]["status"] in ("ok", "warn"), rep["agents"]["ingest"]
    strings = store.strings("HW1"); inv = store.inverters("HW1")
    assert len(inv) == 2 and len(strings) == 40 and (strings["notes"] == "auto:skeleton").all()
    assert strings.set_index("string_id").at["HW1-NB01-pv3", "mppt"] == 2
    assert len(store.available_dates("string", "HW1")) == 5
    susp = store.query("SELECT string_id, grade, rule_cause FROM suspects WHERE scope='string'")
    # 偶数路（空输入）不应被判 A 断线
    empties = susp[susp["string_id"].str.extract(r"pv(\d+)$")[0].astype(int) % 2 == 0]
    assert empties[empties["grade"] == "A"].empty, empties.to_dict("records")
    # 低发串在冷启动期（5 天）至少进观察带
    assert not susp[susp["string_id"] == "HW1-NB02-pv7"].empty, susp.to_dict("records")


def test_ginlong_signature_and_inverter_day_flow(monkeypatch):
    """锦浪：HMAC-SHA1 签名头（Content-Type 不带 charset）、userStationList/inverterList/inverterDay 链路、timeStr 时间、pac W→kW、按名字命名"""
    import base64, hashlib, hmac, json as _json
    import requests
    from pvdiag.adapters.ginlong import GinlongAdapter
    seen = []

    def fake_post(url, data=None, headers=None, timeout=None, **kw):
        path = url.split(":13333")[1]
        seen.append((path, _json.loads(data), headers))
        assert headers["Content-Type"] == "application/json" and headers["Authorization"].startswith("API kid:")
        md5 = base64.b64encode(hashlib.md5(data.encode()).digest()).decode()
        assert headers["Content-MD5"] == md5
        sign = base64.b64encode(hmac.new(b"ksec", f"POST\n{md5}\napplication/json\n{headers['Date']}\n{path}".encode(), hashlib.sha1).digest()).decode()
        assert headers["Authorization"] == f"API kid:{sign}"
        if path == "/v1/api/userStationList":
            return _FakeResp({"success": True, "code": "0", "data": {"page": {"records": [{"id": "1300", "stationName": "乾呈格来", "capacity": 500, "inverterCount": 9}]}}})
        if path == "/v1/api/inverterList":
            return _FakeResp({"success": True, "code": "0", "data": {"page": {"records": [{"sn": "SN001", "name": "1号逆变器", "productModel": "S5-GC100K", "power": 100, "stationId": "1300"}]}}})
        if path == "/v1/api/inverterDay":
            assert data and _json.loads(data)["timeZone"] == "8"
            return _FakeResp({"success": True, "code": "0", "data": [
                {"timeStr": "2026-11-10 12:00:00", "dataTimestamp": "1794283200000", "pac": 85600, "state": 1, "iPv1": 9.1, "uPv1": 640.0, "iPv2": 0.0, "uPv2": 641.0, "iPv3": None},
                {"timeStr": "2026-11-10 12:05:00", "dataTimestamp": "1794283500000", "pac": 86000, "state": 1, "iPv1": 9.2, "uPv1": 639.0, "iPv2": 0.0, "uPv2": 640.0}]})
        raise AssertionError(path)

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr("time.sleep", lambda s: None)
    monkeypatch.setenv("PVDIAG_GL_TEST", "ksec")
    a = GinlongAdapter("QCGL", {"key_id": "kid", "key_secret_env": "PVDIAG_GL_TEST", "plant_id": "1300"})
    st = a.list_stations(); assert st["plant_id"].iloc[0] == "1300"
    res = a.fetch(date(2026, 11, 10), date(2026, 11, 10))
    assert res.calls_used == 1 and seen[-1][1] == {"sn": "SN001", "money": "CNY", "time": "2026-11-10", "timeZone": "8"}
    assert res.inverters["inverter_id"].iloc[0] == "QCGL-1号逆变器" and abs(res.inverters["pac_kw"].iloc[0] - 85.6) < 1e-9
    assert res.inverters["ts"].iloc[0] == pd.Timestamp("2026-11-10 12:00:00")
    assert set(res.strings["string_id"]) == {"QCGL-1号逆变器-pv1", "QCGL-1号逆变器-pv2"}          # iPv3 为 None 不输出
