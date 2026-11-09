"""辐照源：晴空模型、NASA POWER / Meteonorm / Open-Meteo 提供方（mock HTTP）、按序回退与缓存、Meteonorm 导出文件识别"""
import json
from datetime import date, timedelta

import pandas as pd

from pvdiag import irradiance as irr
from pvdiag.adapters.csv_adapter import CsvAdapter
from pvdiag.config import Settings
from pvdiag.storage import Store


class _Resp:
    def __init__(self, body):
        self._b = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._b


def test_clearsky_seasonal_and_latitude():
    summer = irr.clearsky_daily_kwh_m2(31.2, 121.4, date(2026, 6, 21))
    winter = irr.clearsky_daily_kwh_m2(31.2, 121.4, date(2026, 12, 21))
    north = irr.clearsky_daily_kwh_m2(45.0, 125.0, date(2026, 12, 21))
    assert 7 < summer < 10 and 2.5 < winter < 5 and north < winter


def test_providers_parse_and_fallback(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path / "d"); settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    store.execute("INSERT INTO stations (station_id, name, platform, capacity_kwp, region, lat, lon) VALUES ('S1','s','csv',100,'r',31.2,121.4)")
    days = [date(2026, 9, 1) + timedelta(days=k) for k in range(4)]
    # NASA POWER 返回前两天（第三天 -999 缺测），Meteonorm 返回第三天小时值，第四天落到晴空
    def fake_get(url, params=None, headers=None, timeout=None):
        if "power.larc.nasa.gov" in url:
            return _Resp({"properties": {"parameter": {"ALLSKY_SFC_SW_DWN": {"20260901": 5.1, "20260902": 4.8, "20260903": -999.0}}}})
        if "meteonorm" in url:
            assert headers["Authorization"] == "Bearer tok"
            return _Resp({"values": [{"time": f"2026-09-03T{h:02d}:00:00Z", "G_Gh": 600.0} for h in range(6, 16)]})
        raise AssertionError(url)
    import requests
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setenv("METEONORM_TOKEN", "tok")
    settings.irradiance.nasa_power = True
    settings.irradiance.meteonorm = True
    out = irr.resolve_daily_irradiance(settings, store, "S1", days, measured={days[0]: 5.5})
    assert out[days[0]] == (5.5, "measured")
    assert out[days[1]] == (4.8, "nasa_power")
    assert out[days[2]][1] == "meteonorm" and abs(out[days[2]][0] - 6.0) < 1e-6      # 10 小时 × 600 W/m² = 6 kWh/m²
    assert out[days[3]][1] == "clearsky" and out[days[3]][0] > 3
    # 缓存：第二次解析不再出站
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no network expected")))
    out2 = irr.resolve_daily_irradiance(settings, store, "S1", days[:3])
    assert out2[days[1]] == (4.8, "nasa_power") and out2[days[2]][1] == "meteonorm"
    # 提供方失败 → 告警并回退
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("503")))
    out3 = irr.resolve_daily_irradiance(settings, store, "S1", [date(2026, 9, 10)])
    assert out3[date(2026, 9, 10)][1] == "clearsky"
    assert store.scalar("SELECT COUNT(*) FROM alerts WHERE source='irradiance'") >= 1
    # source=none
    settings.irradiance.source = "none"
    assert irr.resolve_daily_irradiance(settings, store, "S1", days[:1])[days[0]] == (None, "none")


def test_meteonorm_export_csv_recognized(tmp_path):
    root = tmp_path / "imp"; (root / "S2" / "irradiance").mkdir(parents=True)
    rows = [{"y": 2026, "m": 9, "dm": 1, "h": h, "G_Gh": 500 if 7 <= h <= 16 else 0, "Ta": 25.0} for h in range(1, 25)]
    pd.DataFrame(rows).to_csv(root / "S2" / "irradiance" / "meteonorm_export.csv", index=False)
    res = CsvAdapter("S2", {"root": str(root)}).fetch(date(2026, 9, 1), date(2026, 9, 1))
    assert res.irradiance is not None and len(res.irradiance) == 24
    assert abs(res.irradiance["ghi_w_m2"].sum() - 5000) < 1e-6
