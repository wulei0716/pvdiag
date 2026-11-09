"""日电量模式：巡检报告宽表 → 档案/日电量/天气 → 同机(或同通道)日电量互比。
用合成电站生成"每串每日 kWh"宽表（与真实巡检报告同版式），不含任何真实客户数据。"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from pvdiag import daily_energy as de
from pvdiag.config import Settings
from pvdiag.models import Cause
from pvdiag.sim.synth import Fault, SimConfig, simulate
from pvdiag.storage import Store


def _report_xlsx(path, cfg: SimConfig, data, extra_rows=None, mppt_style=False):
    """把合成样本积分成每串每日 kWh，写成巡检报告版式：概况表 + 宽表（逆变器, 组串, 状态, 日期..., 合计, 日均）"""
    s = data["samples_strings"].copy()
    s["date"] = pd.to_datetime(s["ts"]).dt.strftime("%Y-%m-%d")
    s["e"] = s["current_a"] * s["voltage_v"] * (5 / 60) / 1000.0
    e = s.groupby(["string_id", "date"])["e"].sum().unstack()
    days = list(e.columns)
    md = [d[5:] for d in days]
    inv_daily = data["samples_inverters"].copy(); inv_daily["date"] = pd.to_datetime(inv_daily["ts"]).dt.strftime("%Y-%m-%d")
    inv_daily["hh"] = pd.to_datetime(inv_daily["ts"]).dt.hour
    noon = inv_daily[inv_daily["hh"].between(11, 13)].groupby("date")["pac_kw"].sum().reindex(days).fillna(0) / 24
    peak = noon.max()
    overview = [[f"合成站 · 组串发电量统计与分析（{days[0]} ~ {days[-1][5:]}）"], ["数据源：合成"], [None],
                ["一、逐日发电概况"], ["日期", "中午功率kW", "占晴峰%", "天气", "日发电量kWh"]]
    for d, m in zip(days, md):
        pct = round(100 * noon[d] / peak)
        overview.append([m, round(noon[d], 1), pct, "晴好" if pct >= 35 else "阴雨", round(float(e[d].sum()), 1)])
    rows = [[f"合成站 全部通道每日发电量（kWh）· {days[0]}~{days[-1][5:]}"], ["逆变器", "组串", "状态"] + md + ["合计kWh", "日均kWh"]]
    for sid in e.index:
        inv, pv = sid.split("-")[1], sid.split("-")[2]
        vals = [round(float(v), 1) for v in e.loc[sid].values]
        rows.append([inv, pv.upper().replace("PV", "MPPT") if mppt_style else pv, "接入"] + vals + [round(sum(vals), 1), round(np.mean(vals), 1)])
    for r in (extra_rows or []):
        rows.append(r + [None] * (len(md) + 2 - len(r) + 3) if len(r) < len(md) + 5 else r)
    with pd.ExcelWriter(path) as w:
        pd.DataFrame(overview).to_excel(w, sheet_name="统计与分析", header=False, index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="组串每日发电量", header=False, index=False)


def test_daily_energy_report_roundtrip_and_diagnosis(tmp_path):
    s = lambda k, n: f"DE1-INV{k:02d}-pv{n}"
    cfg = SimConfig(station_id="DE1", n_inverters=3, strings_per_inverter=10, days=20, seed=21, start=date(2026, 9, 1), use_default_faults=False,
                    faults=[Fault("string", s(1, 4), "shading_fixed", 0, {"start": "10:30", "end": "15:00", "depth": 0.45}),   # 严重遮挡 → 日电量 −25% 左右
                            Fault("string", s(2, 7), "orientation", 0, {"clear_factor": 0.90})],                            # 朝向：晴天 −10%，阴天消失
                    weather_probs=(0.7, 0.1, 0.2))
    data = simulate(cfg)
    # 额外两行：一路"零电流"（同 MPPT 另一路 pv1 在发电 → 空输入）、一路"未接"
    extra = [["INV01", "pv31", "零电流"], ["INV01", "pv40", "未接"]]
    xlsx = tmp_path / "report.xlsx"
    _report_xlsx(xlsx, cfg, data, extra_rows=extra)
    b = de.read_report_xlsx(xlsx, "DE1", "huawei", inputs_per_mppt=2)
    assert len(b.strings) == 32 and b.strings["status"].value_counts().to_dict() == {"connected": 30, "zero_current": 1, "unconnected": 1}
    assert b.station_daily["weather_tag"].isin(["clear", "overcast"]).all() and (b.station_daily["weather_tag"] == "clear").sum() >= 8
    assert b.energy["date"].min() == "2026-09-01"
    settings = Settings(data_dir=tmp_path / "data"); settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    r = de.load_into_store(store, b)
    assert r["strings"] == 32 and r["days"] == 20
    out = de.diagnose(store, "DE1", cfg.start, cfg.start + timedelta(days=19), settings.thresholds)
    st = out["strings"].set_index("string_id")
    # 零电流路 pv31 的同 MPPT 兄弟 pv32 不存在 → 无参照；改用 pv1/pv2 共 MPPT1：把 pv31 当 pv2 验证空输入规则
    assert out["group_mode"] == "inverter"
    assert st.at[s(1, 4), "grade"] == "B" and st.at[s(1, 4), "cause"] == Cause.SHADING.value, st.loc[s(1, 4)].to_dict()
    assert st.at[s(2, 7), "grade"] == "C" and st.at[s(2, 7), "cause"] == Cause.ORIENTATION.value, st.loc[s(2, 7)].to_dict()
    healthy = st[(st["state"] == "connected") & ~st.index.isin([s(1, 4), s(2, 7)])]
    assert healthy["grade"].isin(["A", "B"]).sum() == 0, healthy[healthy["grade"].notna()].to_dict("index")
    assert st.at["DE1-INV01-pv40", "state"] == "unconnected"
    # 写进 suspects，后续复核/工单链路可接
    susp = store.query("SELECT string_id, grade, rule_cause, low_confidence FROM suspects")
    assert set(susp["string_id"]) >= {s(1, 4), s(2, 7)} and (susp["low_confidence"] == 1).all()


def test_empty_input_rule_and_channel_grouping(tmp_path):
    """华为/阳光每 MPPT 两路：同 MPPT 另一路在发电的 0 电量路 = 空输入（不报故障）；小站同机串太少 → 自动改同通道跨设备互比"""
    cfg = SimConfig(station_id="DE2", n_inverters=5, strings_per_inverter=3, days=12, seed=22, start=date(2026, 9, 1), use_default_faults=False,
                    faults=[Fault("string", "DE2-INV04-pv2", "shading_fixed", 0, {"start": "10:00", "end": "15:00", "depth": 0.5})],
                    weather_probs=(0.8, 0.1, 0.1))
    data = simulate(cfg)
    extra = [["INV01", "pv4", "零电流"]]            # pv3/pv4 共 MPPT2，pv3 在发电
    xlsx = tmp_path / "small.xlsx"
    _report_xlsx(xlsx, cfg, data, extra_rows=extra, mppt_style=True)       # 通道写成 MPPT1..3（阳光云口径）
    b = de.read_report_xlsx(xlsx, "DE2", "sungrow", inputs_per_mppt=2, channel_is_mppt=False)
    settings = Settings(data_dir=tmp_path / "data"); settings.ensure_dirs()
    store = Store(settings.db_path, settings.samples_dir)
    de.load_into_store(store, b)
    out = de.diagnose(store, "DE2", cfg.start, cfg.start + timedelta(days=11), settings.thresholds)
    st = out["strings"].set_index("string_id")
    assert out["channel_states"]["DE2-INV01-pv4"] == "empty_input" and pd.isna(st.at["DE2-INV01-pv4", "grade"])
    assert out["group_mode"] == "channel", out["group_mode"]
    assert st.at["DE2-INV04-pv2", "grade"] == "B", st.loc["DE2-INV04-pv2"].to_dict()
    others = st[(st["state"] == "connected") & (st.index != "DE2-INV04-pv2")]
    assert others["grade"].isin(["A", "B"]).sum() == 0, others[others["grade"].notna()].to_dict("index")
