"""基准评测：把公开数据灌进全新的数据目录，跑完整每日闭环，与参考标签对照打分。

评分口径（写在报告里，别让数字脱离口径）：
- Kaggle：
  * 召回：参考标签 underperform 的"组串"（真实逆变器）是否被 pvdiag 以 A/B/D 级报出（持续低发 ≥ -3%）
  * 误报：参考标签 normal 且无间歇停机的组串，不应出现 A/B 级嫌疑
  * 停机识别：intermittent_outage 的组串，pvdiag 给出的原因应是 间歇停机/掉线 或 数据问题（而不是遮挡/二极管）
  * 晴好天判定：pvdiag 的 weather_tag（只用功率）vs 气象站辐照独立判定 的一致率
- NREL（pvanalytics 标注）：
  * 钉死：日内标注钉死样本占比 ≥30% 的日子 → pvdiag daily_features.stuck 或 stale_run_min ≥ 90 应为真；其余日子应为假
  * 限发：NREL 人工标注当日限发点 ≥4（≥1 小时）→ pvdiag inverter_daily.limited_flag 应为真；其余日子应为假
"""
from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..agents.base import AgentContext
from ..agents.ingest import IngestAgent
from ..agents.profile import ProfileAgent
from ..config import Settings
from ..models import Cause
from ..orchestrator import DailyPipeline
from ..storage import Store
from .datasets import dataset_dir, fetch_dataset, is_cached


def _fresh_settings(workdir: Path, name: str, settings: Optional[Settings] = None) -> Settings:
    s = (settings or Settings()).model_copy(deep=True)
    s.data_dir = workdir / name
    if s.data_dir.exists():
        shutil.rmtree(s.data_dir)
    s.ensure_dirs()
    return s


def load_bundle(settings: Settings, bundle: Dict[str, pd.DataFrame]) -> Store:
    store = Store(settings.db_path, settings.samples_dir)
    ProfileAgent.import_stations(store, bundle["stations"])
    ProfileAgent.import_inverters(store, bundle["inverters"])
    ProfileAgent.import_strings(store, bundle["strings"])
    end = pd.to_datetime(bundle["samples_strings"]["ts"]).max().date()
    ctx = AgentContext(settings=settings, store=store, run_date=end)
    sid = bundle["stations"]["station_id"].iloc[0]
    IngestAgent().ingest_frames(ctx, sid, bundle["samples_strings"], bundle["samples_inverters"], bundle.get("irradiance"))
    return store


# ---------------------------------------------------------------------------- Kaggle
def bench_kaggle(workdir: Path, settings: Optional[Settings] = None, plants=(1, 2), public_root=None,
                 fetch: bool = True) -> Dict:
    from .kaggle_solar import irradiance_clear_days, load_kaggle_plant
    if not is_cached("kaggle_solar", public_root):
        if not fetch:
            raise FileNotFoundError("kaggle_solar 未下载，先执行 pvdiag public fetch kaggle_solar")
        fetch_dataset("kaggle_solar", public_root)
    root = dataset_dir("kaggle_solar", public_root)
    out: Dict = {"dataset": "kaggle_solar", "plants": {}}
    for plant in plants:
        bundle = load_kaggle_plant(root, plant)
        s = _fresh_settings(workdir, f"kaggle_p{plant}", settings)
        store = load_bundle(s, bundle)
        ts = pd.to_datetime(bundle["samples_strings"]["ts"])
        start, end = ts.min().date(), ts.max().date()
        pipe = DailyPipeline(s, store=store, ingest=False)
        pipe.backfill(start, end)
        out["plants"][str(plant)] = score_kaggle(store, bundle, irradiance_clear_days(bundle["weather_raw"]), start, end)
        store.close()
    out["summary"] = _kaggle_summary(out["plants"])
    return out


def score_kaggle(store: Store, bundle: Dict[str, pd.DataFrame], ref_weather: pd.DataFrame, start: date, end: date) -> Dict:
    ref = bundle["reference"].set_index("string_id")
    station_id = bundle["stations"]["station_id"].iloc[0]
    susp = store.query("SELECT * FROM suspects WHERE scope='string' AND station_id=?", (station_id,))
    verd = store.query("SELECT suspect_id, cause, confidence, action FROM verdicts")
    susp = susp.merge(verd, on="suspect_id", how="left")
    by_string = {sid: g for sid, g in susp.groupby("string_id")}
    rows = []
    for sid, r in ref.iterrows():
        g = by_string.get(sid)
        grades = sorted(set(g["grade"])) if g is not None else []
        causes = sorted(set(g["rule_cause"].dropna())) if g is not None else []
        ai = sorted(set(g["cause"].dropna())) if g is not None else []
        rows.append({"string_id": sid, "source_key": r["source_key"], "label": r["label"], "intermittent_outage": int(r["intermittent_outage"]),
                     "median_clean_ratio": r["median_clean_ratio"], "dropout_days": int(r["dropout_days"]),
                     "grades": grades, "rule_causes": causes, "ai_causes": ai,
                     "flag_ab": int(any(x in ("A", "B") for x in grades)), "flag_abd": int(any(x in ("A", "B", "D") for x in grades)),
                     "n_cases": int(len(g)) if g is not None else 0})
    tbl = pd.DataFrame(rows)
    under = tbl[tbl["label"] == "underperform"]
    healthy = tbl[(tbl["label"] == "normal") & (tbl["dropout_days"] == 0)]          # 完全没掉过线的正常机器才算"健康"
    sporadic = tbl[(tbl["label"] == "normal") & (tbl["dropout_days"] > 0) & (tbl["intermittent_outage"] == 0)]  # 偶发掉线 1~4 天：报停机合理
    outage = tbl[tbl["intermittent_outage"] == 1]
    outage_ok = outage["rule_causes"].map(lambda cs: any(c in (Cause.OUTAGE.value, Cause.DATA_ISSUE.value, Cause.OPEN_CIRCUIT.value) for c in cs))
    # 晴好天一致率
    sd = store.station_daily(station_id, start, end)[["date", "weather_tag"]]
    wx = sd.merge(ref_weather, on="date", how="inner")
    wx_valid = wx[wx["weather_tag"] != "nodata"]
    agree = float((wx_valid["weather_tag"] == wx_valid["ref_tag"]).mean()) if len(wx_valid) else np.nan
    # 晴好天 vs 非晴好天二分一致率（cloudy/overcast 都视为"不参与判定"）
    bi = float(((wx_valid["weather_tag"] == "clear") == (wx_valid["ref_tag"] == "clear")).mean()) if len(wx_valid) else np.nan
    return {
        "station_id": station_id, "n_strings": int(len(tbl)), "days": int((end - start).days + 1),
        "underperform": {"n": int(len(under)), "recalled_abd": int(under["flag_abd"].sum()), "recalled_ab": int(under["flag_ab"].sum()),
                         "ids": under["source_key"].tolist()},
        "healthy": {"n": int(len(healthy)), "false_alarm_ab": int(healthy["flag_ab"].sum()),
                    "false_alarm_ids": healthy[healthy["flag_ab"] == 1]["source_key"].tolist()},
        "sporadic_outage": {"n": int(len(sporadic)), "flagged_ab": int(sporadic["flag_ab"].sum()),
                            "causes": sporadic.explode("rule_causes")["rule_causes"].value_counts().to_dict() if len(sporadic) else {}},
        "outage": {"n": int(len(outage)), "classified_as_outage_or_data": int(outage_ok.sum()) if len(outage) else 0,
                   "causes": {k: v for k, v in outage.explode("rule_causes")["rule_causes"].value_counts().to_dict().items()} if len(outage) else {}},
        "weather_agreement": {"n_days": int(len(wx_valid)), "exact": None if np.isnan(agree) else round(agree, 3),
                              "clear_vs_not": None if np.isnan(bi) else round(bi, 3),
                              "pvdiag_tags": sd["weather_tag"].value_counts().to_dict(), "ref_tags": ref_weather["ref_tag"].value_counts().to_dict()},
        "strings": tbl.to_dict("records"),
    }


def _kaggle_summary(plants: Dict) -> Dict:
    n_u = sum(p["underperform"]["n"] for p in plants.values())
    r_u = sum(p["underperform"]["recalled_abd"] for p in plants.values())
    n_h = sum(p["healthy"]["n"] for p in plants.values())
    fa = sum(p["healthy"]["false_alarm_ab"] for p in plants.values())
    n_o = sum(p["outage"]["n"] for p in plants.values())
    o_ok = sum(p["outage"]["classified_as_outage_or_data"] for p in plants.values())
    wx = [p["weather_agreement"]["clear_vs_not"] for p in plants.values() if p["weather_agreement"]["clear_vs_not"] is not None]
    return {"underperform_recall": f"{r_u}/{n_u}", "healthy_false_alarm_ab": f"{fa}/{n_h}",
            "outage_classified": f"{o_ok}/{n_o}", "weather_clear_agreement": round(float(np.mean(wx)), 3) if wx else None}


# ---------------------------------------------------------------------------- NREL / pvanalytics
def bench_nrel(workdir: Path, settings: Optional[Settings] = None) -> Dict:
    from .nrel import load_pvanalytics_series, PVANALYTICS_SERIES
    out: Dict = {"dataset": "nrel_pvanalytics", "series": {}}
    for key in PVANALYTICS_SERIES:
        bundle = load_pvanalytics_series(key)
        s = _fresh_settings(workdir, f"nrel_{key}", settings)
        store = load_bundle(s, bundle)
        out["series"][key] = score_nrel(store, bundle)
        store.close()
    out["summary"] = _nrel_summary(out["series"])
    return out


def score_nrel(store: Store, bundle: Dict[str, pd.DataFrame]) -> Dict:
    station_id = bundle["stations"]["station_id"].iloc[0]
    ref = bundle["reference_daily"].copy()           # date, stale_share, clip_points, interp_share, outlier_n
    df = store.query("SELECT f.date, f.stuck, f.stale_run_min FROM daily_features f JOIN strings s ON s.string_id=f.string_id WHERE s.station_id=?", (station_id,))
    inv = store.query("SELECT i.date, i.limited_flag FROM inverter_daily i JOIN inverters v ON v.inverter_id=i.inverter_id WHERE v.station_id=?", (station_id,))
    m = ref.merge(df, on="date", how="left").merge(inv, on="date", how="left")
    res: Dict = {"station_id": station_id, "n_days": int(len(m))}
    if "stale_share" in m.columns and m["stale_share"].notna().any():
        truth = m["stale_share"] >= 0.30
        pred = (m["stuck"].fillna(0) == 1) | (m["stale_run_min"].fillna(0) >= 90)   # 与 thresholds.stale_run_min 默认一致
        res["stale"] = _prf(truth, pred)
    if "clip_points" in m.columns and m["clip_points"].notna().any():
        truth = m["clip_points"] >= 4           # 15 分钟数据 ≥4 个限发点 = ≥1 小时平顶
        pred = m["limited_flag"].fillna(0) == 1
        res["clipping"] = _prf(truth, pred)
    res["days"] = m.to_dict("records")
    return res


def _prf(truth: pd.Series, pred: pd.Series) -> Dict:
    truth, pred = truth.astype(bool), pred.astype(bool)
    tp = int((truth & pred).sum()); fp = int((~truth & pred).sum()); fn = int((truth & ~pred).sum()); tn = int((~truth & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": None if prec is None else round(prec, 3),
            "recall": None if rec is None else round(rec, 3), "accuracy": round((tp + tn) / max(1, len(truth)), 3), "n_positive_days": int(truth.sum())}


def _nrel_summary(series: Dict) -> Dict:
    out = {}
    for k, v in series.items():
        for metric in ("stale", "clipping"):
            if metric in v:
                out[f"{k}.{metric}"] = {kk: v[metric][kk] for kk in ("precision", "recall", "accuracy", "n_positive_days")}
    return out


# ---------------------------------------------------------------------------- 用公开数据训练 AI 判定模型
def bench_train(workdir: Path, settings: Optional[Settings] = None, public_root=None, fetch: bool = True, with_synth: bool = True) -> Dict:
    """把 Kaggle 两站（+ 一座合成电站）灌进同一个库，用参考标签/注入真值自动"复核"，训练分类模型并按时间留出集考试。
    标注口径：
    - Kaggle underperform 串的案例 → 确认（原因取规则原因，未知则记 组件衰减/积灰）
    - Kaggle 间歇掉线串的案例 → 确认 间歇停机/掉线
    - Kaggle 从未掉线的正常串的案例 → 驳回（云影）
    - 合成电站：按注入真值确认（kind → Cause），健康串案例驳回
    这是"弱标注"（由统计参考而非现场确认得出），用途是验证训练/考试/上线链路在真实数据上能跑通，
    以及模型相对规则代理的增益，不等同于现场标注。"""
    from .kaggle_solar import load_kaggle_plant
    from ..agents.review import ReviewAgent
    from ..agents.classify import ClassifyAgent
    if not is_cached("kaggle_solar", public_root):
        if not fetch:
            raise FileNotFoundError("kaggle_solar 未下载")
        fetch_dataset("kaggle_solar", public_root)
    root = dataset_dir("kaggle_solar", public_root)
    s = _fresh_settings(workdir, "train", settings)
    store = Store(s.db_path, s.samples_dir)
    refs: Dict[str, Dict] = {}
    spans = []
    for plant in (1, 2):
        b = load_kaggle_plant(root, plant)
        ProfileAgent.import_stations(store, b["stations"]); ProfileAgent.import_inverters(store, b["inverters"]); ProfileAgent.import_strings(store, b["strings"])
        ts = pd.to_datetime(b["samples_strings"]["ts"])
        ctx = AgentContext(settings=s, store=store, run_date=ts.max().date())
        IngestAgent().ingest_frames(ctx, b["stations"]["station_id"].iloc[0], b["samples_strings"], b["samples_inverters"], b["irradiance"])
        spans.append((ts.min().date(), ts.max().date()))
        for r in b["reference"].to_dict("records"):
            refs[r["string_id"]] = {"kind": "outage" if r["intermittent_outage"] else r["label"], "dropout_days": r["dropout_days"]}
    truth_kind: Dict[str, str] = {}
    if with_synth:
        from ..sim import SimConfig, simulate
        # 合成站提前 30 天开始（给基线留样本），默认故障在第 34~42 天注入 → 正好落在 Kaggle 的 34 天窗口里
        cfg = SimConfig(station_id="SYN1", station_name="合成电站（训练用）", days=64, seed=21, start=spans[0][0] - timedelta(days=30))
        data = simulate(cfg)
        ProfileAgent.import_stations(store, data["stations"]); ProfileAgent.import_inverters(store, data["inverters"]); ProfileAgent.import_strings(store, data["strings"])
        ctx = AgentContext(settings=s, store=store, run_date=spans[0][1])
        IngestAgent().ingest_frames(ctx, "SYN1", data["samples_strings"], data["samples_inverters"])
        for t in data["truth"].to_dict("records"):
            truth_kind[t["target"]] = t["kind"]
    start, end = min(a for a, _ in spans), max(b_ for _, b_ in spans)
    DailyPipeline(s, store=store, ingest=False).backfill(start, end)
    # ---- 自动"复核"
    kind_cause = {"open_circuit": Cause.OPEN_CIRCUIT, "diode_bypass": Cause.BYPASS_DIODE, "shading_fixed": Cause.SHADING,
                  "orientation": Cause.ORIENTATION, "dust_local": Cause.DEGRADATION_DUST, "stuck": Cause.DATA_ISSUE,
                  "wrong_profile": Cause.PROFILE_ERROR, "degradation_step": Cause.DEGRADATION_DUST,
                  "outage_intermittent": Cause.OUTAGE, "stale_partial": Cause.DATA_ISSUE, "mppt_fault": Cause.MPPT}
    susp = store.query("SELECT suspect_id, string_id, inverter_id, scope, grade, rule_cause FROM suspects WHERE grade!='已备案'")
    n_confirm = n_reject = 0
    for r in susp.to_dict("records"):
        key = r["string_id"] if r["scope"] == "string" else r["inverter_id"]
        if key in refs:
            k = refs[key]["kind"]
            if k == "underperform":
                ReviewAgent.confirm(store, r["suspect_id"], Cause.DEGRADATION_DUST.value if r["rule_cause"] == Cause.UNKNOWN.value else None, "reference"); n_confirm += 1
            elif k == "outage":
                ReviewAgent.confirm(store, r["suspect_id"], Cause.OUTAGE.value, "reference"); n_confirm += 1
            elif k == "normal" and refs[key]["dropout_days"] == 0:
                ReviewAgent.reject(store, r["suspect_id"], "云影", "reference"); n_reject += 1
        elif with_synth and (key in truth_kind or key.startswith("SYN1")):
            if key in truth_kind:
                ReviewAgent.confirm(store, r["suspect_id"], kind_cause[truth_kind[key]].value, "truth"); n_confirm += 1
            else:
                ReviewAgent.reject(store, r["suspect_id"], "云影", "truth"); n_reject += 1
    # ---- 训练 + 考试
    ctx = AgentContext(settings=s, store=store, run_date=end)
    s.confidence.min_labels_for_ml = min(s.confidence.min_labels_for_ml, 30)
    res = ClassifyAgent().train(ctx, reason="公开数据弱标注训练")
    out = {"n_confirm": n_confirm, "n_reject": n_reject, "n_labels": res.get("n_labels"), "trained": res.get("trained"),
           "promoted": res.get("promoted"), "version": res.get("version"), "exam": res.get("exam"),
           "classes": res.get("classes"), "feature_importance": res.get("feature_importance"), "rejected_because": res.get("rejected_because")}
    # 上线后再判一遍：模型版本应为 ML
    if res.get("promoted"):
        r2 = ClassifyAgent()(ctx)
        out["verdict_model"] = r2.metrics.get("model")
    store.close()
    return out


# ---------------------------------------------------------------------------- 报告
def write_report(report: Dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out_dir / "bench_report.md").write_text(render_markdown(report), encoding="utf-8")
    return out_dir / "bench_report.md"


def render_markdown(rep: Dict) -> str:
    L: List[str] = [f"# 公开数据基准评测报告（{rep.get('generated_at', '')}）", ""]
    if "kaggle" in rep:
        k = rep["kaggle"]
        L += ["## Kaggle · Solar Power Generation Data（22 台逆变器 → 22 路「组串」，34 天）", "",
              f"- 低发召回（A/B/D）：**{k['summary']['underperform_recall']}**　健康串 A/B 误报：**{k['summary']['healthy_false_alarm_ab']}**　"
              f"间歇停机识别为停机/数据问题：**{k['summary']['outage_classified']}**　晴好天判定一致率（仅用功率 vs 气象站辐照）：**{k['summary']['weather_clear_agreement']}**", ""]
        for p, v in k["plants"].items():
            L += [f"### Plant {p}（{v['station_id']}）", "",
                  f"- 低发参考 {v['underperform']['ids']} → 召回 {v['underperform']['recalled_abd']}/{v['underperform']['n']}",
                  f"- 健康 {v['healthy']['n']} 路，A/B 误报 {v['healthy']['false_alarm_ab']} {v['healthy']['false_alarm_ids']}",
                  f"- 间歇停机 {v['outage']['n']} 路，规则原因分布 {v['outage']['causes']}",
                  f"- 天气：pvdiag {v['weather_agreement']['pvdiag_tags']}；气象站 {v['weather_agreement']['ref_tags']}；晴/非晴一致率 {v['weather_agreement']['clear_vs_not']}", "",
                  "| 组串 | SOURCE_KEY | 参考标签 | 干净日电量比 | 掉线天数 | pvdiag 分级 | 规则原因 | AI 原因 |", "|---|---|---|---|---|---|---|---|"]
            for r in v["strings"]:
                L.append(f"| {r['string_id'].split('-')[-1]} | {r['source_key']} | {r['label']}{'+停机' if r['intermittent_outage'] else ''} | "
                         f"{r['median_clean_ratio']} | {r['dropout_days']} | {''.join(r['grades']) or '-'} | {'/'.join(r['rule_causes']) or '-'} | {'/'.join(r['ai_causes']) or '-'} |")
            L.append("")
    if "nrel" in rep:
        n = rep["nrel"]
        L += ["## NREL 真实逆变器片段（pvanalytics 标注）", "", "| 序列·指标 | 精确率 | 召回 | 准确率 | 阳性天数 |", "|---|---|---|---|---|"]
        for k, v in n["summary"].items():
            L.append(f"| {k} | {v['precision']} | {v['recall']} | {v['accuracy']} | {v['n_positive_days']} |")
        L.append("")
    if "train" in rep:
        t = rep["train"]
        ex = t.get("exam") or {}
        L += ["## 用公开数据训练 AI 判定模型（弱标注：参考标签 / 注入真值）", "",
              f"- 自动复核：确认 {t['n_confirm']} 条、驳回 {t['n_reject']} 条 → 标注 {t['n_labels']} 条，类别 {t.get('classes')}",
              f"- 考试方式 {ex.get('method')}：挑战者 {ex.get('challenger')} · 规则代理 {ex.get('rule_proxy')}",
              f"- 结果：{'已上线 ' + str(t.get('version')) if t.get('promoted') else '未上线（' + '；'.join(t.get('rejected_because') or ['标注不足']) + '）'}",
              f"- 特征重要性 Top：{t.get('feature_importance')}", ""]
    if "synth" in rep:
        s = rep["synth"]
        L += ["## 合成电站（10 种注入故障）", "", f"- 召回 {s['recall']}，健康串 A/B 误报 {s['false_alarm_ab']}，原因分型正确 {s['cause_correct']}", ""]
    return "\n".join(L)
