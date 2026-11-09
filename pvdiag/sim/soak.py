"""连续运行压测（soak）：把系统放进一个"有人用"的模拟世界里跑半年。

每一天：
  ① 流水线跑当天（完整 11 个智能体）
  ② 模拟复核员：对进入复核队列 ≥delay 天的案例做判断——知道真相（注入故障表），但有 err 概率判错
  ③ 模拟运维主管：草稿工单隔天"一键确认发送"（gated 门控放开后系统自己发）
  ④ 模拟现场：故障到期修复 → 回填已派工单（真实原因/动作）；没派过工单的故障由巡检发现 → 记一条"漏报"工单
  ⑤ 记录当日指标：嫌疑/在办/误报率/门控状态/运行口径/模型版本/告警数

验证的是长周期行为，不是单日正确性：
  - 自调参会不会动、动了会不会回滚、会不会来回翻（振荡）
  - 模型标注够了会不会训练、考试过不过、上线后会不会被降级
  - 门控会不会在误报率达标后放开、在异常时收回
  - 自愈/告警不会刷屏（去重），流水线连续 N 天不中断
随机故障：每串每天以小概率随机注入一种故障（持续 10~40 天后修复），另有少量永久性配置差异。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..agents.base import AgentContext
from ..agents.ingest import IngestAgent
from ..agents.profile import ProfileAgent
from ..agents.review import ReviewAgent
from ..agents.workorder import WorkOrderAgent
from ..config import Settings
from ..models import Cause
from ..orchestrator import DailyPipeline
from ..storage import Store
from .synth import Fault, SimConfig, simulate

KIND_CAUSE = {"open_circuit": Cause.OPEN_CIRCUIT, "diode_bypass": Cause.BYPASS_DIODE, "shading_fixed": Cause.SHADING,
              "orientation": Cause.ORIENTATION, "dust_local": Cause.DEGRADATION_DUST, "stuck": Cause.DATA_ISSUE,
              "wrong_profile": Cause.PROFILE_ERROR, "degradation_step": Cause.DEGRADATION_DUST,
              "outage_intermittent": Cause.OUTAGE, "stale_partial": Cause.DATA_ISSUE, "mppt_fault": Cause.MPPT}
KIND_ACTION = {"open_circuit": "维修", "diode_bypass": "更换", "shading_fixed": "维修", "dust_local": "清洗", "stuck": "维修",
               "degradation_step": "清洗", "outage_intermittent": "维修", "stale_partial": "维修", "mppt_fault": "维修"}
RANDOM_KINDS = ["open_circuit", "diode_bypass", "shading_fixed", "dust_local", "stuck", "outage_intermittent", "stale_partial", "degradation_step"]


@dataclass
class SoakConfig:
    days: int = 180
    n_stations: int = 2
    n_inverters: int = 3
    strings_per_inverter: int = 12
    start: date = date(2026, 1, 5)
    seed: int = 1
    fault_rate: float = 0.004           # 每串每天新故障概率（36 串 → 约 15~20 个随机故障 / 180 天 / 站）
    fault_days: Tuple[int, int] = (10, 40)
    review_delay_days: Tuple[int, int] = (1, 3)
    reviewer_error: float = 0.08        # 复核员判错概率
    reviews_per_day: int = 12
    dispatch_delay_days: int = 1        # 主管隔天确认草稿
    report_every_days: int = 30
    settings_overrides: Dict = field(default_factory=dict)   # 例如 {"confidence": {"min_labels_for_ml": 40}}


def _random_faults(rng: random.Random, cfg: SoakConfig, station_id: str) -> List[Fault]:
    inv = lambda k: f"{station_id}-INV{k:02d}"
    strings = [f"{inv(k)}-pv{n}" for k in range(1, cfg.n_inverters + 1) for n in range(1, cfg.strings_per_inverter + 1)]
    faults: List[Fault] = [Fault("string", strings[1], "orientation", 0, {"clear_factor": 0.94}),
                           Fault("string", strings[-1], "archived_shading", 0, {"start": "09:30", "end": "11:00", "depth": 0.3})]
    busy_until: Dict[str, int] = {strings[1]: 10 ** 6, strings[-1]: 10 ** 6}
    for day in range(30, cfg.days - 12):
        for sid in strings:
            if busy_until.get(sid, -1) >= day or rng.random() >= cfg.fault_rate:
                continue
            kind = rng.choice(RANDOM_KINDS)
            dur = rng.randint(*cfg.fault_days)
            params = {"open_circuit": {}, "diode_bypass": {"start": "10:00", "end": "13:00", "depth": 0.33},
                      "shading_fixed": {"start": "14:30", "end": "17:30", "depth": 0.5},
                      "dust_local": {"rate_per_day": 0.006, "floor": 0.80, "rain_recover": 0.5}, "stuck": {"value": 8.0},
                      "outage_intermittent": {"prob": 0.6, "hours": 2.0}, "stale_partial": {"start": "10:30", "hours": 3.0},
                      "degradation_step": {"factor": 0.93}}[kind]
            faults.append(Fault("string", sid, kind, day, params, end_day=min(day + dur, cfg.days)))
            busy_until[sid] = day + dur + 15          # 修复后留 15 天空窗
    return faults


class SoakRunner:
    def __init__(self, cfg: SoakConfig, workdir: Path, settings: Optional[Settings] = None):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.settings = (settings or Settings()).model_copy(deep=True)
        self.settings.data_dir = workdir / "data"
        for k, v in cfg.settings_overrides.items():
            sub = getattr(self.settings, k)
            for kk, vv in v.items():
                setattr(sub, kk, vv)
        self.settings.ensure_dirs()
        self.store = Store(self.settings.db_path, self.settings.samples_dir)
        self.truth: pd.DataFrame = pd.DataFrame()
        self.daily: List[Dict] = []
        self.events: List[Dict] = []

    # ------------------------------------------------------------------ 建世界
    def build_world(self) -> None:
        truths = []
        for k in range(self.cfg.n_stations):
            sid = f"SK{k + 1:02d}"
            faults = _random_faults(self.rng, self.cfg, sid)
            sim = SimConfig(station_id=sid, station_name=f"压测电站{k + 1:02d}", n_inverters=self.cfg.n_inverters,
                            strings_per_inverter=self.cfg.strings_per_inverter, days=self.cfg.days, seed=self.cfg.seed * 100 + k,
                            start=self.cfg.start, faults=faults)
            data = simulate(sim)
            ProfileAgent.import_stations(self.store, data["stations"]); ProfileAgent.import_inverters(self.store, data["inverters"])
            ProfileAgent.import_strings(self.store, data["strings"])
            ctx = AgentContext(settings=self.settings, store=self.store, run_date=self.cfg.start + timedelta(days=self.cfg.days - 1))
            IngestAgent().ingest_frames(ctx, sid, data["samples_strings"], data["samples_inverters"])
            truths.append(data["truth"])
        self.truth = pd.concat(truths, ignore_index=True)
        self.truth["end_date"] = self.truth["end_date"].fillna((self.cfg.start + timedelta(days=self.cfg.days + 1)).isoformat())

    # ------------------------------------------------------------------ 真相查询
    def active_fault(self, target: str, d0: str, d1: str) -> Optional[Dict]:
        """target 在 [d0, d1] 内是否有注入故障（含永久性配置差异）"""
        t = self.truth[(self.truth["target"] == target) & (self.truth["start_date"] <= d1) & (self.truth["end_date"] >= d0)]
        return t.iloc[0].to_dict() if not t.empty else None

    # ------------------------------------------------------------------ 模拟的人
    def reviewer(self, d: date) -> int:
        today = d.isoformat()
        q = self.store.query("SELECT suspect_id, string_id, inverter_id, scope, grade, rule_cause, first_date, date FROM suspects "
                             "WHERE status IN ('open','reviewing') AND grade NOT IN ('已备案') ORDER BY CASE grade WHEN 'A' THEN 0 WHEN 'B' THEN 1 ELSE 2 END, first_date")
        n = 0
        for s in q.to_dict("records"):
            delay = (d - date.fromisoformat(s["first_date"] or s["date"])).days     # suspects.date 每天刷新，不能用
            if delay < self.rng.randint(*self.cfg.review_delay_days):
                continue
            if s["grade"] in ("D", "E") and self.rng.random() < 0.7:
                continue                    # 观察级大多不复核
            target = s["string_id"] if s["scope"] == "string" else s["inverter_id"]
            tr = self.active_fault(target, s["first_date"], today)
            truly_bad = tr is not None and tr["kind"] in KIND_CAUSE
            wrong = self.rng.random() < self.cfg.reviewer_error
            if truly_bad != wrong:          # 真坏且没判错 / 不坏但判错 → 确认
                cause = KIND_CAUSE[tr["kind"]].value if truly_bad else None
                ReviewAgent.confirm(self.store, s["suspect_id"], cause, "soak-reviewer", "")
            else:
                ReviewAgent.reject(self.store, s["suspect_id"], self.rng.choice(["云影", "数据脏", "其他"]), "soak-reviewer", "")
            n += 1
            if n >= self.cfg.reviews_per_day:
                break
        return n

    def dispatcher(self, ctx: AgentContext, d: date) -> int:
        """主管隔天确认草稿。工单号里带生成日（WO-YYYYMMDD-xxx），用它而不是 created_at（墙上时钟）判「隔天」"""
        cut = (d - timedelta(days=self.cfg.dispatch_delay_days)).strftime("%Y%m%d")
        drafts = self.store.query("SELECT wo_id FROM workorders WHERE status='draft' AND wo_id LIKE 'WO-2%'")
        n = 0
        for wo in drafts["wo_id"]:
            if wo.split("-")[1] <= cut:
                WorkOrderAgent().send(ctx, wo)
                n += 1
        return n

    def field_crew(self, d: date) -> Tuple[int, int]:
        """今天到期修复的故障：有已派工单 → 回填；没有 → 巡检发现（漏报）"""
        today = d.isoformat()
        due = self.truth[(self.truth["end_date"] == today) & (self.truth["scope"] == "string")]
        closed = missed = 0
        for t in due.to_dict("records"):
            wos = self.store.query("SELECT wo_id, string_ids_json FROM workorders WHERE status IN ('sent','draft','reopened')")
            hit = [w for w in wos.to_dict("records") if t["target"] in json.loads(w["string_ids_json"])]
            cause = KIND_CAUSE[t["kind"]].value
            if hit:
                WorkOrderAgent.close(self.store, hit[0]["wo_id"], cause, KIND_ACTION.get(t["kind"], "维修"), "soak 现场回填",
                                     done_date=d, recheck_clear_days=self.settings.dispatch.recheck_clear_days)
                closed += 1
            else:
                # 系统报过（有嫌疑）但还没派单 → 不算漏报；完全没报过 → 漏报
                seen = self.store.scalar("SELECT COUNT(*) FROM suspect_daily sd JOIN suspects s ON s.suspect_id=sd.suspect_id "
                                         "WHERE s.string_id=? AND sd.date BETWEEN ? AND ?", (t["target"], t["start_date"], today)) or 0
                note = "巡检发现，系统漏报" if not seen else "巡检先行修复（系统已报、未及派单）"
                if seen:
                    ProfileAgent.add_event(self.store, "string", t["target"], "repair", d, "巡检修复")
                    for sid in self.store.query("SELECT suspect_id FROM suspects WHERE string_id=? AND status IN ('open','reviewing','confirmed')", (t["target"],))["suspect_id"]:
                        ReviewAgent.confirm(self.store, sid, cause, "field", "巡检修复确认")
                    continue
                wo_id = f"WO-FIELD-{d:%Y%m%d}-{self.rng.randrange(16 ** 4):04X}"
                self.store.execute(
                    "INSERT INTO workorders (wo_id, station_id, roof_area, suspect_ids_json, string_ids_json, cause, symptom, checklist_json, "
                    "loss_kwh_per_day, loss_kwh_total, loss_yuan_total, priority, status, channel, created_at, closed_at, field_cause, field_action, field_note) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (wo_id, t["target"].split("-")[0], "", "[]", json.dumps([t["target"]]), cause, "巡检发现", "[]", 0, 0, 0, 0, "done", "field",
                     datetime.combine(d, datetime.min.time()).isoformat(), datetime.combine(d, datetime.min.time()).isoformat(), cause,
                     KIND_ACTION.get(t["kind"], "维修"), note))
                ProfileAgent.add_event(self.store, "string", t["target"], "repair", d, f"{wo_id} 巡检修复")
                missed += 1
        return closed, missed

    # ------------------------------------------------------------------ 主循环
    def run(self, progress=None) -> Dict:
        if self.truth.empty:
            self.build_world()
        pipe = DailyPipeline(self.settings, store=self.store, ingest=False)
        end = self.cfg.start + timedelta(days=self.cfg.days - 1)
        d = self.cfg.start
        k = 0
        while d <= end:
            k += 1
            write = (k % self.cfg.report_every_days == 0) or d == end
            rep = pipe.run(d, write_report=write)
            ctx = AgentContext(settings=pipe.settings, store=self.store, run_date=d)
            n_rev = self.reviewer(d)
            n_sent = self.dispatcher(ctx, d)
            n_closed, n_missed = self.field_crew(d)
            self.daily.append(self._snapshot(d, rep, n_rev, n_sent, n_closed, n_missed))
            if progress and k % 10 == 0:
                progress(d, self.daily[-1])
            d += timedelta(days=1)
        return self.summary()

    def _snapshot(self, d: date, rep: Dict, n_rev: int, n_sent: int, n_closed: int, n_missed: int) -> Dict:
        board = rep.get("board", {})
        kpi = board.get("kpi", {})
        gate = board.get("auto_dispatch_gate", {})
        ov = self.store.query("SELECT param, value_json FROM param_overrides")
        failed = [k for k, v in rep.get("agents", {}).items() if v["status"] == "fail"]
        return {
            "date": d.isoformat(), "suspects": len(rep.get("suspects", [])),
            "open_cases": kpi.get("open_cases"), "false_alarm_rate": kpi.get("false_alarm_rate"), "reviewed_30d": kpi.get("reviewed_30d"),
            "field_missed": kpi.get("field_missed"), "gate_effective": gate.get("effective"), "gate_streak": (board.get("autonomy", {}).get("gate") or {}).get("streak"),
            "model": (board.get("model") or {}).get("active"), "n_labels": (board.get("model") or {}).get("n_labels"),
            "overrides": {r.param.split(".")[-1]: json.loads(r.value_json) for r in ov.itertuples()},
            "alerts": len(rep.get("alerts", [])), "agents_failed": failed, "reviews": n_rev, "sent": n_sent, "field_closed": n_closed, "field_missed_new": n_missed,
            "tuning": (rep.get("tuning") or {}).get("decision"), "retrain": (rep.get("retrain") or {}).get("reason") or ("trained" if (rep.get("retrain") or {}).get("trained") else None),
        }

    # ------------------------------------------------------------------ 评分
    def summary(self) -> Dict:
        df = pd.DataFrame(self.daily)
        susp = self.store.query("SELECT suspect_id, string_id, inverter_id, scope, grade, rule_cause, first_date, date, status FROM suspects")
        snaps = self.store.query("SELECT suspect_id, MIN(date) first_seen, MAX(CASE WHEN grade IN ('A','B') THEN 1 ELSE 0 END) ab FROM suspect_daily GROUP BY suspect_id")
        susp = susp.merge(snaps, on="suspect_id", how="left")
        # 召回：每个随机故障（非永久配置差异、非已备案）在其持续期内是否出现 A/B 级嫌疑
        faults = self.truth[~self.truth["kind"].isin(["orientation", "archived_shading", "wrong_profile"])]
        hit, lat = 0, []
        for t in faults.to_dict("records"):
            key = t["target"]
            m = susp[(susp["string_id"] == key) & (susp["ab"] == 1) & (susp["first_date"] <= t["end_date"]) & (susp["date"] >= t["start_date"])]
            if not m.empty:
                hit += 1
                # 发现时延 = 首次出现在嫌疑表的日期 − 故障注入日（first_date 是"推定起始日"，会早于发现日，不能用）
                lat.append((date.fromisoformat(str(m["first_seen"].min())) - date.fromisoformat(t["start_date"])).days)
        healthy = set(self.store.strings()["string_id"]) - set(self.truth["target"])
        fa_strings = susp[(susp["scope"] == "string") & susp["string_id"].isin(healthy) & (susp["ab"] == 1)]["string_id"].nunique()
        changes = self.store.query("SELECT run_date, param, old_value, new_value, status FROM param_changes ORDER BY change_id")
        applied = changes[changes["status"].isin(["applied", "rolled_back"])]
        flips = {p: int(len(g)) for p, g in applied.groupby("param")} if not applied.empty else {}
        models = self.store.query("SELECT model_version, trained_at, n_labels, status FROM model_registry ORDER BY trained_at")
        gate_hist = (self.store.get_state("dispatch:gate") or {}).get("history", [])
        rv = ReviewAgent.stats(self.store, days=10 ** 4)
        pushes = int(self.store.scalar("SELECT COUNT(*) FROM alerts WHERE source='notify'") or 0)
        return {
            "days": int(len(df)), "stations": self.cfg.n_stations, "strings": int(len(self.store.strings())),
            "faults_injected": int(len(faults)), "faults_recalled_ab": hit, "recall": round(hit / max(1, len(faults)), 3),
            "detection_latency_days_median": float(pd.Series(lat).median()) if lat else None,
            "healthy_strings": len(healthy), "healthy_false_alarm_ab_strings": int(fa_strings),
            "reviews_total": rv.get("n_reviewed", 0), "false_alarm_rate_overall": rv.get("false_alarm_rate"), "field_missed": rv.get("field_missed", 0),
            "agent_failures_days": int((df["agents_failed"].map(len) > 0).sum()),
            "autotune": {"applied": int((changes["status"] == "applied").sum()) if not changes.empty else 0,
                         "rolled_back": int((changes["status"] == "rolled_back").sum()) if not changes.empty else 0,
                         "proposed": int((changes["status"] == "proposed").sum()) if not changes.empty else 0,
                         "changes_per_param": flips, "final_overrides": df["overrides"].iloc[-1] if len(df) else {},
                         "oscillation": any(v > 3 for v in flips.values())},
            "models": models.to_dict("records"), "model_promoted": bool((models["status"] == "active").any()) if not models.empty else False,
            "gate": {"days_auto": int((df["gate_effective"] == "auto").sum()), "transitions": gate_hist,
                     "first_auto": df.loc[df["gate_effective"] == "auto", "date"].min() if (df["gate_effective"] == "auto").any() else None},
            "pushes": pushes,
            "workorders": self.store.query("SELECT status, COUNT(*) n FROM workorders GROUP BY status").set_index("status")["n"].to_dict(),
            "daily": self.daily,
        }


def render_markdown(s: Dict) -> str:
    L = [f"# 连续运行压测报告：{s['days']} 天 · {s['stations']} 站 · {s['strings']} 串", ""]
    L += [f"- 注入随机故障 {s['faults_injected']} 个，A/B 级召回 **{s['faults_recalled_ab']}/{s['faults_injected']}**（{s['recall']:.0%}），发现时延中位 {s['detection_latency_days_median']} 天",
          f"- 健康串 {s['healthy_strings']} 路，出现过 A/B 误报的 **{s['healthy_false_alarm_ab_strings']}** 路；复核 {s['reviews_total']} 次，整体误报率 {s['false_alarm_rate_overall']}，现场漏报 {s['field_missed']} 起",
          f"- 智能体失败天数 {s['agent_failures_days']}；关键事件推送 {s['pushes']} 次（去重后）",
          f"- 自调参：生效 {s['autotune']['applied']} 次、回滚 {s['autotune']['rolled_back']} 次、提议 {s['autotune']['proposed']} 次；每参数变更次数 {s['autotune']['changes_per_param']}；"
          f"最终覆盖层 {s['autotune']['final_overrides']}；振荡：{'是' if s['autotune']['oscillation'] else '否'}",
          f"- 模型：{'已上线' if s['model_promoted'] else '未上线'}；注册表 {[(m['model_version'], m['status'], m['n_labels']) for m in s['models']]}",
          f"- 派单门控：auto 共 {s['gate']['days_auto']} 天，首次放开 {s['gate']['first_auto']}；开合记录 {len(s['gate']['transitions'])} 次",
          f"- 工单状态分布：{s['workorders']}", ""]
    for t in s["gate"]["transitions"]:
        L.append(f"  - {t['date']} {t['event']}")
    L += ["", "## 逐日指标（每 10 天）", "", "| 日期 | 嫌疑 | 在办 | 误报率 | 复核30d | 漏报 | 门控 | 模型 | 标注 | 覆盖层 | 告警 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["daily"][::10]:
        L.append(f"| {r['date']} | {r['suspects']} | {r['open_cases']} | {r['false_alarm_rate']} | {r['reviewed_30d']} | {r['field_missed']} | {r['gate_effective']} | "
                 f"{r['model']} | {r['n_labels']} | {r['overrides']} | {r['alerts']} |")
    return "\n".join(L)


def write_report(summary: Dict, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "soak_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    (out / "soak_report.md").write_text(render_markdown(summary), encoding="utf-8")
    return out / "soak_report.md"
