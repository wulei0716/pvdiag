"""阶段4·块二 · 阈值自调参闭环（TuningAgent）

以前：ReviewAgent 只在日报里"建议把阈值收紧到 -6%"，要人改 config.yaml。
现在：把人工复核结果当"考题"，让候选阈值在历史上重新"考一遍"（影子回放），考得更好才生效，生效后持续观察，变差自动回滚。

口径：
- 考题 = 窗口内已复核的"纯偏差型"案例（B/D 级、无凹陷/停机/钉死/电压异位签名）：确认/修改 → 正例，驳回 → 负例
- 影子回放 = 用候选 (fault_floor, fault_persist_days, noise_k) 在该串的历史晴好天偏差序列上重新判"会不会报 B"
  → 精确率 / 召回 / F1；候选必须：F1 提升 ≥ min_f1_gain 且 召回 ≥ 当前召回 × min_recall_keep
- 只能在安全区间内走一步（fault_floor ±1pct、persist +1、noise_k +0.5），两次调整间隔 ≥ cooldown_days
- 生效方式：写 param_overrides（运行时覆盖层），config.yaml 不动；全部变更进 param_changes 留痕；`pvdiag tune` 可查看/回滚
- 放松（阈值 -5% → -4%）无法用已有标注验证（没报过的串没有标注）→ 现场漏报 ≥N 起时只"提议"，不自动生效
- 回滚：生效 ≥3 天且新增 ≥10 条复核后，B 级确认率比生效前跌 ≥ rollback_confirm_drop → 自动回滚并推送

mode=shadow 时只评估、只记录建议，不改任何参数。
"""
from __future__ import annotations

import itertools
import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import Settings
from ..models import Cause, Grade, Signature
from .base import Agent, AgentContext, AgentResult
from .baseline import BaselineAgent

TUNABLE = {
    "thresholds.fault_floor": float,
    "thresholds.fault_persist_days": int,
    "thresholds.noise_k": float,
}


# ---------------------------------------------------------------------------- 覆盖层
def apply_overrides(settings: Settings, store) -> Settings:
    """把 param_overrides 表叠加到 Settings 副本上（config.yaml 保持人工口径，运行时口径 = yaml + 自调参覆盖）"""
    try:
        rows = store.query("SELECT param, value_json FROM param_overrides")
    except Exception:
        return settings
    if rows.empty:
        return settings
    s = settings.model_copy(deep=True)
    for r in rows.itertuples():
        try:
            set_param(s, r.param, json.loads(r.value_json))
        except Exception:
            continue
    return s


def get_param(settings: Settings, path: str) -> Any:
    obj: Any = settings
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def set_param(settings: Settings, path: str, value: Any) -> None:
    parts = path.split(".")
    obj: Any = settings
    for part in parts[:-1]:
        obj = getattr(obj, part)
    caster = TUNABLE.get(path)
    setattr(obj, parts[-1], caster(value) if caster else value)


class TuningAgent(Agent):
    name = "tuning"
    stage = "阶段4·块二 阈值自调参"

    # ------------------------------------------------------------------ 入口
    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        cfg = ctx.settings.autotune
        if not cfg.enabled:
            res.status, res.summary = "skipped", "自调参未启用"
            return res
        out: Dict[str, Any] = {"mode": cfg.mode, "current": self.current_params(ctx.settings)}
        # ① 回滚检查（先于新调整）
        rb = self.check_rollback(ctx)
        out["rollback"] = rb
        if rb.get("rolled_back"):
            res.alerts.append(f"参数 {rb['param']} 已自动回滚到 {rb['old_value']}：{rb['reason']}")
        # ② 考题
        cases = self.labeled_cases(ctx, cfg.window_days)
        out["n_cases"] = len(cases)
        out["n_positive"] = int(sum(c["label"] for c in cases))
        if len(cases) < cfg.min_labels or out["n_positive"] == 0 or out["n_positive"] == len(cases):
            out["decision"] = f"复核样本不足（纯偏差型 {len(cases)} 条，正例 {out['n_positive']}），不调参"
            res.summary = out["decision"]
            res.artifacts = out
            ctx.shared["tuning_state"] = out
            return res
        # ③ 当前 vs 候选
        cur = out["current"]
        table = self.evaluate_grid(ctx, cases, cur)
        out["evaluation"] = table
        cur_row = next(r for r in table if r["is_current"])
        best = self.pick(table, cur_row, cfg, ctx)
        out["best"] = best
        # ④ 决定
        last = ctx.store.get_state("tuning:last_change") or {}
        in_cooldown = bool(last.get("at")) and (ctx.run_date - date.fromisoformat(last["at"][:10])).days < cfg.cooldown_days
        if best is None:
            out["decision"] = f"当前参数已是候选中最优（F1 {cur_row['f1']:.2f}，精确率 {cur_row['precision']:.2f}，召回 {cur_row['recall']:.2f}）"
        elif in_cooldown:
            out["decision"] = f"有更优候选 {best['params']}（F1 {cur_row['f1']:.2f}→{best['f1']:.2f}），冷却期内（上次 {last.get('at', '')[:10]}）暂不生效"
            self.log_change(ctx, best, cur_row, status="deferred")
        elif cfg.mode != "apply":
            out["decision"] = f"shadow 模式：建议 {best['params']}（F1 {cur_row['f1']:.2f}→{best['f1']:.2f}），未生效"
            self.log_change(ctx, best, cur_row, status="proposed")
            res.alerts.append("阈值调优建议（shadow）：" + out["decision"])
        else:
            applied = self.apply(ctx, best, cur_row)
            out["decision"] = f"已生效：{applied}（F1 {cur_row['f1']:.2f}→{best['f1']:.2f}，精确率 {cur_row['precision']:.2f}→{best['precision']:.2f}，召回 {cur_row['recall']:.2f}→{best['recall']:.2f}）"
            res.alerts.append("阈值自调参生效：" + out["decision"])
        # ⑤ 放松提议（现场漏报）
        loosen = self.loosen_proposal(ctx, cfg, cur)
        if loosen:
            out["loosen_proposal"] = loosen
            res.alerts.append(loosen["text"])
        res.metrics = {"n_cases": len(cases), "f1_current": cur_row["f1"], "f1_best": (best or cur_row)["f1"]}
        res.summary = out["decision"]
        res.artifacts = out
        ctx.shared["tuning_state"] = out
        return res

    # ------------------------------------------------------------------ 考题
    @staticmethod
    def current_params(settings: Settings) -> Dict[str, Any]:
        return {k: get_param(settings, k) for k in TUNABLE}

    @staticmethod
    def labeled_cases(ctx: AgentContext, window_days: int) -> List[Dict]:
        since = (ctx.run_date - timedelta(days=window_days)).isoformat()
        rv = ctx.store.query(
            "SELECT r.suspect_id, r.decision, r.true_cause, r.reject_reason, r.reviewed_at, s.string_id, s.first_date, s.grade, s.signature_json "
            "FROM reviews r JOIN suspects s ON s.suspect_id=r.suspect_id "
            "WHERE r.decision IN ('confirm','modify','reject') AND r.reviewed_at>=? AND s.scope='string' ORDER BY r.reviewed_at", (since,))
        if rv.empty:
            return []
        rv = rv.drop_duplicates(subset=["suspect_id"], keep="last")
        out = []
        for r in rv.to_dict("records"):
            sig = Signature.model_validate_json(r["signature_json"]) if r.get("signature_json") else Signature()
            if sig.dip_depth is not None or sig.outage_hours or sig.zero_all_day or sig.stuck or sig.voltage_offset is not None or sig.inverter_level:
                continue                      # 只考"纯偏差型"
            if r["grade"] not in (Grade.B.value, Grade.D.value):
                continue
            label = 0 if r["decision"] == "reject" or r.get("true_cause") == Cause.NORMAL.value else 1
            start = (date.fromisoformat(r["first_date"]) - timedelta(days=30)).isoformat() if r.get("first_date") else since
            end = str(r["reviewed_at"])[:10]
            b = ctx.store.query(
                "SELECT b.date, b.dev_today, b.group_noise, f.noon_zero_frac, f.zero_all_day FROM baselines b "
                "JOIN daily_features f ON f.string_id=b.string_id AND f.date=b.date "
                "WHERE b.string_id=? AND b.weather_tag='clear' AND b.dev_today IS NOT NULL AND b.date BETWEEN ? AND ? ORDER BY b.date",
                (r["string_id"], start, end))
            if b.empty:
                continue
            noises = b["group_noise"].fillna(0).to_numpy(dtype=float)
            skip = (b["noon_zero_frac"].fillna(0).to_numpy(dtype=float) >= ctx.settings.thresholds.outage_noon_skip_frac) & (b["zero_all_day"].to_numpy() == 0)
            noises = np.where(skip, np.inf, noises)
            out.append({"suspect_id": r["suspect_id"], "string_id": r["string_id"], "label": label,
                        "reject_reason": r.get("reject_reason"), "devs": b["dev_today"].to_numpy(dtype=float), "noises": noises})
        return out

    # ------------------------------------------------------------------ 影子回放
    @staticmethod
    def would_alarm(devs: np.ndarray, noises: np.ndarray, floor: float, persist: int, k: float, skip: float) -> bool:
        for i in range(len(devs)):
            n, _ = BaselineAgent._streak_gated(devs[: i + 1], noises[: i + 1], floor, k, skip)
            if n >= persist:
                return True
        return False

    def evaluate(self, cases: List[Dict], floor: float, persist: int, k: float, skip: float) -> Dict[str, float]:
        tp = fp = fn = tn = 0
        for c in cases:
            a = self.would_alarm(c["devs"], c["noises"], floor, persist, k, skip)
            if a and c["label"]:
                tp += 1
            elif a and not c["label"]:
                fp += 1
            elif (not a) and c["label"]:
                fn += 1
            else:
                tn += 1
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}

    def evaluate_grid(self, ctx: AgentContext, cases: List[Dict], cur: Dict[str, Any]) -> List[Dict]:
        cfg, th = ctx.settings.autotune, ctx.settings.thresholds
        f0, p0, k0 = cur["thresholds.fault_floor"], cur["thresholds.fault_persist_days"], cur["thresholds.noise_k"]
        floors = sorted({round(x, 3) for x in (f0 - cfg.step_floor, f0, f0 + cfg.step_floor) if cfg.fault_floor_bounds[0] - 1e-9 <= x <= cfg.fault_floor_bounds[1] + 1e-9} | {round(f0, 3)})
        persists = sorted({p for p in (p0, p0 + 1) if cfg.persist_bounds[0] <= p <= cfg.persist_bounds[1]} | {p0})
        ks = sorted({round(x, 2) for x in (k0, k0 + 0.5) if cfg.noise_k_bounds[0] - 1e-9 <= x <= cfg.noise_k_bounds[1] + 1e-9} | {k0})
        table = []
        for f, p, k in itertools.product(floors, persists, ks):
            m = self.evaluate(cases, f, p, k, th.noise_skip)
            m["params"] = {"thresholds.fault_floor": f, "thresholds.fault_persist_days": p, "thresholds.noise_k": k}
            m["is_current"] = (abs(f - f0) < 1e-9 and p == p0 and abs(k - k0) < 1e-9)
            m["n_changes"] = int(abs(f - f0) > 1e-9) + int(p != p0) + int(abs(k - k0) > 1e-9)
            table.append(m)
        return table

    @staticmethod
    def pick(table: List[Dict], cur: Dict, cfg, ctx: Optional[AgentContext] = None) -> Optional[Dict]:
        """只走一步（只改一个参数）、F1 提升够大、召回不塌、且不是近 90 天内被回滚过的值 → 取 F1 最高者"""
        cands = [r for r in table if r["n_changes"] == 1 and r["f1"] >= cur["f1"] + cfg.min_f1_gain
                 and r["recall"] >= cur["recall"] * cfg.min_recall_keep]
        if ctx is not None:
            cands = [r for r in cands if not TuningAgent._recently_rolled_back(ctx, r, cur)]
        if not cands:
            return None
        return max(cands, key=lambda r: (r["f1"], r["precision"]))

    @staticmethod
    def _recently_rolled_back(ctx: AgentContext, cand: Dict, cur: Dict, days: int = 90) -> bool:
        """影子回放看 F1，回滚看确认率——两个口径可能打架，被回滚过的值 90 天内不再自动生效（防止来回翻）"""
        for k, v in cand["params"].items():
            if cur["params"][k] != v and TuningAgent._recent_logged(ctx, k, v, "rolled_back", days):
                return True
        return False

    # ------------------------------------------------------------------ 生效 / 留痕 / 回滚
    @staticmethod
    def _recent_logged(ctx: AgentContext, param: str, new_value: Any, status: str, days: int) -> bool:
        n = ctx.store.scalar("SELECT COUNT(*) FROM param_changes WHERE param=? AND new_value=? AND status=? AND run_date>=?",
                             (param, json.dumps(new_value), status, (ctx.run_date - timedelta(days=days)).isoformat())) or 0
        return n > 0

    def log_change(self, ctx: AgentContext, best: Dict, cur: Dict, status: str) -> None:
        for k, v in best["params"].items():
            if cur["params"][k] != v and not self._recent_logged(ctx, k, v, status, ctx.settings.autotune.cooldown_days):
                ctx.store.execute(
                    "INSERT INTO param_changes (run_date, param, old_value, new_value, reason, metrics_json, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                    (ctx.run_date.isoformat(), k, json.dumps(cur["params"][k]), json.dumps(v),
                     f"影子回放 F1 {cur['f1']:.2f}→{best['f1']:.2f}", json.dumps({"before": cur, "after": best}, ensure_ascii=False), status,
                     datetime.now().isoformat()))

    def apply(self, ctx: AgentContext, best: Dict, cur: Dict) -> str:
        from .review import ReviewAgent
        st = ReviewAgent.stats(ctx.store, days=ctx.settings.autotune.window_days)
        before_b = (st.get("confirm_rate_by_grade") or {}).get(Grade.B.value)
        changed = []
        for k, v in best["params"].items():
            if cur["params"][k] == v:
                continue
            ctx.store.execute("INSERT OR REPLACE INTO param_overrides (param, value_json, updated_at, source) VALUES (?,?,?,?)",
                              (k, json.dumps(v), datetime.now().isoformat(), "autotune"))
            ctx.store.execute(
                "INSERT INTO param_changes (run_date, param, old_value, new_value, reason, metrics_json, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (ctx.run_date.isoformat(), k, json.dumps(cur["params"][k]), json.dumps(v),
                 f"影子回放 F1 {cur['f1']:.2f}→{best['f1']:.2f}",
                 json.dumps({"before": cur, "after": best, "before_confirm_rate_b": before_b,
                             "n_reviews_before": st.get("n_reviewed", 0)}, ensure_ascii=False), "applied", datetime.now().isoformat()))
            set_param(ctx.settings, k, v)           # 当天后续智能体立即用新口径
            changed.append(f"{k} {cur['params'][k]}→{v}")
        ctx.store.set_state("tuning:last_change", {"at": ctx.run_date.isoformat(), "params": best["params"]})
        return "；".join(changed)

    def check_rollback(self, ctx: AgentContext) -> Dict:
        cfg = ctx.settings.autotune
        ch = ctx.store.query("SELECT * FROM param_changes WHERE status='applied' ORDER BY change_id DESC LIMIT 1")
        if ch.empty:
            return {"rolled_back": False, "reason": "无生效中的自动调整"}
        c = ch.iloc[0]
        applied_on = date.fromisoformat(c["run_date"])
        if (ctx.run_date - applied_on).days < 3:
            return {"rolled_back": False, "reason": f"{c['param']} 生效不足 3 天，观察中"}
        m = json.loads(c["metrics_json"] or "{}")
        before_b = m.get("before_confirm_rate_b")
        rv = ctx.store.query(
            "SELECT r.decision FROM reviews r JOIN suspects s ON s.suspect_id=r.suspect_id "
            "WHERE r.decision IN ('confirm','modify','reject') AND r.reviewed_at>=? AND s.grade=?", (c["created_at"], Grade.B.value))
        if len(rv) < 10 or before_b is None:
            return {"rolled_back": False, "reason": f"{c['param']} 生效后 B 级新复核 {len(rv)} 条，样本不足"}
        after_b = float((rv["decision"] != "reject").mean())
        if after_b < before_b - cfg.rollback_confirm_drop:
            old = json.loads(c["old_value"])
            ctx.store.execute("INSERT OR REPLACE INTO param_overrides (param, value_json, updated_at, source) VALUES (?,?,?,?)",
                              (c["param"], json.dumps(old), datetime.now().isoformat(), "rollback"))
            ctx.store.execute("UPDATE param_changes SET status='rolled_back', rolled_back_at=? WHERE change_id=?",
                              (datetime.now().isoformat(), int(c["change_id"])))
            set_param(ctx.settings, c["param"], old)
            ctx.store.set_state("tuning:last_change", {"at": ctx.run_date.isoformat(), "rollback": c["param"]})
            return {"rolled_back": True, "param": c["param"], "old_value": old,
                    "reason": f"B 级确认率 {before_b:.0%}→{after_b:.0%}（n={len(rv)}）"}
        return {"rolled_back": False, "reason": f"{c['param']} 生效后 B 级确认率 {after_b:.0%}（生效前 {before_b:.0%}），保持"}

    @staticmethod
    def loosen_proposal(ctx: AgentContext, cfg, cur: Dict[str, Any]) -> Optional[Dict]:
        from .review import ReviewAgent
        st = ReviewAgent.stats(ctx.store, days=cfg.window_days)
        missed = int(st.get("field_missed", 0) or 0)
        if missed < cfg.loosen_on_field_missed:
            return None
        f0 = cur["thresholds.fault_floor"]
        f1 = round(min(f0 + cfg.step_floor, cfg.fault_floor_bounds[1]), 3)
        if f1 == f0:
            return None
        # 预估新增报警：窗口内晴好天偏差落在 (f1, f0] 且连续 ≥persist 天的串数
        since = (ctx.run_date - timedelta(days=cfg.window_days)).isoformat()
        b = ctx.store.query("SELECT string_id, date, dev_today FROM baselines WHERE weather_tag='clear' AND dev_today IS NOT NULL AND date>=? ORDER BY date", (since,))
        extra = 0
        for _, g in b.groupby("string_id"):
            d = g["dev_today"].to_numpy(dtype=float)
            if ((d <= f1) & (d > f0)).sum() >= cur["thresholds.fault_persist_days"]:
                extra += 1
        text = (f"近 {cfg.window_days} 天现场发现 {missed} 起系统未报的故障：提议把持续低发阈值从 {f0:.0%} 放松到 {f1:.0%}"
                f"（预计新增约 {extra} 串进入 B 级）。放松无法用已有标注验证，不自动生效——确认请执行 `pvdiag tune apply thresholds.fault_floor {f1}`")
        if TuningAgent._recent_logged(ctx, "thresholds.fault_floor", f1, "proposed", cfg.cooldown_days):
            return {"param": "thresholds.fault_floor", "from": f0, "to": f1, "estimated_extra_alarms": extra, "text": text}
        ctx.store.execute(
            "INSERT INTO param_changes (run_date, param, old_value, new_value, reason, metrics_json, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
            (ctx.run_date.isoformat(), "thresholds.fault_floor", json.dumps(f0), json.dumps(f1), f"现场漏报 {missed} 起",
             json.dumps({"estimated_extra_alarms": extra}), "proposed", datetime.now().isoformat()))
        return {"param": "thresholds.fault_floor", "from": f0, "to": f1, "estimated_extra_alarms": extra, "text": text}
