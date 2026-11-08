"""阶段5 · AI 模型判定（原因分类 + 置信度）

① 特征工程：偏差时序、曲线签名（全天形状）、天气、配置档案、历史复核结果
② 分类模型：故障类型（断线/二极管/遮挡/朝向/MPPT/衰减积灰/数据问题/正常误报）
   —— 首选可解释树模型（sklearn HistGradientBoosting；装了 lightgbm 可替换），冷启动用"规则代理模型"
③ 置信度分级：高置信 → 直接派单；中置信 → 转人工复核；低置信 → 仅观察
④ 无监督兜底：孤立森林发现"没见过的新模式"，以 E 级入观察表
⑤ 可解释输出：每条判定带"为什么"（偏差趋势 + 签名 + 档案依据）
⑥ 模型漂移：标注分布/特征分布季度重训；标注 < min_labels_for_ml 时不启用 ML
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models import Action, Cause, Grade, Signature, SuspectStatus
from .base import Agent, AgentContext, AgentResult

FEATURES = ["dev_today", "dev_vs_own", "dev_roll7", "persist_days_5", "persist_days_3", "baseline_n", "group_n",
            "zero_all_day", "stuck", "voltage_offset", "dip_depth", "dip_len_h", "dip_center_h", "outside_dip_ratio",
            "constant_offset", "ratio_mean", "ratio_std", "am_pm_skew", "overcast_dev", "known_shading",
            "inverter_level", "low_confidence", "n_clear_recent", "month", "grade_sev",
            "outage_hours", "outage_days", "stale_run_min", "group_noise"]

GRADE_SEV = {Grade.A.value: 0, Grade.B.value: 1, Grade.C.value: 2, Grade.D.value: 3, Grade.E.value: 4, Grade.ARCHIVED.value: 5}


def _hhmm_to_h(s: Optional[str]) -> float:
    if not s:
        return np.nan
    h, m = s.split(":")
    return int(h) + int(m) / 60.0


def build_features(susp: Dict, bl: Optional[pd.Series], sig: Signature, prof: Optional[pd.Series], run_date: date) -> Dict[str, float]:
    g = lambda s, k, d=np.nan: (float(s[k]) if (s is not None and k in s and pd.notna(s[k])) else d)
    dip_len = (_hhmm_to_h(sig.dip_end) - _hhmm_to_h(sig.dip_start)) if sig.dip_depth is not None else 0.0
    dip_center = (_hhmm_to_h(sig.dip_end) + _hhmm_to_h(sig.dip_start)) / 2 if sig.dip_depth is not None else np.nan
    return {
        "dev_today": susp.get("dev_today") if susp.get("dev_today") is not None else g(bl, "dev_today"),
        "dev_vs_own": g(bl, "dev_vs_own"), "dev_roll7": g(bl, "dev_roll7"),
        "persist_days_5": g(bl, "persist_days_5", 0), "persist_days_3": g(bl, "persist_days_3", 0),
        "baseline_n": g(bl, "baseline_n", 0), "group_n": g(bl, "group_n", 0),
        "zero_all_day": float(sig.zero_all_day), "stuck": float(sig.stuck),
        "voltage_offset": sig.voltage_offset if sig.voltage_offset is not None else 0.0,
        "dip_depth": sig.dip_depth if sig.dip_depth is not None else 0.0, "dip_len_h": dip_len, "dip_center_h": dip_center,
        "outside_dip_ratio": sig.outside_dip_ratio if sig.outside_dip_ratio is not None else np.nan,
        "constant_offset": float(sig.constant_offset),
        "ratio_mean": sig.ratio_mean if sig.ratio_mean is not None else np.nan,
        "ratio_std": sig.ratio_std if sig.ratio_std is not None else np.nan,
        "am_pm_skew": np.nan, "overcast_dev": sig.overcast_dev if sig.overcast_dev is not None else np.nan,
        "known_shading": float(bool(prof.get("known_shading"))) if prof is not None else 0.0,
        "inverter_level": float(sig.inverter_level), "low_confidence": float(susp.get("low_confidence", 0) or 0),
        "n_clear_recent": g(bl, "n_clear_recent", 0), "month": float(run_date.month),
        "grade_sev": float(GRADE_SEV.get(susp.get("grade"), 4)),
        "outage_hours": float(sig.outage_hours or 0.0), "outage_days": float(sig.outage_days or 0),
        "stale_run_min": float(sig.stale_run_min or 0), "group_noise": sig.group_noise if sig.group_noise is not None else np.nan,
    }


# ---------------------------------------------------------------------------- 模型
class RuleProxyModel:
    """冷启动代理：原因 = 规则假设；置信度 = 规则强度的经验标定"""
    version = "rule-proxy-v1"

    def predict(self, susp: Dict, feats: Dict[str, float], sig: Signature) -> Tuple[str, float, Dict[str, float]]:
        grade, cause = susp["grade"], susp["rule_cause"]
        p5 = feats.get("persist_days_5", 0) or 0
        persist = susp.get("persist_days", 0) or 0
        dev = feats.get("dev_today")
        if grade == Grade.A.value:
            conf = {Cause.OPEN_CIRCUIT.value: 0.95, Cause.DATA_ISSUE.value: 0.85, Cause.PROFILE_ERROR.value: 0.80}.get(cause, 0.8)
        elif grade == Grade.B.value:
            if sig.inverter_level:
                conf = 0.80 if persist >= 5 else 0.70
            elif sig.outage_hours > 0:
                conf = 0.85 if sig.outage_days >= 3 else 0.70
            elif sig.dip_depth is not None:
                conf = 0.85 if persist >= 3 else 0.75
            elif sig.constant_offset:
                conf = 0.75 if (p5 >= 5 and dev is not None and dev <= -0.08) else 0.60
            else:
                conf = 0.55
        elif grade == Grade.C.value:
            conf = 0.70
        elif grade == Grade.D.value:
            conf = 0.40
        elif grade == Grade.ARCHIVED.value:
            conf = 0.20
        else:
            conf = 0.30
        if susp.get("low_confidence"):
            conf *= 0.75
        if susp.get("scope") == "string" and (feats.get("baseline_n") or 0) < 5 and grade not in (Grade.A.value,):
            conf *= 0.9     # 冷启动期（基线样本不足）下调
        proba = {cause: round(conf, 3), Cause.NORMAL.value: round(1 - conf, 3)}
        return cause, float(round(conf, 3)), proba


class MLModel:
    """sklearn HistGradientBoosting 分类器（可解释：特征重要性 + 规则说明一并输出）"""

    def __init__(self, version: str, clf, classes: List[str], feature_names: List[str]):
        self.version, self.clf, self.classes, self.feature_names = version, clf, classes, feature_names

    def predict(self, feats: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        x = np.array([[feats.get(f, np.nan) if feats.get(f) is not None else np.nan for f in self.feature_names]], dtype=float)
        proba = self.clf.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return self.classes[idx], float(proba[idx]), {c: float(round(p, 3)) for c, p in zip(self.classes, proba)}

    @staticmethod
    def make_estimator(n_samples: int):
        """样本少（冷启动早期）用随机森林更稳；样本多用直方图梯度提升。两者都支持缺失值、都可输出特征重要性。"""
        if n_samples < 200:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                          random_state=42, n_jobs=-1), "rf"
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=0.5,
                                              class_weight="balanced", random_state=42), "hgb"

    @staticmethod
    def train(X: pd.DataFrame, y: pd.Series, version: str) -> "MLModel":
        clf, kind = MLModel.make_estimator(len(y))
        clf.fit(X.to_numpy(dtype=float), y.to_numpy())
        return MLModel(f"{kind}-{version}", clf, list(clf.classes_), list(X.columns))

    def feature_importance(self) -> Dict[str, float]:
        imp = getattr(self.clf, "feature_importances_", None)
        if imp is None:
            return {}
        return {f: round(float(v), 4) for f, v in sorted(zip(self.feature_names, imp), key=lambda t: -t[1])[:10]}

    def save(self, path: Path) -> None:
        import joblib
        joblib.dump({"version": self.version, "clf": self.clf, "classes": self.classes, "features": self.feature_names}, path)

    @staticmethod
    def load(path: Path) -> "MLModel":
        import joblib
        d = joblib.load(path)
        return MLModel(d["version"], d["clf"], d["classes"], d["features"])


# ---------------------------------------------------------------------------- 智能体
class ClassifyAgent(Agent):
    name = "classify"
    stage = "阶段5 AI判定"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        today = ctx.run_date.isoformat()
        susp = ctx.store.query("SELECT * FROM suspects WHERE date=? AND status NOT IN (?,?)",
                               (today, SuspectStatus.REJECTED.value, SuspectStatus.CLOSED.value))
        model = self._load_active_model(ctx)
        proxy = RuleProxyModel()
        verdicts: List[Dict] = []
        novelty = self._novelty_scores(ctx)
        for s in susp.to_dict("records"):
            sig = Signature.model_validate_json(s["signature_json"]) if s.get("signature_json") else Signature()
            bl, prof, sg = None, None, pd.DataFrame()
            if s["scope"] == "string":
                b = ctx.store.query("SELECT * FROM baselines WHERE string_id=? AND date=?", (s["string_id"], today))
                bl = b.iloc[0] if not b.empty else None
                p = ctx.store.query("SELECT * FROM strings WHERE string_id=?", (s["string_id"],))
                prof = p.iloc[0] if not p.empty else None
                sg = ctx.store.query("SELECT am_pm_skew FROM signatures WHERE string_id=? AND date=?", (s["string_id"], today))
            feats = build_features(s, bl, sig, prof, ctx.run_date)
            if s["scope"] == "string" and not sg.empty and pd.notna(sg["am_pm_skew"].iloc[0]):
                feats["am_pm_skew"] = float(sg["am_pm_skew"].iloc[0])
            r_cause, r_conf, r_proba = proxy.predict(s, feats, sig)
            if model is not None:
                m_cause, m_conf, m_proba = model.predict(feats)
                cause, conf, proba, mv = m_cause, m_conf, m_proba, model.version
                # 护栏：模型判"误报"但规则为 A/B 强信号时，转人工而不是直接忽略
                if cause == Cause.NORMAL.value and s["grade"] in (Grade.A.value, Grade.B.value):
                    conf = min(conf, 0.8)
            else:
                cause, conf, proba, mv = r_cause, r_conf, r_proba, proxy.version
            action = self._action(ctx, s, cause, conf)
            expl = self._explain(s, bl, sig, prof, cause, conf, action, mv)
            verdicts.append({
                "suspect_id": s["suspect_id"], "model_version": mv, "cause": cause, "confidence": conf, "action": action,
                "explanation": expl, "proba_json": json.dumps(proba, ensure_ascii=False),
                "novelty_score": novelty.get(s.get("string_id")), "created_at": datetime.now().isoformat(),
            })
        # ---- 无监督兜底：不在嫌疑表里但孤立森林打分很低的串 → E 级观察
        extra = self._novelty_suspects(ctx, novelty, set(susp["string_id"].dropna()))
        if extra:
            ctx.store.upsert_df("suspects", pd.DataFrame(extra))
            for e in extra:
                verdicts.append({"suspect_id": e["suspect_id"], "model_version": "isolation-forest", "cause": Cause.UNKNOWN.value,
                                 "confidence": 0.3, "action": Action.OBSERVE.value,
                                 "explanation": f"无监督异常分 {novelty[e['string_id']]:.3f}（越低越异常），规则未命中，疑新模式，入观察",
                                 "proba_json": "{}", "novelty_score": novelty[e["string_id"]], "created_at": datetime.now().isoformat()})
        if verdicts:
            ctx.store.upsert_df("verdicts", pd.DataFrame(verdicts))
        by_action = pd.Series([v["action"] for v in verdicts]).value_counts().to_dict() if verdicts else {}
        res.metrics = {"verdicts": len(verdicts), "by_action": by_action, "model": model.version if model else proxy.version,
                       "novelty_new": len(extra)}
        res.artifacts = {"verdicts": verdicts}
        ctx.shared["verdicts_today"] = verdicts
        res.summary = f"判定 {len(verdicts)} 条（模型 {res.metrics['model']}）：" + " ".join(f"{k}={v}" for k, v in by_action.items())
        return res

    # ------------------------------------------------------------------ 分流
    def _action(self, ctx: AgentContext, s: Dict, cause: str, conf: float) -> str:
        c = ctx.settings.confidence
        if s["grade"] == Grade.ARCHIVED.value or cause == Cause.NORMAL.value:
            return Action.OBSERVE.value if conf >= c.mid or s["grade"] == Grade.ARCHIVED.value else Action.REVIEW.value
        if s["grade"] == Grade.C.value or cause in (Cause.ORIENTATION.value, Cause.PROFILE_ERROR.value):
            return Action.REVIEW.value          # 档案问题：核排布图，不派现场
        if s["grade"] in (Grade.A.value, Grade.B.value) and conf >= c.high:
            return Action.DISPATCH.value
        if conf >= c.mid:
            return Action.REVIEW.value
        return Action.OBSERVE.value

    # ------------------------------------------------------------------ 解释
    @staticmethod
    def _explain(s: Dict, bl: Optional[pd.Series], sig: Signature, prof: Optional[pd.Series], cause: str, conf: float,
                 action: str, mv: str) -> str:
        parts = []
        if s["scope"] == "inverter":
            parts.append(f"逆变器 {s['inverter_id']} 整机参照偏差 {s['dev_today']:+.1%}，连续 {s['persist_days']} 个晴好天")
        else:
            dev = s.get("dev_today")
            if dev is not None and not (isinstance(dev, float) and np.isnan(dev)):
                parts.append(f"中午窗口互比偏差 {dev:+.1%}")
            if bl is not None and pd.notna(bl.get("baseline_ratio")):
                parts.append(f"自身基线比 {bl['baseline_ratio']:.3f}（{int(bl['baseline_n'])} 个晴好天）")
            if s.get("persist_days"):
                parts.append(f"持续 {s['persist_days']} 天")
        parts.extend(sig.notes)
        if prof is not None:
            p = []
            if pd.notna(prof.get("pv_index")):
                p.append(f"pv{int(prof['pv_index'])}")
            if pd.notna(prof.get("n_modules")):
                p.append(f"{int(prof['n_modules'])} 块")
            if pd.notna(prof.get("azimuth")) and pd.notna(prof.get("tilt")):
                p.append(f"朝向 {prof['azimuth']:.0f}° 倾角 {prof['tilt']:.0f}°")
            if prof.get("roof_area"):
                p.append(str(prof["roof_area"]))
            p.append("已备案遮挡" if prof.get("known_shading") else "无备案遮挡")
            parts.append("档案：" + "，".join(p))
        parts.append(f"→ {cause}（置信度 {conf:.0%}，{mv}）→ {action}")
        return "；".join(parts)

    # ------------------------------------------------------------------ 模型管理（生命周期：训练 → 留出集考试 → 达标上线 → 线上监督 → 不达标回退）
    def _load_active_model(self, ctx: AgentContext) -> Optional[MLModel]:
        row = ctx.store.query("SELECT * FROM model_registry WHERE active=1 ORDER BY trained_at DESC LIMIT 1")
        if row.empty:
            return None
        try:
            model = MLModel.load(Path(row["path"].iloc[0]))
        except Exception as e:
            ctx.store.alert(ctx.run_date, "warn", self.name, f"模型加载失败，回退规则代理: {e}")
            return None
        demoted, why = self.live_supervision(ctx, model.version)
        if demoted:
            ctx.store.alert(ctx.run_date, "warn", self.name, f"模型 {model.version} 已自动降级回规则代理：{why}")
            return None
        return model

    @staticmethod
    def training_set(ctx: AgentContext, with_meta: bool = False):
        """标注 = 人工复核（确认/修改 → 真实原因；驳回 → 正常(误报)）；特征取嫌疑当日基线+签名。
        with_meta=True 时额外返回 meta（suspect_id, date, rule_cause）用于按时间留出与规则代理对照。"""
        rv = ctx.store.query(
            "SELECT s.*, r.decision, r.true_cause AS label_cause FROM reviews r JOIN suspects s ON s.suspect_id=r.suspect_id "
            "WHERE r.decision IN ('confirm','modify','reject') ORDER BY r.reviewed_at")
        if rv.empty:
            empty = (pd.DataFrame(), pd.Series(dtype=str))
            return (*empty, pd.DataFrame()) if with_meta else empty
        rv = rv.drop_duplicates(subset=["suspect_id"], keep="last")
        rows, ys, meta = [], [], []
        for s in rv.to_dict("records"):
            sig = Signature.model_validate_json(s["signature_json"]) if s.get("signature_json") else Signature()
            bl = prof = None
            if s["scope"] == "string":
                b = ctx.store.query("SELECT * FROM baselines WHERE string_id=? AND date=?", (s["string_id"], s["date"]))
                bl = b.iloc[0] if not b.empty else None
                p = ctx.store.query("SELECT * FROM strings WHERE string_id=?", (s["string_id"],))
                prof = p.iloc[0] if not p.empty else None
            feats = build_features(s, bl, sig, prof, date.fromisoformat(s["date"]))
            rows.append(feats)
            ys.append(Cause.NORMAL.value if s["decision"] == "reject" else (s["label_cause"] or s["rule_cause"]))
            # 时间留出按案例"首次报警日"排序（suspects.date 是案例最近更新日，在办案例都等于今天，不能用）
            meta.append({"suspect_id": s["suspect_id"], "date": s.get("first_date") or s["date"], "rule_cause": s["rule_cause"], "grade": s["grade"]})
        X = pd.DataFrame(rows)[FEATURES]
        y = pd.Series(ys)
        if with_meta:
            return X, y, pd.DataFrame(meta)
        return X, y

    # ---- 评估
    @staticmethod
    def _metrics(y_true: pd.Series, y_pred) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, f1_score
        y_true = pd.Series(y_true).astype(str).to_numpy()
        y_pred = pd.Series(y_pred).astype(str).to_numpy()
        return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4), "n": int(len(y_true))}

    @staticmethod
    def time_split(meta: pd.DataFrame, frac: float, min_holdout: int = 10):
        """按嫌疑日期排序，最近 frac（至少 min_holdout 条）留出；样本太少时返回 None（改用交叉验证）"""
        order = np.argsort(meta["date"].to_numpy())
        n = len(order)
        k = max(min_holdout, int(round(n * frac)))
        if n - k < max(10, min_holdout):
            return None
        return order[:n - k], order[n - k:]

    def _exam(self, ctx: AgentContext, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame, incumbent: Optional[MLModel]) -> Dict:
        """挑战者 / 现役 / 规则代理 在同一留出集上的成绩"""
        split = self.time_split(meta, ctx.settings.confidence.holdout_fraction)
        out: Dict = {"method": "time_holdout" if split else "cv"}
        if split:
            tr, ho = split
            ch = MLModel.train(X.iloc[tr], y.iloc[tr], "exam")
            pred = [ch.predict(r)[0] for r in X.iloc[ho].to_dict("records")]
            out["challenger"] = self._metrics(y.iloc[ho], pred)
            out["rule_proxy"] = self._metrics(y.iloc[ho], meta.iloc[ho]["rule_cause"])
            if incumbent is not None:
                try:
                    ipred = [incumbent.predict(r)[0] for r in X.iloc[ho].to_dict("records")]
                    out["incumbent"] = self._metrics(y.iloc[ho], ipred)
                except Exception:
                    out["incumbent"] = None
            out["holdout_dates"] = [str(meta.iloc[ho]["date"].min()), str(meta.iloc[ho]["date"].max())]
        else:
            from sklearn.model_selection import cross_val_predict
            cv = int(min(5, y.value_counts().min()))
            if cv >= 2:
                est, _ = MLModel.make_estimator(len(y))
                pred = cross_val_predict(est, X.to_numpy(dtype=float), y.to_numpy(), cv=cv)
                out["challenger"] = self._metrics(y, pred)
                out["rule_proxy"] = self._metrics(y, meta["rule_cause"])
            else:
                out["challenger"] = {"accuracy": float("nan"), "macro_f1": float("nan"), "n": int(len(y))}
                out["rule_proxy"] = self._metrics(y, meta["rule_cause"])
        return out

    def train(self, ctx: AgentContext, force: bool = False, reason: str = "") -> Dict:
        """训练 + 考试 + 决定是否上线。force=True 为人工强制上线（仍记录成绩）。"""
        X, y, meta = self.training_set(ctx, with_meta=True)
        n = len(y)
        c = ctx.settings.confidence
        if n < c.min_labels_for_ml and not force:
            return {"trained": False, "reason": f"标注 {n} 条 < {c.min_labels_for_ml}，继续使用规则代理模型"}
        if y.nunique() < 2:
            return {"trained": False, "reason": "标注类别不足 2 类"}
        incumbent = None
        row = ctx.store.query("SELECT * FROM model_registry WHERE active=1 ORDER BY trained_at DESC LIMIT 1")
        if not row.empty:
            try:
                incumbent = MLModel.load(Path(row["path"].iloc[0]))
            except Exception:
                incumbent = None
        exam = self._exam(ctx, X, y, meta, incumbent)
        ch = exam.get("challenger") or {}
        px = exam.get("rule_proxy") or {}
        inc = exam.get("incumbent") or {}
        ch_acc = ch.get("accuracy", float("nan"))
        verdict_ok = True
        why = []
        if not np.isnan(ch_acc):
            if px.get("accuracy") is not None and ch_acc < px["accuracy"] - c.promote_tolerance:
                verdict_ok, why = False, why + [f"挑战者准确率 {ch_acc:.0%} 低于规则代理 {px['accuracy']:.0%}"]
            if inc and inc.get("accuracy") is not None and ch_acc < inc["accuracy"] - c.promote_tolerance:
                verdict_ok, why = False, why + [f"挑战者准确率 {ch_acc:.0%} 低于现役模型 {inc['accuracy']:.0%}"]
        promote = verdict_ok or force
        stamp = f"{datetime.now():%Y%m%d%H%M%S}"
        model = MLModel.train(X, y, stamp)          # 最终模型用全部标注训练
        path = ctx.settings.models_dir / f"{model.version}.joblib"
        model.save(path)
        metrics = {"exam": exam, "class_counts": y.value_counts().to_dict(), "feature_importance": model.feature_importance(),
                   "reason": reason, "forced": bool(force), "rejected_because": why}
        status = "active" if promote else "rejected"
        if promote:
            ctx.store.execute("UPDATE model_registry SET active=0, status=CASE WHEN status='active' THEN 'retired' ELSE status END")
        ctx.store.execute(
            "INSERT OR REPLACE INTO model_registry (model_version, trained_at, n_labels, classes_json, metrics_json, path, active, status, holdout_json) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (model.version, datetime.now().isoformat(), n, json.dumps(model.classes, ensure_ascii=False),
             json.dumps(metrics, ensure_ascii=False, default=str), str(path), int(promote), status,
             json.dumps(exam, ensure_ascii=False, default=str)))
        ctx.store.set_state("model:last_train", {"at": datetime.now().isoformat(), "n_labels": n, "version": model.version, "promoted": promote})
        return {"trained": True, "promoted": promote, "version": model.version, "n_labels": n, "exam": exam,
                "cv_accuracy": ch_acc, "classes": model.classes, "feature_importance": model.feature_importance(),
                "reason": reason, "rejected_because": why}

    @staticmethod
    def retrain_due(ctx: AgentContext, drift_psi: Optional[float] = None) -> Tuple[bool, str]:
        """是否该重训：无模型但标注够了 / 新标注够多 / 到周期且有新标注 / 分布漂移且有新标注"""
        c = ctx.settings.confidence
        n_labels = int(ctx.store.scalar("SELECT COUNT(DISTINCT suspect_id) FROM reviews WHERE decision IN ('confirm','modify','reject')") or 0)
        last = ctx.store.get_state("model:last_train") or {}
        active = ctx.store.query("SELECT trained_at FROM model_registry WHERE active=1")
        new_labels = n_labels - int(last.get("n_labels", 0))
        if n_labels < c.min_labels_for_ml:
            return False, f"标注 {n_labels} < {c.min_labels_for_ml}"
        if active.empty and not last:
            return True, f"标注已达 {n_labels} 条，首次训练"
        if new_labels >= c.retrain_min_new_labels:
            return True, f"新增标注 {new_labels} ≥ {c.retrain_min_new_labels}"
        if last.get("at"):
            days = (datetime.now() - datetime.fromisoformat(last["at"])).days
            if days >= c.retrain_every_days and new_labels >= 5:
                return True, f"距上次训练 {days} 天且新增标注 {new_labels}"
        if drift_psi is not None and not np.isnan(drift_psi) and drift_psi > c.drift_psi_retrain and new_labels >= 5:
            return True, f"偏差分布漂移 PSI={drift_psi:.2f} 且新增标注 {new_labels}"
        return False, f"新增标注 {new_labels}，未到重训条件"

    @staticmethod
    def live_supervision(ctx: AgentContext, model_version: str) -> Tuple[bool, str]:
        """线上监督：近 30 天已复核的嫌疑里，模型判定与复核结论一致率 vs 规则代理一致率。差太多 → 降级。"""
        c = ctx.settings.confidence
        since = (ctx.run_date - timedelta(days=30)).isoformat()
        df = ctx.store.query(
            "SELECT r.decision, r.true_cause, s.rule_cause, v.cause AS ml_cause FROM reviews r "
            "JOIN suspects s ON s.suspect_id=r.suspect_id JOIN verdicts v ON v.suspect_id=r.suspect_id "
            "WHERE r.decision IN ('confirm','modify','reject') AND r.reviewed_at>=? AND v.model_version=?", (since, model_version))
        if len(df) < c.demote_min_samples:
            return False, f"样本 {len(df)} 不足"
        label = np.where(df["decision"] == "reject", Cause.NORMAL.value, df["true_cause"].fillna(df["rule_cause"]))
        ml = float((df["ml_cause"].to_numpy() == label).mean())
        px = float((df["rule_cause"].to_numpy() == label).mean())
        ctx.store.set_state("model:live_agreement", {"model": ml, "rule_proxy": px, "n": int(len(df)), "date": ctx.run_date.isoformat()})
        if ml < px - c.demote_margin:
            ctx.store.execute("UPDATE model_registry SET active=0, status='demoted' WHERE model_version=?", (model_version,))
            return True, f"近 30 天一致率 模型 {ml:.0%} vs 规则 {px:.0%}（n={len(df)}）"
        return False, f"一致率 模型 {ml:.0%} vs 规则 {px:.0%}"

    # ------------------------------------------------------------------ 无监督兜底
    def _novelty_scores(self, ctx: AgentContext) -> Dict[str, float]:
        """孤立森林：用近 30 天全部组串的 (基线+签名) 特征训练，对今天打分（越低越异常）"""
        today = ctx.run_date.isoformat()
        start = (ctx.run_date - timedelta(days=30)).isoformat()
        df = ctx.store.query(
            "SELECT b.string_id, b.date, b.dev_today, b.dev_vs_own, b.dev_roll7, s.ratio_std, s.am_pm_skew, s.dip_depth, s.voltage_offset "
            "FROM baselines b LEFT JOIN signatures s ON s.string_id=b.string_id AND s.date=b.date "
            "WHERE b.date BETWEEN ? AND ? AND b.weather_tag='clear'", (start, today))
        if df.empty or (df["date"] == today).sum() < 10 or len(df) < 60:
            return {}
        cols = ["dev_today", "dev_vs_own", "dev_roll7", "ratio_std", "am_pm_skew", "dip_depth", "voltage_offset"]
        X = df[cols].astype(float).fillna(0.0)
        try:
            from sklearn.ensemble import IsolationForest
            iso = IsolationForest(n_estimators=150, contamination=0.02, random_state=42).fit(X.to_numpy())
            sc = iso.decision_function(X.to_numpy())
        except Exception:
            return {}
        df["score"] = sc
        t = df[df["date"] == today]
        return dict(zip(t["string_id"], t["score"].astype(float)))

    def _novelty_suspects(self, ctx: AgentContext, novelty: Dict[str, float], already: set) -> List[Dict]:
        if not novelty:
            return []
        cand = sorted(((v, k) for k, v in novelty.items() if k not in already and v < -0.05))[:3]
        out = []
        prof = ctx.store.strings().set_index("string_id") if cand else None
        for score, sid in cand:
            if sid not in prof.index:
                continue
            key = f"{sid}@novel@{ctx.run_date.isoformat()}"
            exists = ctx.store.scalar("SELECT COUNT(*) FROM suspects WHERE string_id=? AND status IN ('open','reviewing')", (sid,))
            if exists:
                continue
            out.append({"suspect_id": key, "string_id": sid, "inverter_id": prof.loc[sid, "inverter_id"],
                        "station_id": prof.loc[sid, "station_id"], "date": ctx.run_date.isoformat(),
                        "first_date": ctx.run_date.isoformat(), "scope": "string", "grade": Grade.E.value,
                        "rule_cause": Cause.UNKNOWN.value, "dev_today": None, "persist_days": 0,
                        "signature_json": Signature(notes=[f"孤立森林异常分 {score:.3f}"]).model_dump_json(),
                        "low_confidence": 1, "archived_shading": 0, "status": SuspectStatus.OPEN.value,
                        "created_at": datetime.now().isoformat()})
        return out
