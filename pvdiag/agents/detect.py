"""阶段4 · 异常检测分析（可解释规则引擎）

① 每日自动巡检：全窗口重算嫌疑串（偏差 ≤-3% 入表），产出分级清单 A/B/C/D/E
② 全天曲线签名分型：
   - 分时段凹陷（某时段电流跌到群组 ~70%，其余时段正常）→ 固定遮挡 / 旁路二极管
   - 全天恒定比例偏差、阴天消失 → 朝向/倾角差异（档案差异）
   - 上下午比值倾斜（东西向）→ 朝向差异
   - 全天 0A 多日 → 断线/未接；电流钉死 → 数据/通信问题；电压异位 → 串长/MPPT 配置
   - 整机参照偏低（机间互比）→ MPPT 异常 / 限发
③ 归一化点：晴好天过滤、持续度（≥2 晴好天）、中午中位数抗抖动、限发平顶剔除、停机剔除、已备案遮挡抑制
每日把全部组串的签名写入 signatures 表（AI 特征 + 复核界面 + 凹陷持续度）。
"""
from __future__ import annotations

import json
import warnings
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..features import slot_label
from ..models import Cause, Grade, Signature, SuspectStatus
from .base import Agent, AgentContext, AgentResult

SEVERITY = {Grade.A.value: 0, Grade.B.value: 1, Grade.C.value: 2, Grade.D.value: 3, Grade.E.value: 4, Grade.ARCHIVED.value: 5}


class DetectAgent(Agent):
    name = "detect"
    stage = "阶段4 异常检测（规则引擎）"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        all_susp: List[Dict] = []
        for sid in ctx.stations():
            all_susp.extend(self.detect_station(ctx, sid, ctx.run_date))
        all_susp = self._reconcile_cases(ctx, all_susp)
        if all_susp:
            ctx.store.upsert_df("suspects", pd.DataFrame(all_susp))
            # 每日快照（审计留痕 + 自调参影子回放的素材：某天某案例当时是什么分级/原因/偏差）
            snap = pd.DataFrame([{"suspect_id": s["suspect_id"], "date": s["date"], "grade": s["grade"], "rule_cause": s["rule_cause"],
                                  "dev_today": s.get("dev_today"), "persist_days": s.get("persist_days", 0),
                                  "low_confidence": s.get("low_confidence", 0)} for s in all_susp])
            ctx.store.upsert_df("suspect_daily", snap)
        by_grade = pd.Series([s["grade"] for s in all_susp]).value_counts().to_dict() if all_susp else {}
        res.metrics = {"suspects": len(all_susp), "by_grade": by_grade,
                       "new_cases": sum(1 for s in all_susp if s.get("_new"))}
        for s in all_susp:
            s.pop("_new", None); s.pop("_hard", None)
        res.artifacts = {"suspects": all_susp}
        ctx.shared["suspects_today"] = all_susp
        res.summary = f"嫌疑 {len(all_susp)} 条（新案例 {res.metrics['new_cases']}）：" + " ".join(f"{k}={v}" for k, v in sorted(by_grade.items()))
        return res

    # ------------------------------------------------------------------ 案例连续性
    def _reconcile_cases(self, ctx: AgentContext, cands: List[Dict]) -> List[Dict]:
        """同一组串/逆变器的异常在多日之间是同一个"案例"：
        - 已有未关闭案例 → 沿用其 suspect_id/first_date/流程状态，只刷新偏差、持续度，分级只升不降
        - 最近一次案例被驳回（误报）→ 偏差未显著恶化、分级未升级时不再重复报警
        - 否则开新案例"""
        hist = ctx.store.query("SELECT * FROM suspects ORDER BY date")
        active: Dict[str, Dict] = {}
        latest_rejected: Dict[str, Dict] = {}
        for r in hist.to_dict("records"):
            key = r["string_id"] if r["scope"] == "string" else f"inv:{r['inverter_id']}"
            if r["status"] in (SuspectStatus.REJECTED.value,):
                latest_rejected[key] = r
            elif r["status"] != SuspectStatus.CLOSED.value:
                active[key] = r
        out = []
        for c in cands:
            key = c["string_id"] if c["scope"] == "string" else f"inv:{c['inverter_id']}"
            a = active.get(key)
            if a is not None:
                c["suspect_id"] = a["suspect_id"]
                c["first_date"] = a.get("first_date") or a["date"]
                c["created_at"] = a["created_at"]
                c["status"] = a["status"]
                # 分级只升不降；C（朝向/档案解释）是对偏差的"解释"，优先于 B 的"未知低发"
                escalated = SEVERITY.get(c["grade"], 9) < SEVERITY.get(a["grade"], 9)
                # 已被解释为朝向/档案差异（C）的案例：非晴好天"沿用持续低发"给出的 B 不能推翻解释，只有 A 级硬证据才升级
                physics_c = a["grade"] == Grade.C.value and '"orientation_fit":"' in (a.get("signature_json") or "")
                if physics_c and c["grade"] == Grade.B.value and not (c.get("_hard") or False):
                    c["grade"], c["rule_cause"], escalated = Grade.C.value, a["rule_cause"], False
                    try:        # 把物理拟合结论带到今天的签名里，否则下一个阴天保护就失效
                        old_sig, new_sig = json.loads(a["signature_json"]), json.loads(c["signature_json"])
                        new_sig["orientation_fit"], new_sig["orientation_like"] = old_sig.get("orientation_fit"), True
                        c["signature_json"] = json.dumps(new_sig, ensure_ascii=False)
                    except Exception:
                        pass
                if c["grade"] != Grade.C.value and SEVERITY.get(a["grade"], 9) < SEVERITY.get(c["grade"], 9):
                    c["grade"] = a["grade"]
                    if c["rule_cause"] == Cause.UNKNOWN.value:
                        c["rule_cause"] = a["rule_cause"]
                # 原因稳定性："未知"不覆盖已有的具体原因；两个具体原因打架时按案例历史多数票（分级升级时以今天为准）
                if not escalated and c["rule_cause"] != a["rule_cause"]:
                    if c["rule_cause"] == Cause.UNKNOWN.value:
                        c["rule_cause"] = a["rule_cause"]
                    elif a["rule_cause"] != Cause.UNKNOWN.value:
                        c["rule_cause"] = self._majority_cause(ctx, a["suspect_id"], c["rule_cause"], a["rule_cause"])
                c["_new"] = False
                out.append(c)
                continue
            rj = latest_rejected.get(key)
            if rj is not None:
                worse = (c.get("dev_today") is not None and rj.get("dev_today") is not None
                         and c["dev_today"] <= rj["dev_today"] - 0.05)
                escalated = SEVERITY.get(c["grade"], 9) < SEVERITY.get(rj["grade"], 9)
                recent = (pd.Timestamp(c["date"]) - pd.Timestamp(rj["date"])).days <= 30
                if recent and not worse and not escalated:
                    continue
            c["first_date"] = c["date"]
            c["_new"] = True
            out.append(c)
        # 自愈关闭（带滞回）：open/reviewing 案例今天不再是嫌疑，且最近 N 个晴好天偏差都回到正常带 → 关闭
        # （单个"看似晴天"的平滑多云日会让朝向/遮挡偏差暂时变小，不能据此关案）
        today = ctx.run_date.isoformat()
        n_ok = ctx.settings.dispatch.recheck_clear_days
        cand_keys = {(c["string_id"] if c["scope"] == "string" else f"inv:{c['inverter_id']}") for c in cands}
        for key, a in active.items():
            if key in cand_keys or a["status"] not in (SuspectStatus.OPEN.value, SuspectStatus.REVIEWING.value):
                continue
            if a["scope"] != "string":
                continue
            recent = ctx.store.query(
                "SELECT b.dev_today, f.zero_all_day, f.stuck, s.dip_depth, s.voltage_offset FROM baselines b "
                "JOIN daily_features f ON f.string_id=b.string_id AND f.date=b.date "
                "LEFT JOIN signatures s ON s.string_id=b.string_id AND s.date=b.date "
                "WHERE b.string_id=? AND b.date<=? AND b.weather_tag='clear' ORDER BY b.date DESC LIMIT ?", (key, today, n_ok))
            if len(recent) < n_ok:
                continue
            th = ctx.settings.thresholds
            healthy = recent["dev_today"].notna() & (recent["dev_today"] > th.normal_floor) \
                & (recent["zero_all_day"] == 0) & (recent["stuck"] == 0) \
                & (recent["dip_depth"].fillna(0) < 0.1) & (recent["voltage_offset"].fillna(0).abs() <= th.voltage_offset)
            # 间歇停机是"来来回回"的：近 7 天内还有停机签名就不算痊愈（否则会关了又开、一串多案例）
            recent_outage = ctx.store.scalar(
                "SELECT COALESCE(SUM(CASE WHEN outage_slots>=? THEN 1 ELSE 0 END),0) FROM signatures WHERE string_id=? AND date BETWEEN ? AND ?",
                (th.outage_min_slots, key, (ctx.run_date - timedelta(days=7)).isoformat(), today)) or 0
            if healthy.all() and not recent_outage:
                ctx.store.execute("UPDATE suspects SET status=? WHERE suspect_id=?", (SuspectStatus.CLOSED.value, a["suspect_id"]))
                ctx.store.execute(
                    "INSERT INTO reviews (suspect_id, reviewer, decision, true_cause, reject_reason, note, reviewed_at) VALUES (?,?,?,?,?,?,?)",
                    (a["suspect_id"], "system", "auto_close", None, None, f"{today} 连续 {n_ok} 个晴好天偏差回到正常带，自愈关闭",
                     pd.Timestamp.now().isoformat()))
        return out

    @staticmethod
    def _majority_cause(ctx: AgentContext, suspect_id: str, today_cause: str, active_cause: str) -> str:
        h = ctx.store.query("SELECT rule_cause, COUNT(*) n FROM suspect_daily WHERE suspect_id=? AND rule_cause!=? GROUP BY rule_cause",
                            (suspect_id, Cause.UNKNOWN.value))
        votes = dict(zip(h["rule_cause"], h["n"])) if not h.empty else {}
        votes[today_cause] = votes.get(today_cause, 0) + 1
        best = max(votes.items(), key=lambda kv: (kv[1], kv[0] == active_cause))
        return best[0]

    # ------------------------------------------------------------------ 单站
    def detect_station(self, ctx: AgentContext, station_id: str, d: date) -> List[Dict]:
        th, cfg = ctx.settings.thresholds, ctx.settings.baseline
        today = d.isoformat()
        strings = ctx.store.strings(station_id)
        bl = ctx.store.baselines(station_id, d, d)
        if bl.empty or strings.empty:
            return []
        feats = ctx.store.daily_features(station_id, d - timedelta(days=10), d)
        ft = feats[feats["date"] == today].set_index("string_id")
        weather_row = ctx.store.station_daily(station_id, d, d)
        weather = weather_row["weather_tag"].iloc[0] if not weather_row.empty else "nodata"
        inv_daily = ctx.store.query(
            "SELECT i.* FROM inverter_daily i JOIN inverters v ON v.inverter_id=i.inverter_id WHERE v.station_id=? AND i.date=?",
            (station_id, today)).set_index("inverter_id")
        inv_bl = ctx.store.query(
            "SELECT b.* FROM inverter_baselines b JOIN inverters v ON v.inverter_id=b.inverter_id WHERE v.station_id=? AND b.date=?",
            (station_id, today))
        prof = strings.set_index("string_id")
        bl = bl.set_index("string_id")
        st_row = ctx.store.query("SELECT lat, lon, extra_json FROM stations WHERE station_id=?", (station_id,))
        from ..solar import lon_offset_minutes
        lon_offset = lon_offset_minutes(float(st_row["lon"].iloc[0]) if not st_row.empty and pd.notna(st_row["lon"].iloc[0]) else None,
                                        cfg.tz_meridian_deg)
        try:
            st_extra = json.loads(st_row["extra_json"].iloc[0] or "{}") if not st_row.empty and st_row["extra_json"].iloc[0] else {}
            st_extra = st_extra if isinstance(st_extra, dict) else {}
        except Exception:
            st_extra = {}
        ref_orient = {inv: self._ref_orientation(prof, inv) for inv in prof["inverter_id"].unique()}      # 每机算一次
        mount = st_extra.get("mount", "fixed")
        lat = float(st_row["lat"].iloc[0]) if not st_row.empty and pd.notna(st_row["lat"].iloc[0]) else 30.0

        group_curves = self._group_curves(ft, prof, bl)
        voff = self._voltage_offsets(ft, prof, bl)
        zero_streak, stuck_streak = self._zero_stuck_streaks(feats, ctx.store, station_id, d, th.stale_run_min)
        sig_hist = ctx.store.query(
            "SELECT s.string_id, s.date, s.dip_depth, s.dip_start, s.dip_end, s.outside_dip_ratio, s.voltage_offset, s.weather_tag, s.outage_slots, s.tracker_flag "
            "FROM signatures s JOIN strings p ON p.string_id=s.string_id WHERE p.station_id=? AND s.date BETWEEN ? AND ? ORDER BY s.date DESC",
            (station_id, (d - timedelta(days=15)).isoformat(), (d - timedelta(days=1)).isoformat()))
        hist_by: Dict[str, List[Dict]] = {sid: g.to_dict("records") for sid, g in sig_hist.groupby("string_id", sort=False)} if not sig_hist.empty else {}
        outage_cut = (d - timedelta(days=7)).isoformat()
        prof_rows = prof.to_dict("index"); ft_rows = ft.to_dict("index")
        rainy_caution = weather == "clear" and ("n_clear_recent" in bl and bl["n_clear_recent"].max() <= 1)
        # 近 10 个晴好天偏差的离散度：朝向差异是恒定的（std 很小），积灰/衰减会漂移
        dev_hist = ctx.store.query(
            "SELECT b.string_id, b.dev_today FROM baselines b JOIN strings p ON p.string_id=b.string_id "
            "WHERE p.station_id=? AND b.weather_tag='clear' AND b.dev_today IS NOT NULL AND b.date BETWEEN ? AND ? ORDER BY b.date",
            (station_id, (d - timedelta(days=30)).isoformat(), today))
        dev_std = dev_hist.groupby("string_id")["dev_today"].apply(lambda x: float(x.tail(10).std()) if len(x) >= 4 else np.nan).to_dict() if not dev_hist.empty else {}

        out: List[Dict] = []
        sig_rows: List[Dict] = []
        for sid, b in bl.to_dict("index").items():
            p = prof_rows.get(sid)
            if p is None:
                continue
            f = ft_rows.get(sid)
            h = hist_by.get(sid, [])
            ref = ref_orient.get(p["inverter_id"])
            geo = (lat, ref[0], ref[1], d, lon_offset) if ref is not None else None          # 档案没有朝向 → 不做物理拟合
            sig, skew, rcurve = self._signature(sid, f, p, b, group_curves, inv_daily, th, cfg, mount, geo)
            if sig.tracker_stuck:
                sig.notes.append(f"跟踪电站：清晨比值 {sig.am_ratio:.0%}、傍晚 {sig.pm_ratio:.0%}，中午正常 → 失去跟踪增益（支架卡滞/控制器停）")
            if sig.orientation_like and not sig.orientation_fit and not np.isnan(dev_std.get(sid, np.nan)) and dev_std[sid] > th.orientation_max_drift:
                sig.orientation_like = False      # 偏差在漂移 → 不是恒定的朝向差异（更像积灰/衰减）
                sig.notes.append(f"晴好天偏差离散度 {dev_std[sid]:.1%} 偏大，不按朝向差异处理")
            v_off = voff.get(sid)
            v_days = self._voltage_days(h, v_off, th) if v_off is not None else 0
            if v_off is not None and abs(v_off) > th.voltage_offset and v_days + 1 >= th.voltage_offset_days:
                sig.voltage_offset = v_off
            dip_days = self._dip_days(h, sig, th)
            if sig.outage_hours > 0:
                sig.outage_days = sum(1 for r in h if r["date"] >= outage_cut and r.get("outage_slots") is not None
                                      and not pd.isna(r["outage_slots"]) and r["outage_slots"] >= th.outage_min_slots) + 1
            out_slots = int(round(sig.outage_hours * 60 / cfg.slot_minutes)) if sig.outage_hours else 0
            tracker_days = self._leading_flag_days(h, "tracker_flag") if sig.tracker_stuck else 0
            sig_rows.append({"string_id": sid, "date": today, "weather_tag": weather, "ratio_mean": sig.ratio_mean,
                             "ratio_std": sig.ratio_std, "am_pm_skew": skew, "dip_depth": sig.dip_depth,
                             "dip_start": sig.dip_start, "dip_end": sig.dip_end, "outside_dip_ratio": sig.outside_dip_ratio,
                             "voltage_offset": v_off, "outage_slots": out_slots, "tracker_flag": int(sig.tracker_stuck)})
            grade, cause, notes = self._grade(b, sig, p, zero_streak.get(sid, 0), stuck_streak.get(sid, 0),
                                              weather, dip_days, th, tracker_days)
            if grade is None:
                continue

            # 冷启动（该串晴好天基线样本不足）：仅凭互比偏差不下 B 级结论——只做同日互比、进观察带
            if grade == Grade.B.value and int(b.get("cold_start", 0) or 0) == 1 and not (sig.dip_depth or sig.outage_hours or sig.inverter_level):
                grade = th.cold_start_cap_grade
                notes.append(f"冷启动期（晴好天基线样本 {int(b.get('baseline_n', 0))} < {cfg.min_clear_days}），分级上限 {grade}，待基线建立后再升级")
            sig.notes.extend(notes)
            archived = grade == Grade.ARCHIVED.value
            low_conf = bool(rainy_caution and grade in (Grade.B.value, Grade.D.value, Grade.E.value))
            if low_conf:
                sig.notes.append(f"连续 ≥{th.rainy_streak_caution_days} 天无晴好天后的首个晴天，置信度下调")
            first = b.get("first_breach_date") if isinstance(b.get("first_breach_date"), str) else None
            case_date = first or today
            streak = max(zero_streak.get(sid, 0), stuck_streak.get(sid, 0))
            if grade == Grade.A.value and streak:
                case_date = (d - timedelta(days=streak - 1)).isoformat()
            elif sig.dip_depth is not None and dip_days:
                case_date = min(case_date, (d - timedelta(days=dip_days)).isoformat())
            persist = int(b["persist_days_5"]) if grade == Grade.B.value else int(b["persist_days_3"])
            persist = max(persist, dip_days + 1 if sig.dip_depth is not None else 0, streak)
            hard = bool(sig.dip_depth is not None or sig.outage_hours or sig.zero_all_day or sig.stuck or sig.voltage_offset is not None or sig.tracker_stuck)
            out.append({"_hard": hard,
                "suspect_id": f"{sid}@{case_date}", "string_id": sid, "inverter_id": p["inverter_id"],
                "station_id": station_id, "date": today, "scope": "string", "grade": grade, "rule_cause": cause,
                "dev_today": None if pd.isna(b["dev_today"]) else float(b["dev_today"]),
                "persist_days": persist,
                "signature_json": sig.model_dump_json(), "low_confidence": int(low_conf),
                "archived_shading": int(archived), "status": SuspectStatus.OPEN.value,
                "created_at": pd.Timestamp.now().isoformat(),
            })
        if sig_rows:
            ctx.store.upsert_df("signatures", pd.DataFrame(sig_rows))

        # ---- 整机级嫌疑（机间互比兜底整机问题）
        for r in inv_bl.itertuples():
            if r.station_n < 3 or pd.isna(r.inv_dev):
                continue
            if r.inv_dev <= th.fault_floor and r.persist_days >= th.fault_persist_days:
                limited_days = ctx.store.scalar(
                    "SELECT SUM(limited_flag) FROM inverter_daily WHERE inverter_id=? AND date BETWEEN ? AND ?",
                    (r.inverter_id, (d - timedelta(days=7)).isoformat(), today)) or 0
                cause = Cause.CURTAILMENT.value if limited_days >= 3 else Cause.MPPT.value
                sig = Signature(inverter_level=True, notes=[
                    f"整机参照（各串中位电流/Imp）低于站内中位 {r.inv_dev:+.1%}，连续 {r.persist_days} 个晴好天",
                    f"近 7 天限发标记 {int(limited_days)} 天"])
                first_clear = ctx.store.query(
                    "SELECT date FROM inverter_baselines WHERE inverter_id=? AND date<=? AND inv_dev<=? ORDER BY date",
                    (r.inverter_id, today, th.fault_floor))
                first = first_clear["date"].iloc[0] if not first_clear.empty else today
                out.append({
                    "suspect_id": f"{r.inverter_id}@inv@{first}", "string_id": None, "inverter_id": r.inverter_id,
                    "station_id": station_id, "date": today, "scope": "inverter", "grade": Grade.B.value,
                    "rule_cause": cause, "dev_today": float(r.inv_dev), "persist_days": int(r.persist_days),
                    "signature_json": sig.model_dump_json(), "low_confidence": 0, "archived_shading": 0,
                    "status": SuspectStatus.OPEN.value, "created_at": pd.Timestamp.now().isoformat(),
                })
        return out

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _group_curves(ft: pd.DataFrame, prof: pd.DataFrame, bl: pd.DataFrame) -> Dict[str, np.ndarray]:
        curves: Dict[str, List[np.ndarray]] = {}
        inv_of = prof["inverter_id"].to_dict()
        gkey, glevel = bl["group_key"].to_dict(), bl["group_level"].to_dict()
        for sid, row in ft.to_dict("index").items():
            if sid not in inv_of or sid not in gkey:
                continue
            if row["zero_all_day"] or row["stuck"] or pd.isna(row["noon_median_i"]) or row["noon_median_i"] < 0.5:
                continue
            key = gkey[sid] if glevel[sid] == "group" else inv_of[sid]
            vec = np.array([np.nan if v is None else v for v in json.loads(row["i_profile_json"])], dtype=float)
            curves.setdefault(key, []).append(vec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)      # 夜间槽全 NaN 属正常
            return {k: np.nanmedian(np.vstack(v), axis=0) for k, v in curves.items() if len(v) >= 3}

    @staticmethod
    def _voltage_offsets(ft: pd.DataFrame, prof: pd.DataFrame, bl: pd.DataFrame) -> Dict[str, float]:
        """中午电压 vs 同机（同群组）中位电压的相对偏差"""
        if ft.empty:
            return {}
        x = ft[["noon_median_v", "zero_all_day", "stuck"]].join(prof[["inverter_id"]], how="inner")
        x = x[(x["zero_all_day"] == 0) & (x["stuck"] == 0) & x["noon_median_v"].notna() & (x["noon_median_v"] > 0)]
        out = {}
        for inv, g in x.groupby("inverter_id"):
            if len(g) < 3:
                continue
            med = g["noon_median_v"].median()
            for sid, r in g.iterrows():
                out[sid] = float(r["noon_median_v"] / med - 1)
        return out

    @staticmethod
    def _voltage_days(hist: List[Dict], v_off: float, th) -> int:
        """hist：该串近 15 天签名记录（按日期倒序）"""
        n = 0
        for r in hist:
            vo = r.get("voltage_offset")
            if vo is not None and not pd.isna(vo) and abs(vo) > th.voltage_offset and np.sign(vo) == np.sign(v_off):
                n += 1
            else:
                break
        return n

    @staticmethod
    def _dip_days(hist: List[Dict], sig: Signature, th) -> int:
        """此前连续多少个晴好天出现（时段重叠的）同类凹陷；hist 为该串签名记录（日期倒序）"""
        if sig.dip_depth is None or not hist:
            return 0
        n = 0
        for r in hist:
            if r.get("weather_tag") != "clear":
                continue
            dd = r.get("dip_depth")
            if dd is None or pd.isna(dd) or dd < 0.1:
                break
            if r.get("dip_start") and sig.dip_start and not (r["dip_end"] <= sig.dip_start or r["dip_start"] >= sig.dip_end):
                n += 1
            else:
                break
        return n

    @staticmethod
    def _zero_stuck_streaks(feats: pd.DataFrame, store, station_id: str, d: date, stale_min: int = 90) -> Tuple[Dict[str, int], Dict[str, int]]:
        """连续全天 0 电流 / 钉死天数；跳过全站无数据日（缺数据≠零发电）"""
        if feats.empty:
            return {}, {}
        sd = store.station_daily(station_id, d - timedelta(days=10), d)
        producing = set(sd[sd["weather_tag"] != "nodata"]["date"]) if not sd.empty else set(feats["date"])
        zero, stuck = {}, {}
        fp = feats[feats["date"].isin(producing)].sort_values(["string_id", "date"], ascending=[True, False])
        z_all = fp["zero_all_day"].to_numpy(dtype=float) == 1
        sr = fp["stale_run_min"].to_numpy(dtype=float) if "stale_run_min" in fp.columns else np.zeros(len(fp))
        # 全天钉死，或日间连续 ≥stale_min 分钟钉死不动（半天冻结：pvanalytics 标注数据里常见）
        s_all = (fp["stuck"].to_numpy(dtype=float) == 1) | (np.nan_to_num(sr) >= stale_min)
        for sid, idx in fp.groupby("string_id", sort=False).indices.items():
            zero[sid] = int(DetectAgent._leading_true(z_all[idx]))
            stuck[sid] = int(DetectAgent._leading_true(s_all[idx]))
        return zero, stuck

    @staticmethod
    def _leading_true(mask: np.ndarray) -> int:
        """从头开始连续 True 的个数（序列已按日期倒序 → 即"连续至今"的天数）"""
        if len(mask) == 0:
            return 0
        nz = np.flatnonzero(~mask)
        return int(nz[0]) if len(nz) else int(len(mask))

    @staticmethod
    def _runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
        """布尔序列中长度 ≥ min_len 的连续 True 段 [(起, 止)]"""
        runs, k = [], 0
        while k < len(mask):
            if mask[k]:
                j = k
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                if j - k + 1 >= min_len:
                    runs.append((k, j))
                k = j + 1
            else:
                k += 1
        return runs

    # ------------------------------------------------------------------ 签名
    @staticmethod
    def _ref_orientation(prof: pd.DataFrame, inverter_id: str) -> Optional[Tuple[float, float]]:
        """同机参考朝向 = (azimuth, tilt) 众数；档案没有朝向 → None（不做物理拟合）"""
        g = prof[(prof["inverter_id"] == inverter_id)].dropna(subset=["azimuth", "tilt"])
        if g.empty:
            return None
        pair = g.groupby(["azimuth", "tilt"]).size().idxmax()
        return float(pair[0]), float(pair[1])

    @staticmethod
    def _leading_flag_days(hist: List[Dict], col: str) -> int:
        """此前连续多少个晴好天该标记为 1（hist 日期倒序；非晴好天跳过）"""
        n = 0
        for r in hist:
            if r.get("weather_tag") != "clear":
                continue
            v = r.get(col)
            if v is None or pd.isna(v) or int(v) != 1:
                break
            n += 1
        return n

    def _signature(self, sid: str, f: Optional[pd.Series], p: pd.Series, b: pd.Series,
                   group_curves: Dict[str, np.ndarray], inv_daily: pd.DataFrame, th, cfg,
                   mount: str = "fixed", geo: Optional[Tuple] = None) -> Tuple[Signature, Optional[float], Optional[np.ndarray]]:
        """geo = (lat, ref_azimuth, ref_tilt, date, lon_offset_min)：有它才做朝向物理拟合；lon_offset 把时钟窗口平移到太阳时"""
        lon_off_slots = int(round((geo[4] if geo and len(geo) > 4 else 0.0) / cfg.slot_minutes))
        sig = Signature()
        if f is None:
            return sig, None, None
        sig.zero_all_day = bool(f["zero_all_day"])
        sig.stuck = bool(f["stuck"])
        sig.limited = bool(b.get("limited", 0))
        _sr = f.get("stale_run_min") if "stale_run_min" in f else None
        sig.stale_run_min = 0 if _sr is None or pd.isna(_sr) else int(_sr)        # v0.1 老库补列后为 NULL
        gn = b.get("group_noise")
        sig.group_noise = None if gn is None or pd.isna(gn) else float(gn)
        oc = b.get("overcast_dev")
        sig.overcast_dev = None if oc is None or pd.isna(oc) else float(oc)
        if sig.zero_all_day or sig.stuck:
            return sig, None, None
        key = b["group_key"] if b["group_level"] == "group" else p["inverter_id"]
        gcurve = group_curves.get(key)
        vec = np.array([np.nan if v is None else v for v in json.loads(f["i_profile_json"])], dtype=float)
        if gcurve is None or len(vec) != len(gcurve):
            return sig, None, None
        valid = (gcurve >= max(0.5, 0.15 * np.nanmax(gcurve))) & ~np.isnan(vec)
        if not inv_daily.empty and p["inverter_id"] in inv_daily.index:    # 剔除限发时段
            try:
                for s in json.loads(inv_daily.loc[p["inverter_id"], "limited_slots_json"] or "[]"):
                    if 0 <= s < len(valid):
                        valid[s] = False
            except Exception:
                pass
        if valid.sum() < 6:
            return sig, None, None
        ratio = np.full(len(vec), np.nan)
        ratio[valid] = vec[valid] / gcurve[valid]
        # ---- 间歇停机/掉线：群组在发电（≥ 当日峰值 30%）而本串整段为 0 → 不是遮挡（遮挡不会到 0），单独成签名
        # "整段为 0"按相对口径：≤ 群组同时段 3%（15 分钟数据插值到 5 分钟后，停机边沿的半小时槽均值不会精确为 0；遮挡不会低到 3%）
        zero_run = valid & (vec <= np.maximum(th.zero_current_a, th.outage_rel_zero * gcurve)) & (gcurve >= th.outage_group_ratio * np.nanmax(gcurve))
        out_slots = self._runs(zero_run, th.outage_min_slots)
        if out_slots:
            n_slots = sum(z - a + 1 for a, z in out_slots)
            sig.outage_hours = round(n_slots * cfg.slot_minutes / 60.0, 2)
            sig.notes.append("白天整段 0 输出 " + "、".join(f"{slot_label(a, cfg.slot_minutes)}-{slot_label(z + 1, cfg.slot_minutes)}" for a, z in out_slots)
                             + f"（共 {sig.outage_hours}h，同机其它串正常）")
            for a, z in out_slots:        # 停机段不参与凹陷/恒定偏差统计
                valid[a:z + 1] = False
            ratio[~valid] = np.nan
            if valid.sum() < 6:
                return sig, None, None
        sig.ratio_mean = float(np.nanmean(ratio))
        sig.ratio_std = float(np.nanstd(ratio))
        # ---- 跟踪电站：清晨/傍晚双侧偏低、中午正常 = 失去跟踪增益（卡滞）。固定电站不判（那是遮挡/朝向）
        n_slots = len(ratio); per_h = 60 // cfg.slot_minutes
        w = lambda h0, h1: ratio[max(0, int(h0 * per_h) + lon_off_slots):min(n_slots, int(h1 * per_h) + lon_off_slots)]   # 按太阳时平移
        am = w(7.5, 10.5); pm = w(14, 17); noon = w(11.5, 13.5)
        if np.sum(~np.isnan(am)) >= 2 and np.sum(~np.isnan(pm)) >= 2 and np.sum(~np.isnan(noon)) >= 2:
            sig.am_ratio, sig.pm_ratio = float(np.nanmedian(am)), float(np.nanmedian(pm))
            if mount == "tracker" and sig.am_ratio <= th.tracker_side_ratio and sig.pm_ratio <= th.tracker_side_ratio \
                    and float(np.nanmedian(noon)) >= 0.93 and abs(sig.am_ratio - sig.pm_ratio) < 0.15:
                sig.tracker_stuck = True
                return sig, None, ratio          # 不再按"凹陷"解释
        # 分时段凹陷：比值 < dip_ratio 的连续槽段
        low = np.where(valid, ratio < th.dip_ratio, False)
        runs, k = [], 0
        while k < len(low):
            if low[k]:
                j = k
                while j + 1 < len(low) and low[j + 1]:
                    j += 1
                runs.append((k, j))
                k = j + 1
            else:
                k += 1
        runs = [r for r in runs if r[1] - r[0] + 1 >= th.dip_min_slots]
        if runs:
            a, z = min(runs, key=lambda r: np.nanmin(ratio[r[0]:r[1] + 1]))
            inside = np.zeros(len(ratio), bool)
            inside[a:z + 1] = True
            outside = valid & ~inside
            sig.dip_depth = float(1 - np.nanmin(ratio[a:z + 1]))
            sig.dip_start, sig.dip_end = slot_label(a, cfg.slot_minutes), slot_label(z + 1, cfg.slot_minutes)
            sig.outside_dip_ratio = float(np.nanmean(ratio[outside])) if outside.sum() >= 3 else None
        # 恒定偏差 / 上下午倾斜（东西朝向差异的签名）
        slots = np.arange(len(ratio))
        am, pm = ratio[valid & (slots < 22 + lon_off_slots)], ratio[valid & (slots >= 28 + lon_off_slots)]
        skew = float(np.nanmean(am) - np.nanmean(pm)) if len(am) >= 2 and len(pm) >= 2 else None
        if skew is not None and abs(skew) > 0.10:
            sig.notes.append(f"上下午比值倾斜 {skew:+.3f}（{'上午高' if skew > 0 else '下午高'}，疑东西向差异）")
            # 物理校核：在朝向/倾角网格里找最像的曲线形状；若明显不是参考朝向且拟合得好 → 这是朝向差异不是遮挡
            if geo is not None:
                from ..solar import best_fit_orientation
                lat, ref_az, ref_tilt, dd = geo[0], geo[1], geo[2], geo[3]
                fit = best_fit_orientation(lat, dd, ratio, ref_az, ref_tilt, cfg.slot_minutes, lon_offset_min=geo[4] if len(geo) > 4 else 0.0)
                # 条件：拟合得好（rmse<0.05）、明显优于参考朝向（至少好 0.03）、朝向差 ≥30° 或倾角差 ≥10°（避免均匀低发被当成朝向）
                if fit is not None and fit[2] < 0.05 and fit[3] - fit[2] >= 0.03 \
                        and (abs(fit[0] - ref_az) >= 30 or abs(fit[1] - ref_tilt) >= 10):
                    sig.orientation_fit = f"方位 {fit[0]:.0f}° 倾角 {fit[1]:.0f}°（档案 {p.get('azimuth')}°/{p.get('tilt')}°，参考 {ref_az:.0f}°/{ref_tilt:.0f}°，拟合误差 {fit[2]:.3f}）"
                    sig.orientation_like = True
                    sig.dip_depth, sig.dip_start, sig.dip_end, sig.outside_dip_ratio = None, None, None, None   # 东西向的"下午低"不是遮挡
                    sig.notes.append("曲线形状最接近 " + sig.orientation_fit)
        sig.constant_offset = bool(sig.ratio_std < 0.03 and sig.ratio_mean <= 1 + th.normal_floor and not runs)
        dev = b.get("dev_today")
        if sig.overcast_dev is not None and dev is not None and not pd.isna(dev) and dev <= th.normal_floor:
            if abs(sig.overcast_dev) < th.orientation_overcast_abs and (sig.constant_offset or (skew is not None and abs(skew) > 0.10)):
                sig.orientation_like = True
        if sig.orientation_fit and not (dev is not None and not pd.isna(dev) and dev <= th.normal_floor):
            sig.orientation_like = False       # 中午偏差正常、只是形状像别的朝向：不报（可能是参考朝向选错了），留给复核员看注记
        return sig, skew, ratio

    # ------------------------------------------------------------------ 分级
    def _grade(self, b: pd.Series, sig: Signature, p: pd.Series, zero_days: int, stuck_days: int,
               weather: str, dip_days: int, th, tracker_days: int = 0) -> Tuple[Optional[str], str, List[str]]:
        notes: List[str] = []
        dev = b.get("dev_today")
        dev = None if dev is None or pd.isna(dev) else float(dev)
        p5, p3 = int(b.get("persist_days_5", 0)), int(b.get("persist_days_3", 0))
        # ---- A 故障（不依赖晴好天）
        if zero_days >= th.zero_days:
            notes.append(f"全天 0 电流已连续 {zero_days} 天（断线 / 未接 / 保险丝 / 接头）")
            return Grade.A.value, Cause.OPEN_CIRCUIT.value, notes
        if stuck_days >= th.zero_days:
            notes.append(f"电流钉死不动已连续 {stuck_days} 天（采样/通信异常，或逆变器显示冻结）")
            return Grade.A.value, Cause.DATA_ISSUE.value, notes
        if sig.voltage_offset is not None:
            notes.append(f"中午电压偏离同机中位 {sig.voltage_offset:+.1%}，连续 ≥{th.voltage_offset_days} 天（串长/MPPT 配置差异，先核档案）")
            return Grade.A.value, Cause.PROFILE_ERROR.value, notes
        # ---- 间歇停机/掉线（白天整段 0 输出而同机其它串正常；全天 0 走上面的 A 级断线逻辑）
        if sig.outage_hours > 0:
            notes.append(f"白天停机/掉线 {sig.outage_hours}h，近 7 日已出现 {sig.outage_days} 天（逆变器/采集器掉线、直流开关跳脱、通讯中断；不是遮挡——遮挡不会到 0）")
            if sig.outage_days >= th.fault_persist_days:
                return Grade.B.value, Cause.OUTAGE.value, notes
            return Grade.E.value, Cause.OUTAGE.value, notes
        # ---- 跟踪支架卡滞（只在晴好天判：阴天没有跟踪增益可比）
        if sig.tracker_stuck and weather == "clear":
            notes.append(f"跟踪增益丢失：清晨 {sig.am_ratio:.0%} / 傍晚 {sig.pm_ratio:.0%}，已连续 {tracker_days + 1} 个晴好天")
            if tracker_days + 1 >= th.fault_persist_days:
                return Grade.B.value, Cause.TRACKER.value, notes
            return Grade.E.value, Cause.TRACKER.value, notes
        # ---- 非晴好天只沿用已有持续低发判定
        if weather != "clear":
            if p5 >= th.fault_persist_days:
                notes.append(f"今日非晴好天（{weather}），沿用此前 {p5} 个晴好天的持续低发判定")
                return Grade.B.value, self._b_cause(sig), notes
            return None, "", notes
        # ---- 已备案遮挡：预期范围内不报
        if bool(p.get("known_shading")):
            exp = p.get("shading_expected_dev")
            exp = float(exp) if exp is not None and not pd.isna(exp) else 0.0
            worse = dev is not None and dev <= exp + th.archived_shading_extra
            if not worse:
                if (dev is not None and dev <= th.normal_floor) or sig.dip_depth is not None:
                    notes.append(f"已备案遮挡（{p.get('shading_note') or ''}），当前表现在备案预期内，不报警")
                    return Grade.ARCHIVED.value, Cause.SHADING.value, notes
                return None, "", notes
            notes.append(f"已备案遮挡串偏差 {dev:+.1%} 明显低于备案预期 {exp:+.1%}，疑新增问题")
        # ---- 朝向差异（物理拟合确认）：优先于凹陷解释
        if sig.orientation_like and sig.orientation_fit and dev is not None and dev <= th.normal_floor:
            notes.append(f"晴天偏差 {dev:+.1%}，曲线形状与档案朝向不符，最像 {sig.orientation_fit} → 核排布图/更正档案，不派现场")
            return Grade.C.value, Cause.ORIENTATION.value, notes
        # ---- 分时段凹陷（中午窗口可能完全正常，故独立判定）
        if sig.dip_depth is not None and sig.outside_dip_ratio is not None and sig.outside_dip_ratio >= th.dip_normal_outside:
            cause = Cause.BYPASS_DIODE.value if 0.25 <= sig.dip_depth <= 0.42 else Cause.SHADING.value
            notes.append(f"{sig.dip_start}-{sig.dip_end} 电流跌至群组 {1 - sig.dip_depth:.0%}，其余时段正常（{sig.outside_dip_ratio:.0%}）；"
                         f"同类凹陷已连续 {dip_days + 1} 个晴好天")
            if dip_days + 1 >= th.fault_persist_days:
                return Grade.B.value, cause, notes
            return Grade.E.value, cause, notes
        if dev is None:
            return None, "", notes
        if sig.orientation_like:
            notes.append(f"晴天偏差 {dev:+.1%}、阴天偏差 {sig.overcast_dev:+.1%} → 比例恒定且阴天消失，疑朝向/倾角差异（核排布图）")
            return Grade.C.value, Cause.ORIENTATION.value, notes
        if dev <= th.fault_floor and p5 >= th.fault_persist_days:
            notes.append(f"互比偏差 {dev:+.1%}，连续 {p5} 个晴好天 ≤{th.fault_floor:.0%}")
            return Grade.B.value, self._b_cause(sig), notes
        if dev <= th.normal_floor and p3 >= th.observe_persist_days:
            notes.append(f"互比偏差 {dev:+.1%}，连续 {p3} 个晴好天处于观察带（{th.fault_floor:.0%}~{th.normal_floor:.0%}）")
            return Grade.D.value, self._b_cause(sig), notes
        if dev <= th.normal_floor:
            notes.append(f"互比偏差 {dev:+.1%} 首次/短期超出正常噪声带，入观察表")
            return Grade.E.value, Cause.UNKNOWN.value, notes
        return None, "", notes

    @staticmethod
    def _b_cause(sig: Signature) -> str:
        if sig.inverter_level:
            return Cause.MPPT.value
        if sig.outage_hours > 0:
            return Cause.OUTAGE.value
        if sig.dip_depth is not None and (sig.outside_dip_ratio or 0) >= 0.9:
            return Cause.BYPASS_DIODE.value if 0.25 <= sig.dip_depth <= 0.42 else Cause.SHADING.value
        if sig.constant_offset:
            return Cause.DEGRADATION_DUST.value
        return Cause.UNKNOWN.value
