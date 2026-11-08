"""阶段8 · 损失电量/电费核实

① 单串损失电量 = Σ晴好天 Σ时段 max(0, 基线电流 − 实际电流) × 串电压 × 时长
   基线电流(时段) = 群组同时段中位电流 × 该串正常期基线比；限发时段不计；停机/缺测不计
② 损失电费 = 损失电量 × 结算电价（峰谷平分时；站级可覆盖 EMC 协议电价）
③ 偏差起算日 = 偏差首次跌破 -3% 的晴好日（而非报警日），中间延迟就是"未挽回损失"
④ 台账：每串每日一行（幂等），月度报表 + 累计未修复损失（"钱在漏"看板，驱动派单优先级）
缺组件参数不影响电量（用实测 V/I）；缺电压的平台用 n_modules×Vmp 估算，再缺则只标"待量化"。
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..models import SuspectStatus
from .base import Agent, AgentContext, AgentResult

ACTIVE = (SuspectStatus.OPEN.value, SuspectStatus.REVIEWING.value, SuspectStatus.CONFIRMED.value,
          SuspectStatus.DISPATCHED.value)


class LossAgent(Agent):
    name = "loss"
    stage = "阶段8 损失量化"

    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        today = ctx.run_date.isoformat()
        susp = ctx.store.query(
            "SELECT * FROM suspects WHERE status IN (?,?,?,?) AND grade != '已备案' AND date=?", (*ACTIVE, today))
        rows: List[Dict] = []
        unquantified = 0
        for s in susp.to_dict("records"):
            onset = self.onset_date(s)
            have = set(ctx.store.query("SELECT date FROM loss_ledger WHERE string_id=? AND date<?",
                                       (s["string_id"] or f"inv:{s['inverter_id']}", today))["date"])
            d = max(onset, ctx.run_date - timedelta(days=60))
            while d <= ctx.run_date:
                if d.isoformat() in have:
                    d += timedelta(days=1)
                    continue
                r = (self._string_day_loss(ctx, s, d) if s["scope"] == "string" else self._inverter_day_loss(ctx, s, d))
                if r is not None:
                    rows.append(r)
                    unquantified += int(not r["quantified"])
                d += timedelta(days=1)
        if rows:
            ctx.store.upsert_df("loss_ledger", pd.DataFrame(rows))
        summary = self.ledger_summary(ctx.store)
        res.metrics = {"ledger_rows": len(rows), "unquantified_days": unquantified,
                       "open_loss_kwh": summary["totals"]["lost_kwh"], "open_loss_yuan": summary["totals"]["lost_yuan"]}
        res.artifacts = summary
        ctx.shared["loss_summary"] = summary
        res.summary = f"累计未修复损失 {summary['totals']['lost_kwh']:.0f} kWh ≈ {summary['totals']['lost_yuan']:.0f} 元（{len(summary['by_case'])} 个案例）"
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def onset_date(s: Dict) -> date:
        """案例起算日：suspect_id 后缀中的推定起始日（首次跌破 -3% 的晴好日 / 0 电流首日）"""
        tail = s["suspect_id"].rsplit("@", 1)[-1]
        try:
            return date.fromisoformat(tail)
        except ValueError:
            return date.fromisoformat(s.get("first_date") or s["date"])

    def _prices(self, ctx: AgentContext, station_id: str) -> Tuple[Dict[str, float], Dict[str, List[List[int]]]]:
        cfg = ctx.settings.loss
        tariff = {}
        row = ctx.store.query("SELECT tariff_json FROM stations WHERE station_id=?", (station_id,))
        if not row.empty and row["tariff_json"].iloc[0]:
            try:
                tariff = json.loads(row["tariff_json"].iloc[0])
            except Exception:
                tariff = {}
        if "price" in tariff:        # 单一电价（EMC 协议价/上网价）
            p = float(tariff["price"])
            return {"peak": p, "flat": p, "valley": p}, cfg.tou
        return {**cfg.tou_price, **tariff.get("tou_price", {})}, tariff.get("tou", cfg.tou)

    @staticmethod
    def _price_for_slot(slot: int, slot_minutes: int, prices: Dict[str, float], tou: Dict[str, List[List[int]]], default: float) -> float:
        h = slot * slot_minutes / 60.0
        for k, ranges in tou.items():
            for a, b in ranges:
                if a <= h < b:
                    return prices.get(k, default)
        return default

    def _string_day_loss(self, ctx: AgentContext, s: Dict, d: date) -> Optional[Dict]:
        cfg = ctx.settings.baseline
        sid, today = s["string_id"], d.isoformat()
        w = ctx.store.query("SELECT weather_tag FROM station_daily WHERE station_id=? AND date=?", (s["station_id"], today))
        if w.empty or w["weather_tag"].iloc[0] != "clear":
            return None                                  # 口径：只累计晴好天
        f = ctx.store.query("SELECT * FROM daily_features WHERE string_id=? AND date=?", (sid, today))
        b = ctx.store.query("SELECT * FROM baselines WHERE string_id=? AND date=?", (sid, today))
        if f.empty:
            return None
        f = f.iloc[0]
        base = {"string_id": sid, "date": today, "suspect_id": s["suspect_id"], "lost_kwh": 0.0, "lost_yuan": 0.0, "quantified": 1}
        if f["stuck"]:
            return {**base, "quantified": 0}             # 数据问题：实际发电未知 → 待量化
        # 群组曲线
        grp_key, level = (b["group_key"].iloc[0], b["group_level"].iloc[0]) if not b.empty else (None, "none")
        if level == "none" or grp_key is None:
            return {**base, "quantified": 0}
        peers = ctx.store.query(
            "SELECT f.i_profile_json, f.v_profile_json FROM daily_features f JOIN strings p ON p.string_id=f.string_id "
            "WHERE f.date=? AND f.string_id!=? AND f.zero_all_day=0 AND f.stuck=0 AND f.noon_median_i>=0.5 AND "
            + ("p.group_key=?" if level == "group" else "p.inverter_id=?"),
            (today, sid, grp_key if level == "group" else s["inverter_id"]))
        if len(peers) < 3:
            return {**base, "quantified": 0}
        I = np.vstack([np.array([np.nan if v is None else v for v in json.loads(x)], dtype=float) for x in peers["i_profile_json"]])
        V = np.vstack([np.array([np.nan if v is None else v for v in json.loads(x)], dtype=float) for x in peers["v_profile_json"]])
        g_i, g_v = np.nanmedian(I, axis=0), np.nanmedian(V, axis=0)
        actual = np.array([np.nan if v is None else v for v in json.loads(f["i_profile_json"])], dtype=float)
        ratio = float(b["baseline_ratio"].iloc[0]) if (not b.empty and pd.notna(b["baseline_ratio"].iloc[0]) and not b["cold_start"].iloc[0]) else 1.0
        ratio = min(max(ratio, 0.9), 1.1)
        limited = set()
        inv = ctx.store.query("SELECT limited_slots_json FROM inverter_daily WHERE inverter_id=? AND date=?", (s["inverter_id"], today))
        if not inv.empty and inv["limited_slots_json"].iloc[0]:
            limited = set(json.loads(inv["limited_slots_json"].iloc[0]))
        prices, tou = self._prices(ctx, s["station_id"])
        default_price = ctx.settings.loss.default_price_yuan_per_kwh
        kwh = yuan = 0.0
        for k in range(len(g_i)):
            if k in limited or np.isnan(g_i[k]) or g_i[k] < 0.3 or np.isnan(g_v[k]):
                continue
            a = 0.0 if np.isnan(actual[k]) else actual[k]
            gap = max(0.0, g_i[k] * ratio - a)
            e = gap * g_v[k] * (cfg.slot_minutes / 60.0) / 1000.0
            kwh += e
            yuan += e * self._price_for_slot(k, cfg.slot_minutes, prices, tou, default_price)
        return {**base, "lost_kwh": round(kwh, 3), "lost_yuan": round(yuan, 2)}

    def _inverter_day_loss(self, ctx: AgentContext, s: Dict, d: date) -> Optional[Dict]:
        today = d.isoformat()
        ib = ctx.store.query("SELECT inv_dev, weather_tag FROM inverter_baselines WHERE inverter_id=? AND date=?", (s["inverter_id"], today))
        if ib.empty or ib["weather_tag"].iloc[0] != "clear" or pd.isna(ib["inv_dev"].iloc[0]) or ib["inv_dev"].iloc[0] >= 0:
            return None
        dev = float(ib["inv_dev"].iloc[0])
        e = ctx.store.scalar("SELECT SUM(f.day_energy_kwh) FROM daily_features f JOIN strings p ON p.string_id=f.string_id "
                             "WHERE p.inverter_id=? AND f.date=?", (s["inverter_id"], today)) or 0.0
        lost = float(e) * (-dev / (1 + dev))
        prices, _ = self._prices(ctx, s["station_id"])
        price = prices.get("flat", ctx.settings.loss.default_price_yuan_per_kwh)
        return {"string_id": f"inv:{s['inverter_id']}", "date": today, "suspect_id": s["suspect_id"],
                "lost_kwh": round(lost, 3), "lost_yuan": round(lost * price, 2), "quantified": 1}

    # ------------------------------------------------------------------ 台账
    @staticmethod
    def ledger_summary(store) -> Dict:
        df = store.query(
            "SELECT l.suspect_id, l.string_id, s.station_id, s.grade, s.rule_cause, s.status, "
            "SUM(l.lost_kwh) lost_kwh, SUM(l.lost_yuan) lost_yuan, MIN(l.date) since, MAX(l.date) until, "
            "COUNT(*) days, SUM(1-l.quantified) unquantified_days "
            "FROM loss_ledger l JOIN suspects s ON s.suspect_id=l.suspect_id GROUP BY l.suspect_id ORDER BY lost_yuan DESC")
        open_df = df[df["status"].isin(ACTIVE)] if not df.empty else df
        by_station = open_df.groupby("station_id")[["lost_kwh", "lost_yuan"]].sum().round(1).to_dict("index") if not open_df.empty else {}
        monthly = store.query("SELECT substr(date,1,7) month, SUM(lost_kwh) lost_kwh, SUM(lost_yuan) lost_yuan FROM loss_ledger GROUP BY month ORDER BY month")
        return {
            "totals": {"lost_kwh": float(open_df["lost_kwh"].sum()) if not open_df.empty else 0.0,
                       "lost_yuan": float(open_df["lost_yuan"].sum()) if not open_df.empty else 0.0},
            "by_station": by_station,
            "by_case": open_df.round(2).to_dict("records") if not open_df.empty else [],
            "monthly": monthly.round(1).to_dict("records") if not monthly.empty else [],
        }

    @staticmethod
    def case_loss(store, suspect_id: str) -> Dict[str, float]:
        r = store.query("SELECT SUM(lost_kwh) k, SUM(lost_yuan) y, COUNT(*) n FROM loss_ledger WHERE suspect_id=?", (suspect_id,))
        if r.empty or pd.isna(r["k"].iloc[0]):
            return {"lost_kwh": 0.0, "lost_yuan": 0.0, "days": 0, "per_day_kwh": 0.0}
        n = int(r["n"].iloc[0]) or 1
        return {"lost_kwh": float(r["k"].iloc[0]), "lost_yuan": float(r["y"].iloc[0]), "days": n,
                "per_day_kwh": float(r["k"].iloc[0]) / n}
