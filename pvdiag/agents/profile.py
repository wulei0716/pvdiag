"""阶段2 · 组串档案智能体（AI 归因的地基）

① 每串建档：所属逆变器/MPPT、串长、组件型号、朝向、倾角、屋面位置、是否已知遮挡
② 接线变更/组件更换/清洗维修事件记录（驱动基线分段重置）
③ 配置差异串分组：group_key = 逆变器|串长|朝向桶|倾角桶|组件型号 —— 不同配置不混合互比
④ 档案校验：实测中午电压 vs 群组中位电压（串长错 → 电压异位 → 系统性误报的最大来源）
⑤ 完整度报告：缺组件参数的站只做"异常判定"不做"金额量化"
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..models import EventType
from .base import Agent, AgentContext, AgentResult


def make_group_key(row: pd.Series, az_bucket: float = 10.0, tilt_bucket: float = 5.0) -> str:
    def b(v, w):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "na"
        return str(int(round(float(v) / w) * w))
    nm = row.get("n_modules")
    nm = "na" if nm is None or (isinstance(nm, float) and np.isnan(nm)) else str(int(nm))
    return f"{row['inverter_id']}|n{nm}|az{b(row.get('azimuth'), az_bucket)}|t{b(row.get('tilt'), tilt_bucket)}|{row.get('module_model') or ''}"


class ProfileAgent(Agent):
    name = "profile"
    stage = "阶段2 组串档案"

    # ------------------------------------------------------------------ 导入
    @staticmethod
    def import_stations(store, df: pd.DataFrame) -> int:
        df = df.copy()
        for c in ("tariff", "extra"):
            if c in df.columns:
                df[f"{c}_json"] = df[c].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, dict) else (v if isinstance(v, str) else "{}"))
        for c in ("tariff_json", "extra_json"):
            if c not in df.columns:
                df[c] = "{}"
        return store.upsert_df("stations", df)

    @staticmethod
    def import_inverters(store, df: pd.DataFrame) -> int:
        df = df.copy()
        if "extra_json" not in df.columns:
            df["extra_json"] = "{}"
        return store.upsert_df("inverters", df)

    @staticmethod
    def import_strings(store, df: pd.DataFrame) -> int:
        df = df.copy()
        if "known_shading" in df.columns:
            df["known_shading"] = df["known_shading"].fillna(0).astype(int)
        else:
            df["known_shading"] = 0
        for c in ("shading_expected_dev", "shading_note", "notes", "module_model", "roof_area"):
            if c not in df.columns:
                df[c] = None
        # 设备类型：string（组串，默认）| optimizer（组件级优化器）| microinverter（微逆）；组件：mono（默认）| bifacial（双面）
        df["device_type"] = df["device_type"].fillna("string") if "device_type" in df.columns else "string"
        df["module_type"] = df["module_type"].fillna("mono") if "module_type" in df.columns else "mono"
        # 并联串数：一路 MPPT 接了两串并联时填 2（电流翻倍不是别人低）；默认 1
        df["n_parallel"] = pd.to_numeric(df["n_parallel"], errors="coerce").fillna(1).clip(lower=1).astype(int) if "n_parallel" in df.columns else 1
        df["group_key"] = df.apply(make_group_key, axis=1)
        df["updated_at"] = datetime.now().isoformat()
        return store.upsert_df("strings", df)

    @staticmethod
    def add_event(store, scope: str, target_id: str, event_type: str, event_date: date | str, notes: str = "") -> None:
        EventType(event_type)  # 校验枚举
        store.execute("INSERT INTO events (scope, target_id, event_type, event_date, notes, created_at) VALUES (?,?,?,?,?,?)",
                      (scope, target_id, event_type, str(event_date)[:10], notes, datetime.now().isoformat()))

    @staticmethod
    def segment_start_for(string_row: pd.Series, events: pd.DataFrame, upto: date) -> Optional[str]:
        """该串当前基线分段起点 = 影响该串的最近一次事件日期（不含备案类事件）"""
        if events.empty:
            return None
        resetting = events[events["event_type"].isin([e.value for e in EventType if e != EventType.SHADING_REGISTERED])]
        hit = resetting[
            ((resetting["scope"] == "string") & (resetting["target_id"] == string_row["string_id"])) |
            ((resetting["scope"] == "inverter") & (resetting["target_id"] == string_row["inverter_id"])) |
            ((resetting["scope"] == "station") & (resetting["target_id"] == string_row["station_id"]))]
        hit = hit[hit["event_date"] <= upto.isoformat()]
        if hit.empty:
            return None
        return str(hit["event_date"].max())

    # ------------------------------------------------------------------ 每日运行：校验与完整度
    def run(self, ctx: AgentContext) -> AgentResult:
        res = AgentResult(agent=self.name)
        strings = ctx.store.strings()
        if strings.empty:
            res.status, res.summary = "skipped", "无组串档案"
            return res
        # 重算 group_key（档案可能被人工编辑过）
        strings["group_key"] = strings.apply(make_group_key, axis=1)
        ctx.store.upsert_df("strings", strings)

        issues: List[Dict] = []
        # ① 电压校验：串长档案 vs 实测中午电压（同机互比，>8% 视为疑错）
        feats = ctx.store.query("SELECT string_id, noon_median_v FROM daily_features WHERE date=?", (ctx.run_date.isoformat(),))
        if not feats.empty:
            m = strings.merge(feats, on="string_id", how="inner")
            m = m[m["noon_median_v"].notna() & (m["noon_median_v"] > 0)]
            if "device_type" in m.columns:
                m = m[m["device_type"].fillna("string") == "string"]       # 优化器/微逆的电压不可互比
            for inv, g in m.groupby("inverter_id"):
                if len(g) < 3:
                    continue
                med = g["noon_median_v"].median()
                for r in g.itertuples():
                    off = r.noon_median_v / med - 1
                    if abs(off) > ctx.settings.thresholds.voltage_offset:
                        issues.append({"string_id": r.string_id, "issue": "电压异位",
                                       "detail": f"中午电压 {r.noon_median_v:.0f}V 偏离同机中位 {med:.0f}V {off:+.1%}，核对串长/MPPT配置"})
        # ② 完整度：缺组件参数 → 不做金额量化
        missing_params = strings[strings["n_modules"].isna() | strings["module_wp"].isna()]
        by_station = missing_params.groupby("station_id").size().to_dict() if not missing_params.empty else {}
        # ③ 小群组：互比样本不足
        small = strings.groupby("group_key").size()
        small = small[small < ctx.settings.baseline.min_group_size]
        res.metrics = {"n_strings": int(len(strings)), "n_groups": int(strings["group_key"].nunique()),
                       "voltage_issues": len(issues), "missing_param_strings": int(len(missing_params)),
                       "small_groups": int(len(small))}
        res.artifacts = {"voltage_issues": issues, "missing_params_by_station": by_station,
                         "small_groups": small.index.tolist()}
        ctx.shared["profile_issues"] = issues
        if by_station:
            res.alerts.append(f"缺组件参数组串: {by_station}（这些串只判异常不量化金额）")
        res.summary = f"{len(strings)} 串 / {strings['group_key'].nunique()} 个互比群组；电压异位 {len(issues)}；缺参 {len(missing_params)}"
        return res

    # ------------------------------------------------------------------ 从目录批量导入
    @classmethod
    def import_bundle(cls, store, folder: Path | str) -> Dict[str, int]:
        folder = Path(folder)
        out = {}
        if (folder / "stations.csv").exists():
            out["stations"] = cls.import_stations(store, pd.read_csv(folder / "stations.csv"))
        if (folder / "inverters.csv").exists():
            out["inverters"] = cls.import_inverters(store, pd.read_csv(folder / "inverters.csv"))
        if (folder / "strings.csv").exists():
            out["strings"] = cls.import_strings(store, pd.read_csv(folder / "strings.csv"))
        if (folder / "events.csv").exists():
            ev = pd.read_csv(folder / "events.csv")
            for r in ev.itertuples():
                cls.add_event(store, r.scope, r.target_id, r.event_type, r.event_date, getattr(r, "notes", "") or "")
            out["events"] = len(ev)
        return out
