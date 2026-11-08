"""本地 CSV / Parquet 适配器。

用途：
1. 对接已有资产（F:\\Hermes 华为北向 pull_range.py 等）落地的文件——不改旧脚本，把输出目录指过来即可
2. 合成数据 / 历史导出文件回灌

目录约定（可在 station.extra.csv 中覆盖列映射）：
    <root>/<station_id>/strings/*.csv|parquet     列: string_id, ts, current_a, voltage_v
    <root>/<station_id>/inverters/*.csv|parquet   列: inverter_id, ts, pac_kw, status
    <root>/<station_id>/irradiance/*.csv          列: ts, ghi_w_m2 (可选)

也支持"宽表"（华为导出常见）：每行一个逆变器时刻，列 pv1_i, pv1_u, ... pv36_i, pv36_u → 自动拆成长表。
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .base import FetchResult, PlatformAdapter, STRING_COLS, INVERTER_COLS

_PV_RE = re.compile(r"^pv(\d+)_([iu])$", re.I)


class CsvAdapter(PlatformAdapter):
    platform = "csv"

    def __init__(self, station_id: str, config: Dict[str, Any], state_get=None, state_set=None):
        super().__init__(station_id, config, state_get, state_set)
        self.root = Path(self.config.get("root", "./import")) / station_id
        self.colmap: Dict[str, str] = self.config.get("colmap", {})
        self.min_file_bytes = int(self.config.get("min_file_bytes", 0))   # 半成品文件校验（如 >4MB 判完整）

    # ------------------------------------------------------------------
    def list_devices(self) -> pd.DataFrame:
        inv = self._read_dir("inverters")
        if inv.empty:
            return pd.DataFrame(columns=["inverter_id", "name", "model", "rated_ac_kw", "n_strings"])
        ids = sorted(inv["inverter_id"].astype(str).unique())
        return pd.DataFrame({"inverter_id": ids, "name": ids, "model": "", "rated_ac_kw": None, "n_strings": None})

    def fetch(self, start: date, end: date) -> FetchResult:
        res = FetchResult()
        s = self._read_dir("strings")
        if not s.empty:
            s = self._filter(s, start, end)
            res.strings = self.normalize_strings(s)
        i = self._read_dir("inverters")
        if not i.empty:
            i = self._filter(i, start, end)
            res.inverters = self.normalize_inverters(i)
        irr = self._read_dir("irradiance")
        if not irr.empty and "ghi_w_m2" in irr.columns:
            irr = self._filter(irr, start, end)
            res.irradiance = irr[["ts", "ghi_w_m2"]]
        if res.strings.empty and res.inverters.empty:
            res.notes.append(f"{self.root} 下无可用文件")
        return res

    # ------------------------------------------------------------------
    def _read_dir(self, sub: str) -> pd.DataFrame:
        p = self.root / sub
        if not p.exists():
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for f in sorted(list(p.glob("*.csv")) + list(p.glob("*.parquet"))):
            if self.min_file_bytes and f.stat().st_size < self.min_file_bytes:
                continue  # 半成品文件
            df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
            df = df.rename(columns=self.colmap)
            if sub == "strings":
                df = self._maybe_melt_wide(df)
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["ts"] = pd.to_datetime(out["ts"], errors="coerce")
        return out.dropna(subset=["ts"])

    @staticmethod
    def _filter(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
        d = df["ts"].dt.date
        return df[(d >= start) & (d <= end)]

    @staticmethod
    def _maybe_melt_wide(df: pd.DataFrame) -> pd.DataFrame:
        """华为宽表 → 长表。宽表特征：有 inverter_id/dev_id 列与 pvN_i / pvN_u 列。"""
        pv_cols = [c for c in df.columns if _PV_RE.match(str(c))]
        if not pv_cols or "string_id" in df.columns:
            return df
        dev_col = "inverter_id" if "inverter_id" in df.columns else ("dev_id" if "dev_id" in df.columns else None)
        if dev_col is None:
            return df
        rows = []
        idx = sorted({int(_PV_RE.match(c).group(1)) for c in pv_cols})
        for n in idx:
            ci, cu = f"pv{n}_i", f"pv{n}_u"
            if ci not in df.columns:
                continue
            part = pd.DataFrame({
                "string_id": df[dev_col].astype(str) + f"-pv{n}",
                "ts": df["ts"],
                "current_a": df[ci],
                "voltage_v": df[cu] if cu in df.columns else None,
            })
            rows.append(part)
        return pd.concat(rows, ignore_index=True) if rows else df
