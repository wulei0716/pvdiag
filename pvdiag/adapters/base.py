"""平台适配器接口：三家云平台 + 本地 CSV，全部输出统一列。

统一输出：
- 组串 5 分钟明细：string_id, ts, current_a, voltage_v
- 逆变器 5 分钟明细：inverter_id, ts, pac_kw, status
- 设备清单：inverter_id, name, model, rated_ac_kw, n_strings
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

STRING_COLS = ["string_id", "ts", "current_a", "voltage_v"]
INVERTER_COLS = ["inverter_id", "ts", "pac_kw", "status"]


@dataclass
class FetchResult:
    strings: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=STRING_COLS))
    inverters: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=INVERTER_COLS))
    irradiance: Optional[pd.DataFrame] = None     # 可选：ts, ghi_w_m2（有辐照仪/卫星数据时）
    calls_used: int = 0
    notes: List[str] = field(default_factory=list)


class PlatformAdapter(ABC):
    platform: str = "base"

    def __init__(self, station_id: str, config: Dict[str, Any], state_get=None, state_set=None):
        self.station_id = station_id
        self.config = config or {}
        self._state_get = state_get or (lambda k, d=None: d)
        self._state_set = state_set or (lambda k, v: None)

    @abstractmethod
    def list_devices(self) -> pd.DataFrame: ...

    @abstractmethod
    def fetch(self, start: date, end: date) -> FetchResult:
        """拉取 [start, end] 闭区间的 5 分钟数据。实现方负责分批、限流与 token 续期。"""

    # ---- 通用工具 ----
    @staticmethod
    def date_batches(start: date, end: date, batch_days: int) -> List[tuple[date, date]]:
        out, cur = [], start
        while cur <= end:
            nxt = min(cur + timedelta(days=batch_days - 1), end)
            out.append((cur, nxt))
            cur = nxt + timedelta(days=1)
        return out

    @staticmethod
    def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=STRING_COLS)
        df = df[STRING_COLS].copy()
        df["ts"] = pd.to_datetime(df["ts"]).dt.floor("5min")   # 时钟漂移对齐到 5 分钟格
        df["current_a"] = pd.to_numeric(df["current_a"], errors="coerce")
        df["voltage_v"] = pd.to_numeric(df["voltage_v"], errors="coerce")
        df = df.dropna(subset=["ts"]).drop_duplicates(subset=["string_id", "ts"], keep="last")
        return df.sort_values(["string_id", "ts"]).reset_index(drop=True)

    @staticmethod
    def normalize_inverters(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=INVERTER_COLS)
        df = df.reindex(columns=INVERTER_COLS).copy()
        df["ts"] = pd.to_datetime(df["ts"]).dt.floor("5min")
        df["pac_kw"] = pd.to_numeric(df["pac_kw"], errors="coerce")
        df = df.dropna(subset=["ts"]).drop_duplicates(subset=["inverter_id", "ts"], keep="last")
        return df.sort_values(["inverter_id", "ts"]).reset_index(drop=True)


def build_adapter(platform: str, station_id: str, config: Dict[str, Any], store=None) -> PlatformAdapter:
    from .csv_adapter import CsvAdapter
    from .huawei import HuaweiAdapter
    from .ginlong import GinlongAdapter
    from .sungrow import SungrowAdapter

    cls = {"csv": CsvAdapter, "huawei": HuaweiAdapter, "ginlong": GinlongAdapter, "sungrow": SungrowAdapter}.get(platform)
    if cls is None:
        raise ValueError(f"未知平台: {platform}")
    sg = store.get_state if store else None
    ss = store.set_state if store else None
    return cls(station_id, config, sg, ss)
