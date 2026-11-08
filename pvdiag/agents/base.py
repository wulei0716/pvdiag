"""智能体基类：统一的上下文、运行日志、指标与可追溯输出。"""
from __future__ import annotations

import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from ..config import Settings
from ..storage import Store


@dataclass
class AgentResult:
    agent: str
    status: str = "ok"                     # ok | warn | fail | skipped
    summary: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)   # 供下游/报告消费的结构化输出
    alerts: List[str] = field(default_factory=list)


@dataclass
class AgentContext:
    settings: Settings
    store: Store
    run_date: date
    station_ids: Optional[List[str]] = None   # None → 全部电站
    shared: Dict[str, Any] = field(default_factory=dict)  # 智能体之间传递的中间产物
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("pvdiag"))

    def stations(self) -> List[str]:
        if self.station_ids:
            return list(self.station_ids)
        return self.store.stations()["station_id"].tolist()


class Agent(ABC):
    name: str = "agent"
    stage: str = ""

    @abstractmethod
    def run(self, ctx: AgentContext) -> AgentResult: ...

    def __call__(self, ctx: AgentContext) -> AgentResult:
        started = datetime.now()
        ctx.logger.info("▶ %s (%s) run_date=%s", self.name, self.stage, ctx.run_date)
        try:
            res = self.run(ctx)
        except Exception as e:  # 单个智能体失败不拖垮整条流水线，但必须留痕告警
            ctx.logger.exception("✖ %s failed", self.name)
            res = AgentResult(agent=self.name, status="fail", summary=f"{type(e).__name__}: {e}",
                              artifacts={"traceback": traceback.format_exc()})
            ctx.store.alert(ctx.run_date, "error", self.name, res.summary)
        for a in res.alerts:
            ctx.store.alert(ctx.run_date, "warn", self.name, a)
        ctx.store.log_run(ctx.run_date, self.name, res.status, started, res.metrics, res.summary)
        ctx.shared[self.name] = res
        ctx.logger.info("■ %s → %s %s", self.name, res.status, res.summary)
        return res
