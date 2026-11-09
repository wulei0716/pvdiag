"""按站并行：基线与诊断都是"每站独立读库 → 算 → 写"的形状，站与站之间没有数据依赖，
万串级多站时把"算"的部分丢进进程池，主进程只负责写（SQLite 单写者，避免并发写锁）。

- 每个工作进程打开自己的 Store（只读），绝不复用父进程的 sqlite 连接
- 结果（DataFrame/字典列表）序列化回主进程，顺序与站列表一致，落库结果与串行完全相同
- workers=1、站数不足、或进程池起不来（受限容器）→ 自动退回串行
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
from datetime import date
from typing import Any, Callable, List, Sequence

from .config import Settings

log = logging.getLogger("pvdiag")

_WORKER_STORE = None
_WORKER_SETTINGS = None


def effective_workers(settings: Settings, n_stations: int) -> int:
    w = settings.runtime.workers
    env = os.environ.get("PVDIAG_WORKERS")
    if env and env.strip().isdigit():
        w = int(env)
    if w == 0:
        w = os.cpu_count() or 1
    if n_stations < settings.runtime.parallel_min_stations:
        return 1
    return max(1, min(w, n_stations))


def _init_worker(settings: Settings) -> None:
    global _WORKER_STORE, _WORKER_SETTINGS
    from .storage import Store
    _WORKER_SETTINGS = settings
    _WORKER_STORE = Store(settings.db_path, settings.samples_dir)


def _call(fn: Callable, station_id: str, d: date, extra: tuple) -> Any:
    """工作进程里跑 fn(settings, store, station_id, d, *extra)；异常原样带回主进程由调用方决定怎么告警"""
    from .agents.base import AgentContext
    ctx = AgentContext(settings=_WORKER_SETTINGS, store=_WORKER_STORE, run_date=d, station_ids=[station_id])
    return fn(ctx, station_id, d, *extra)


def map_stations(settings: Settings, ctx_serial, fn: Callable, station_ids: Sequence[str], d: date, *extra) -> List[Any]:
    """对每个站调用 fn(ctx, station_id, d, *extra)，返回与 station_ids 同序的结果列表。
    fn 必须只读库、不写（写由调用方在主进程做）；并行时 fn 需是模块级可 pickle 的函数或绑定方法。"""
    n = effective_workers(settings, len(station_ids))
    if n <= 1:
        return [fn(ctx_serial, sid, d, *extra) for sid in station_ids]
    try:
        # 不用 fork：父进程里 sqlite/pandas 可能已有线程，fork 有死锁风险；forkserver/spawn 干净地重新 import
        ctx_mp = mp.get_context("forkserver") if "forkserver" in mp.get_all_start_methods() else mp.get_context("spawn")
        with ctx_mp.Pool(processes=n, initializer=_init_worker, initargs=(settings,)) as pool:
            jobs = [pool.apply_async(_call, (fn, sid, d, tuple(extra))) for sid in station_ids]
            out = [j.get() for j in jobs]
        log.info("按站并行：%d 站 / %d 进程", len(station_ids), n)
        return out
    except Exception as e:        # 进程池不可用（受限容器/序列化失败）→ 串行兜底，结果一致只是慢
        log.warning("按站并行失败，退回串行: %s", e)
        return [fn(ctx_serial, sid, d, *extra) for sid in station_ids]
