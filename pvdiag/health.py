"""健康自检（`pvdiag doctor` / MonitorAgent 每日自动跑）：
数据库完整性、磁盘、数据新鲜度、调度是否在跑、模型文件、配置自洽、覆盖层、输出目录可写。
每项 ok / warn / fail + 一句可操作的说明，写入 health_checks 表；fail 由 MonitorAgent 推送。
"""
from __future__ import annotations

import os
import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

from .config import Settings
from .storage import Store


def run_health_checks(settings: Settings, store: Store, run_date: date) -> List[Dict]:
    checks: List[Dict] = []
    add = lambda name, status, detail: checks.append({"name": name, "status": status, "detail": detail})

    # 数据库
    qc = store.quick_check()
    add("sqlite 完整性", "ok" if qc == "ok" else "fail", qc if qc != "ok" else "PRAGMA quick_check ok")
    try:
        wal = settings.db_path.with_name(settings.db_path.name + "-wal")
        wal_mb = wal.stat().st_size / 1e6 if wal.exists() else 0.0
        add("sqlite WAL 大小", "ok" if wal_mb < 512 else "warn", f"{wal_mb:.0f} MB" + ("，建议 `pvdiag doctor --vacuum`" if wal_mb >= 512 else ""))
    except Exception as e:  # pragma: no cover
        add("sqlite WAL 大小", "warn", str(e))

    # 磁盘
    try:
        usage = shutil.disk_usage(settings.data_dir)
        free_gb = usage.free / 1e9
        add("磁盘剩余", "ok" if free_gb > 5 else ("warn" if free_gb > 1 else "fail"), f"{free_gb:.1f} GB 可用（data_dir={settings.data_dir}）")
    except Exception as e:  # pragma: no cover
        add("磁盘剩余", "warn", str(e))

    # 输出目录可写
    for name, p in (("outbox 可写", settings.outbox_dir), ("reports 可写", settings.reports_dir), ("models 可写", settings.models_dir)):
        try:
            p.mkdir(parents=True, exist_ok=True)
            t = p / ".write_test"
            t.write_text("ok", encoding="utf-8")
            t.unlink()
            add(name, "ok", str(p))
        except Exception as e:
            add(name, "fail", f"{p}: {e}")

    # 数据新鲜度
    stations = store.stations()
    if stations.empty:
        add("电站档案", "warn", "无电站档案（先 pvdiag import-profiles 或 simulate）")
    for st in stations["station_id"]:
        ds = store.available_dates("string", st)
        if not ds:
            add(f"数据新鲜度 {st}", "warn", "尚无明细数据")
            continue
        lag = (run_date - date.fromisoformat(ds[-1])).days
        fails = int(store.get_state(f"ingest:{st}:failures", 0) or 0)
        status = "ok" if lag <= settings.ingest.stale_data_alert_days else ("warn" if lag <= 5 else "fail")
        add(f"数据新鲜度 {st}", status, f"最后数据 {ds[-1]}（时延 {lag} 天）" + (f"，连续拉取失败 {fails} 次" if fails else ""))

    # 调度是否在跑：最近一次完整流水线
    last = store.query("SELECT MAX(run_date) d FROM pipeline_runs WHERE agent='monitor'")
    last_d = last["d"].iloc[0] if not last.empty else None
    if last_d:
        gap = (run_date - date.fromisoformat(last_d)).days
        add("每日巡检调度", "ok" if gap <= 1 else ("warn" if gap <= 3 else "fail"), f"最近一次完整巡检 {last_d}" + (f"，已 {gap} 天未跑：检查 launchd/cron" if gap > 1 else ""))
    else:
        add("每日巡检调度", "warn", "尚未跑过完整巡检")
    # 只看当天最近一轮（以最后一次 profile/ingest 开跑为界），早上失败、修好重跑成功不算
    first = store.scalar("SELECT MAX(started_at) FROM pipeline_runs WHERE run_date=? AND agent IN ('ingest','profile')", (run_date.isoformat(),)) or "0000"
    failed = store.query("SELECT agent, message FROM pipeline_runs WHERE run_date=? AND status='fail' AND started_at>=?", (run_date.isoformat(), first))
    add("今日智能体", "ok" if failed.empty else "fail", "全部正常" if failed.empty else "失败：" + "；".join(f"{r.agent}: {r.message[:80]}" for r in failed.itertuples()))

    # 模型
    mr = store.query("SELECT model_version, path, trained_at FROM model_registry WHERE active=1")
    if mr.empty:
        add("分类模型", "ok", "规则代理（标注不足或未训练）")
    else:
        p = Path(mr["path"].iloc[0])
        add("分类模型", "ok" if p.exists() else "fail", f"{mr['model_version'].iloc[0]} 训练于 {mr['trained_at'].iloc[0][:10]}" + ("" if p.exists() else f"，模型文件缺失 {p}（将回退规则代理）"))
    live = store.get_state("model:live_agreement")
    if live:
        add("模型线上一致率", "ok" if live["model"] >= live["rule_proxy"] - 0.05 else "warn",
            f"模型 {live['model']:.0%} vs 规则 {live['rule_proxy']:.0%}（n={live['n']}）")

    # 配置自洽
    th = settings.thresholds
    cfg_ok = th.fault_floor < th.normal_floor < 0 and th.fault_persist_days >= 1 and 0 < settings.baseline.clear_day_ratio < 1
    add("阈值配置自洽", "ok" if cfg_ok else "fail", f"fault_floor={th.fault_floor} normal_floor={th.normal_floor} persist={th.fault_persist_days} noise_k={th.noise_k}")
    ov = store.query("SELECT param, value_json, source, updated_at FROM param_overrides")
    add("自调参覆盖层", "ok", "无覆盖（使用 config.yaml 口径）" if ov.empty else "；".join(f"{r.param}={r.value_json}({r.source})" for r in ov.itertuples()))
    gate = store.get_state("dispatch:gate") or {}
    add("派单门控", "ok", f"配置 {settings.dispatch.mode} / 当前 {gate.get('effective', 'draft')}（{gate.get('reason', '-')}）")

    # 落库
    now = datetime.now().isoformat()
    store.execute("DELETE FROM health_checks WHERE run_date=?", (run_date.isoformat(),))
    store.executemany("INSERT INTO health_checks (run_date, name, status, detail, created_at) VALUES (?,?,?,?,?)",
                      [(run_date.isoformat(), c["name"], c["status"], c["detail"], now) for c in checks])
    return checks


def summarize(checks: List[Dict]) -> Dict:
    return {"ok": sum(c["status"] == "ok" for c in checks), "warn": sum(c["status"] == "warn" for c in checks),
            "fail": sum(c["status"] == "fail" for c in checks), "failed": [c for c in checks if c["status"] == "fail"]}


def vacuum(store: Store) -> str:
    store._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store._conn.execute("VACUUM")
    return "已执行 wal_checkpoint(TRUNCATE) + VACUUM"
