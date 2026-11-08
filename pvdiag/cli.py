"""pvdiag 命令行：

  pvdiag init --data-dir ./data                         生成 config.yaml 与数据库
  pvdiag simulate --out ./demo --days 60 --load          合成电站数据（含注入故障）并直接入库
  pvdiag import-profiles ./import                        导入 stations/inverters/strings/events.csv
  pvdiag ingest --root ./import --station ST01           CSV/平台拉数（按电站平台类型自动选适配器）
  pvdiag run --date 2026-10-30 [--no-ingest]             跑一次每日闭环
  pvdiag backfill --start 2026-10-01 --end 2026-10-30    历史回放
  pvdiag review list|confirm|reject|batch                复核操作
  pvdiag wo list|show|send|close                         工单操作
  pvdiag event add --scope string --target X --type cleaning --date 2026-10-20
  pvdiag train                                           用标注训练/重训分类模型
  pvdiag report --date 2026-10-30                        打印当日 Markdown 报告
  pvdiag serve --port 8080                               人工复核 Web 界面
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .config import Settings
from .storage import Store


def _settings(args) -> Settings:
    s = Settings.load(args.config)
    if getattr(args, "data_dir", None):
        s.data_dir = Path(args.data_dir)
    s.ensure_dirs()
    return s


def _date(s: str | None) -> date:
    return date.fromisoformat(s) if s else date.today()


def cmd_init(args):
    s = Settings(data_dir=Path(args.data_dir))
    s.ensure_dirs()
    s.dump(args.config)
    Store(s.db_path, s.samples_dir)
    print(f"已生成 {args.config} 与数据库 {s.db_path}")


def cmd_simulate(args):
    from .sim import SimConfig, simulate
    from .sim.synth import write_csv_bundle
    out = Path(args.out)
    bundles = []
    for k in range(args.stations):
        cfg = SimConfig(station_id=f"ST{k + 1:02d}", station_name=f"合成电站{k + 1:02d}", days=args.days, seed=7 + k,
                        start=_date(args.start))
        data = simulate(cfg)
        write_csv_bundle(out, data, cfg.station_id)
        bundles.append(data)
        print(f"{cfg.station_id}: {len(data['samples_strings'])} 行组串明细，{len(data['truth'])} 个注入故障 → {out}")
    # 合并档案文件
    for name in ("stations", "inverters", "strings", "truth"):
        pd.concat([b[name] for b in bundles], ignore_index=True).to_csv(out / f"{name}.csv", index=False)
    if args.load:
        s = _settings(args)
        store = Store(s.db_path, s.samples_dir)
        from .agents.profile import ProfileAgent
        from .agents.ingest import IngestAgent
        from .agents.base import AgentContext
        for b in bundles:
            st = b["stations"].copy()
            st["extra_json"] = json.dumps({"platform_config": {"root": str(out.resolve())}}, ensure_ascii=False)
            ProfileAgent.import_stations(store, st)
            ProfileAgent.import_inverters(store, b["inverters"])
            ProfileAgent.import_strings(store, b["strings"])
            ctx = AgentContext(settings=s, store=store, run_date=date.today())
            n = IngestAgent().ingest_frames(ctx, b["stations"]["station_id"].iloc[0], b["samples_strings"], b["samples_inverters"])
            print(f"入库 {b['stations']['station_id'].iloc[0]}: {n} 行，特征已计算")


def cmd_import_profiles(args):
    from .agents.profile import ProfileAgent
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    print(ProfileAgent.import_bundle(store, args.folder))


def cmd_ingest(args):
    from .agents.ingest import IngestAgent
    from .agents.base import AgentContext
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    overrides = {"csv": {"root": args.root}} if args.root else {}
    end = _date(args.end)
    start = _date(args.start) if args.start else end
    ctx = AgentContext(settings=s, store=store, run_date=end, station_ids=[args.station] if args.station else None)
    res = IngestAgent(backfill_days=(end - start).days, platform_overrides=overrides)(ctx)
    print(res.summary, res.alerts)


def cmd_run(args):
    from .orchestrator import DailyPipeline
    s = _settings(args)
    p = DailyPipeline(s, ingest=not args.no_ingest, backfill_days=args.backfill_days)
    rep = p.run(_date(args.date), station_ids=[args.station] if args.station else None)
    print(f"报告：{s.reports_dir / rep['date'] / 'report.md'}")
    for k, v in rep["agents"].items():
        print(f"  {v['status']:7s} {k:10s} {v['summary']}")


def cmd_backfill(args):
    from .orchestrator import DailyPipeline
    s = _settings(args)
    p = DailyPipeline(s, ingest=not args.no_ingest)
    reps = p.backfill(_date(args.start), _date(args.end), station_ids=[args.station] if args.station else None)
    print(f"回放完成，最终报告：{s.reports_dir / reps[-1]['date'] / 'report.md'}" if reps else "无输出")


def cmd_review(args):
    from .agents.review import ReviewAgent
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 60)
    if args.action == "list":
        q = ReviewAgent.queue(store, args.station)
        cols = ["suspect_id", "grade", "rule_cause", "ai_cause", "confidence", "dev_today", "persist_days", "first_date", "action", "status"]
        print(q[[c for c in cols if c in q.columns]].to_string(index=False) if not q.empty else "复核队列为空")
    elif args.action == "confirm":
        ReviewAgent.confirm(store, args.id, args.cause, args.reviewer, args.note or "")
        print("已确认")
    elif args.action == "reject":
        ReviewAgent.reject(store, args.id, args.reason or "其他", args.reviewer, args.note or "")
        print("已驳回（误报）")
    elif args.action == "batch":
        print(ReviewAgent.weekly_batch(store, args.station).to_string(index=False))
    elif args.action == "stats":
        print(json.dumps(ReviewAgent.stats(store), ensure_ascii=False, indent=2))


def cmd_wo(args):
    from .agents.workorder import WorkOrderAgent
    from .agents.base import AgentContext
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    ctx = AgentContext(settings=s, store=store, run_date=date.today())
    if args.action == "list":
        df = store.query("SELECT wo_id, station_id, roof_area, cause, priority, loss_kwh_per_day, loss_yuan_total, status, created_at FROM workorders ORDER BY created_at DESC")
        print(df.to_string(index=False) if not df.empty else "无工单")
    elif args.action == "show":
        wo = store.query("SELECT * FROM workorders WHERE wo_id=?", (args.id,))
        print(WorkOrderAgent.render(store, wo.iloc[0].to_dict()) if not wo.empty else "不存在")
    elif args.action == "send":
        print(WorkOrderAgent().send(ctx, args.id))
    elif args.action == "close":
        WorkOrderAgent.close(store, args.id, args.cause or "未填", args.field_action or "维修", args.note or "",
                             done_date=_date(args.date), recheck_clear_days=s.dispatch.recheck_clear_days)
        print("已回填，等待复查")


def cmd_event(args):
    from .agents.profile import ProfileAgent
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    ProfileAgent.add_event(store, args.scope, args.target, args.type, _date(args.date), args.notes or "")
    print("事件已登记（基线将从该日分段重置）")


def cmd_train(args):
    from .agents.classify import ClassifyAgent
    from .agents.base import AgentContext
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    ctx = AgentContext(settings=s, store=store, run_date=date.today())
    print(json.dumps(ClassifyAgent().train(ctx, force=args.force), ensure_ascii=False, indent=2, default=str))


def cmd_report(args):
    s = _settings(args)
    p = s.reports_dir / _date(args.date).isoformat() / "report.md"
    print(p.read_text(encoding="utf-8") if p.exists() else f"无报告 {p}")


def cmd_serve(args):
    s = _settings(args)
    from .web.app import create_app
    import uvicorn
    uvicorn.run(create_app(s), host=args.host, port=args.port)


def main(argv=None):
    # 全局选项既可放在子命令前，也可放在子命令后
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--data-dir", default=argparse.SUPPRESS)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    ap = argparse.ArgumentParser("pvdiag", description="组串发电量异常诊断智能体系统", parents=[common])
    ap.set_defaults(config="config.yaml", data_dir=None, verbose=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    add = lambda name: sub.add_parser(name, parents=[common])

    p = add("init"); p.set_defaults(fn=cmd_init)
    p = add("simulate"); p.add_argument("--out", default="./demo"); p.add_argument("--days", type=int, default=60)
    p.add_argument("--stations", type=int, default=1); p.add_argument("--start", default="2026-09-01"); p.add_argument("--load", action="store_true")
    p.set_defaults(fn=cmd_simulate)
    p = add("import-profiles"); p.add_argument("folder"); p.set_defaults(fn=cmd_import_profiles)
    p = add("ingest"); p.add_argument("--root"); p.add_argument("--station"); p.add_argument("--start"); p.add_argument("--end")
    p.set_defaults(fn=cmd_ingest)
    p = add("run"); p.add_argument("--date"); p.add_argument("--station"); p.add_argument("--no-ingest", action="store_true")
    p.add_argument("--backfill-days", type=int, default=0); p.set_defaults(fn=cmd_run)
    p = add("backfill"); p.add_argument("--start", required=True); p.add_argument("--end", required=True)
    p.add_argument("--station"); p.add_argument("--no-ingest", action="store_true"); p.set_defaults(fn=cmd_backfill)
    p = add("review"); p.add_argument("action", choices=["list", "confirm", "reject", "batch", "stats"])
    p.add_argument("--id"); p.add_argument("--cause"); p.add_argument("--reason"); p.add_argument("--note"); p.add_argument("--station")
    p.add_argument("--reviewer", default="reviewer"); p.set_defaults(fn=cmd_review)
    p = add("wo"); p.add_argument("action", choices=["list", "show", "send", "close"]); p.add_argument("--id")
    p.add_argument("--cause"); p.add_argument("--field-action"); p.add_argument("--note"); p.add_argument("--date"); p.set_defaults(fn=cmd_wo)
    p = add("event"); p.add_argument("action", choices=["add"]); p.add_argument("--scope", required=True)
    p.add_argument("--target", required=True); p.add_argument("--type", required=True); p.add_argument("--date", required=True)
    p.add_argument("--notes"); p.set_defaults(fn=cmd_event)
    p = add("train"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_train)
    p = add("report"); p.add_argument("--date"); p.set_defaults(fn=cmd_report)
    p = add("serve"); p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=8080); p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
