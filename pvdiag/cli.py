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
  pvdiag train                                           用标注训练/重训分类模型（考试达标才上线；--force 强制上线）
  pvdiag report --date 2026-10-30                        打印当日 Markdown 报告
  pvdiag serve --port 8080                               人工复核 Web 界面
  pvdiag doctor [--date D] [--vacuum]                    健康自检（数据库/磁盘/数据新鲜度/调度/模型/配置），有 fail 退出码 1
  pvdiag tune show|apply|rollback|clear                  自调参：查看评估与变更留痕 / 手动生效某参数 / 回滚 / 清空覆盖层
  pvdiag gate status|open|close                          派单门控：状态 / 手动放开 / 手动收回
  pvdiag model list|promote|demote                       模型注册表：列出 / 指定版本上线 / 降级回规则代理
  pvdiag public fetch|load|list                          公开数据：下载（GitHub 镜像+sha256）/ 转成电站入库 / 列表
  pvdiag bench [--dataset kaggle|nrel|train|all]         公开数据基准评测（召回/误报/停机识别/钉死/限发/弱标注训练）
  pvdiag soak --days 180 --stations 2                    连续运行压测：模拟复核员/主管/现场，看自调参/模型/门控长期行为
  pvdiag explain <suspect_id> [--ask "问题"]              LLM 分析助手：案卷 → 结论/依据/建议/不确定性；无 key 时模板文案
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import Settings
from .storage import Store


def _settings(args, with_overrides: bool = True) -> Settings:
    s = Settings.load(args.config)
    if getattr(args, "data_dir", None):
        s.data_dir = Path(args.data_dir)
    if getattr(args, "workers", None) is not None:
        s.runtime.workers = int(args.workers)
    s.ensure_dirs()
    if with_overrides:
        from .agents.tuning import apply_overrides
        s = apply_overrides(s, Store(s.db_path, s.samples_dir))
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


def cmd_platform(args):
    """云平台接入：register 登记电站+凭证；test 列设备/令牌状态（1~2 次调用）；auth 阳光授权码换令牌；hints 华为 MPPT 容量→档案建议"""
    from .adapters import build_adapter
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    if args.action == "register":
        from .agents.profile import ProfileAgent
        cfg = json.loads(Path(args.conf).read_text(encoding="utf-8")) if args.conf else {}
        if args.set:
            for kv in args.set:
                k, v = kv.split("=", 1); cfg[k] = v
        extra = {"platform_config": cfg}
        if args.mount:
            extra["mount"] = args.mount
        row = {"station_id": args.station, "name": args.name or args.station, "platform": args.platform, "capacity_kwp": args.capacity,
               "region": args.region, "lat": args.lat, "lon": args.lon, "extra_json": json.dumps(extra, ensure_ascii=False)}
        old = store.query("SELECT extra_json, tariff_json FROM stations WHERE station_id=?", (args.station,))
        if not old.empty:                                    # 保留已有的其它 extra 字段
            ex = json.loads(old["extra_json"].iloc[0] or "{}"); ex.update(extra); row["extra_json"] = json.dumps(ex, ensure_ascii=False)
            row["tariff_json"] = old["tariff_json"].iloc[0]
        ProfileAgent.import_stations(store, pd.DataFrame([row]))
        secret_keys = [k for k in cfg if any(t in k.lower() for t in ("code", "secret", "password", "pwd"))]
        print(f"已登记 {args.station}（{args.platform}）；凭证字段 {secret_keys} 已写入本地库 stations.extra_json（勿把 data 目录上传）。"
              f"建议用 *_env 把密钥放环境变量：如 system_code_env=PVDIAG_HW_SECRET")
        return
    st = store.query("SELECT * FROM stations WHERE station_id=?", (args.station,))
    if st.empty:
        print(f"电站 {args.station} 未登记：pvdiag platform register --station ... --platform ... --conf cfg.json"); return
    pconf = json.loads(st["extra_json"].iloc[0] or "{}").get("platform_config", {})
    adapter = build_adapter(st["platform"].iloc[0], args.station, pconf, store)
    if args.action == "auth":
        if st["platform"].iloc[0] != "sungrow":
            print("auth 只用于阳光云（授权码换令牌）"); return
        if not args.code:
            print(f"请业主登录阳光云授权页勾选电站并同意授权，把回调地址里的 ?code=XXXX 交给：\n"
                  f"  pvdiag platform auth --station {args.station} --code XXXX\n当前令牌状态：{adapter.token_status()}"); return
        t = adapter.token_by_code(args.code)
        print(f"令牌已保存：授权电站 {t.get('auth_ps_list')}，有效至 {datetime.fromtimestamp(t['expire_at']):%Y-%m-%d %H:%M}（48h 自动续期）")
    elif args.action == "test":
        if hasattr(adapter, "token_status"):
            print("令牌：", adapter.token_status())
        if hasattr(adapter, "list_stations"):
            try:
                sts = adapter.list_stations()
                print(f"账号下电站 {len(sts)} 个："); print(sts.to_string(index=False) if not sts.empty else "（空）")
            except Exception as e:
                print("电站列表获取失败：", e)
        devs = adapter.list_devices()
        print(f"{args.station}（{st['platform'].iloc[0]}）设备 {len(devs)} 台：")
        if not devs.empty:
            print(devs.to_string(index=False))
        if hasattr(adapter, "calls_used_today"):
            print(f"今日已用调用 {adapter.calls_used_today()} 次")
    elif args.action == "hints":
        from .adapters.huawei import HuaweiAdapter
        caps = store.get_state(f"huawei:{args.station}:mppt_caps") or []
        end = _date(args.end) if args.end else date.today()
        start = _date(args.start) if args.start else end - timedelta(days=3)
        frames = []
        for k in range((end - start).days + 1):
            df_ = store.read_samples("string", args.station, start + timedelta(days=k))
            if df_ is not None and not df_.empty:
                frames.append(df_)
        smp = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if smp.empty:
            print("没有该站的 5 分钟样本（先 pvdiag ingest --station ...）"); return
        h = HuaweiAdapter.profile_hints(smp, pd.DataFrame(caps), module_wp=args.module_wp, inputs_per_mppt=args.inputs_per_mppt,
                                        typical_n_modules=args.typical_modules)
        print(h["state"].value_counts().to_dict())
        print(h[h["state"] != "connected"][["string_id", "mppt", "cap_w", "state", "note"]].to_string(index=False))
        if args.out:
            h.to_csv(args.out, index=False, encoding="utf-8-sig"); print("档案建议：", args.out, "（核对后可并入 strings.csv 导入）")


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


def cmd_doctor(args):
    from .health import run_health_checks, summarize, vacuum
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    checks = run_health_checks(s, store, _date(args.date))     # --date 可按某个巡检日评估（回放演示数据时用）
    icon = {"ok": "✅", "warn": "⚠️ ", "fail": "❌"}
    for c in checks:
        print(f"{icon[c['status']]} {c['name']}: {c['detail']}")
    sm = summarize(checks)
    print(f"\n{sm['ok']} 项正常 / {sm['warn']} 项提醒 / {sm['fail']} 项失败")
    if args.vacuum:
        print(vacuum(store))
    sys.exit(1 if sm["fail"] else 0)


def cmd_tune(args):
    from .agents.tuning import TUNABLE, TuningAgent, set_param
    from .agents.base import AgentContext
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    pd.set_option("display.width", 220)
    if args.action == "show":
        print("当前运行口径：", json.dumps(TuningAgent.current_params(s), ensure_ascii=False))
        ov = store.query("SELECT * FROM param_overrides")
        print("覆盖层：", "无" if ov.empty else "")
        if not ov.empty:
            print(ov.to_string(index=False))
        ch = store.query("SELECT change_id, run_date, param, old_value, new_value, status, reason FROM param_changes ORDER BY change_id DESC LIMIT 20")
        print("\n变更留痕（最近 20 条）：", "无" if ch.empty else "")
        if not ch.empty:
            print(ch.to_string(index=False))
        ctx = AgentContext(settings=s, store=store, run_date=date.today())
        cases = TuningAgent.labeled_cases(ctx, s.autotune.window_days)
        print(f"\n窗口内纯偏差型复核样本 {len(cases)} 条（正例 {sum(c['label'] for c in cases)}）")
        if len(cases) >= 5:
            agent = TuningAgent()
            table = agent.evaluate_grid(ctx, cases, TuningAgent.current_params(s))
            print(pd.DataFrame(table)[["params", "precision", "recall", "f1", "is_current"]].to_string(index=False))
    elif args.action == "apply":
        if args.param not in TUNABLE or args.value is None:
            sys.exit(f"用法：pvdiag tune apply <param> <value>；可调参数：{list(TUNABLE)}")
        val = TUNABLE[args.param](args.value)
        old = json.dumps(TuningAgent.current_params(s)[args.param])
        store.execute("INSERT OR REPLACE INTO param_overrides (param, value_json, updated_at, source) VALUES (?,?,?,?)",
                      (args.param, json.dumps(val), datetime.now().isoformat(), "manual"))
        store.execute("INSERT INTO param_changes (run_date, param, old_value, new_value, reason, metrics_json, status, created_at) VALUES (?,?,?,?,?,?,?,?)",
                      (date.today().isoformat(), args.param, old, json.dumps(val), args.note or "人工生效", "{}", "applied", datetime.now().isoformat()))
        print(f"已生效 {args.param} {old}→{val}（写入覆盖层，config.yaml 未改）")
    elif args.action == "rollback":
        ch = store.query("SELECT * FROM param_changes WHERE status='applied' ORDER BY change_id DESC LIMIT 1")
        if ch.empty:
            sys.exit("没有生效中的自动/人工调整")
        c = ch.iloc[0]
        store.execute("INSERT OR REPLACE INTO param_overrides (param, value_json, updated_at, source) VALUES (?,?,?,?)",
                      (c["param"], c["old_value"], datetime.now().isoformat(), "rollback"))
        store.execute("UPDATE param_changes SET status='rolled_back', rolled_back_at=? WHERE change_id=?", (datetime.now().isoformat(), int(c["change_id"])))
        store.set_state("tuning:last_change", {"at": date.today().isoformat(), "rollback": c["param"], "by": "manual"})
        print(f"已回滚 {c['param']} → {c['old_value']}（该值 90 天内不会再自动生效）")
    elif args.action == "clear":
        store.execute("DELETE FROM param_overrides")
        store.execute("UPDATE param_changes SET status='rolled_back', rolled_back_at=? WHERE status='applied'", (datetime.now().isoformat(),))
        store.set_state("tuning:last_change", {"at": date.today().isoformat(), "rollback": "all", "by": "manual"})
        print("覆盖层已清空，回到 config.yaml 口径")


def cmd_gate(args):
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    state = store.get_state("dispatch:gate") or {"effective": "draft", "streak": 0}
    if args.action == "status":
        print(json.dumps({"config_mode": s.dispatch.mode, **state}, ensure_ascii=False, indent=2, default=str))
    else:
        eff = "auto" if args.action == "open" else "draft"
        state.update({"effective": eff, "streak": s.dispatch.gate_open_streak_days if eff == "auto" else 0,
                      "since": date.today().isoformat(), "reason": f"人工 {args.action}"})
        hist = state.get("history", []); hist.append({"date": date.today().isoformat(), "event": f"人工 {args.action} → {eff}"}); state["history"] = hist[-20:]
        store.set_state("dispatch:gate", state)
        print(f"派单门控已{'放开（auto）' if eff == 'auto' else '收回（draft）'}；下次巡检若条件不满足仍会按 gated 规则自动调整")


def cmd_model(args):
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    pd.set_option("display.width", 220)
    if args.action == "list":
        df = store.query("SELECT model_version, trained_at, n_labels, active, status, metrics_json FROM model_registry ORDER BY trained_at DESC")
        if df.empty:
            print("无模型（规则代理）"); return
        df["exam"] = df["metrics_json"].map(lambda m: json.dumps((json.loads(m or "{}").get("exam") or {}).get("challenger"), ensure_ascii=False))
        print(df.drop(columns=["metrics_json"]).to_string(index=False))
        print("线上一致率：", store.get_state("model:live_agreement"))
    elif args.action == "promote":
        if not args.version or store.scalar("SELECT COUNT(*) FROM model_registry WHERE model_version=?", (args.version,)) == 0:
            sys.exit("用法：pvdiag model promote <model_version>（见 pvdiag model list）")
        store.execute("UPDATE model_registry SET active=0, status=CASE WHEN status='active' THEN 'retired' ELSE status END")
        store.execute("UPDATE model_registry SET active=1, status='active' WHERE model_version=?", (args.version,))
        print(f"已上线 {args.version}")
    elif args.action == "demote":
        store.execute("UPDATE model_registry SET active=0, status='demoted' WHERE active=1")
        print("已降级回规则代理")


def cmd_public(args):
    from .public import DATASETS, dataset_dir, fetch_dataset
    if args.action == "list":
        for k, d in DATASETS.items():
            print(f"{k}: {d.title}\n    {d.license_note}\n    {d.source}")
    elif args.action == "fetch":
        keys = [args.name] if args.name else [k for k, d in DATASETS.items() if d.files]
        for k in keys:
            print(k, "→", fetch_dataset(k, args.public_dir, force=args.force))
    elif args.action == "load":
        from .public.kaggle_solar import load_kaggle_plant
        from .public.bench import load_bundle
        s = _settings(args)
        root = dataset_dir("kaggle_solar", args.public_dir)
        if not root.exists():
            fetch_dataset("kaggle_solar", args.public_dir)
        for plant in ([args.plant] if args.plant else [1, 2]):
            b = load_kaggle_plant(root, plant)
            store = load_bundle(s, b)
            ts = pd.to_datetime(b["samples_strings"]["ts"])
            print(f"已入库 Kaggle Plant {plant} → 电站 {b['stations']['station_id'].iloc[0]}（{len(b['strings'])} 路，{ts.min().date()}~{ts.max().date()}）；"
                  f"回放：pvdiag backfill --start {ts.min().date()} --end {ts.max().date()} --no-ingest")
            store.close()


def cmd_bench(args):
    from .public.bench import bench_kaggle, bench_nrel, write_report
    s = _settings(args, with_overrides=False)
    out = Path(args.out or (s.data_dir / "bench"))
    work = out / "work"
    rep = {"generated_at": datetime.now().isoformat(timespec="minutes")}
    if args.dataset in ("kaggle", "all"):
        rep["kaggle"] = bench_kaggle(work, s, public_root=args.public_dir)
        print("Kaggle:", json.dumps(rep["kaggle"]["summary"], ensure_ascii=False))
    if args.dataset in ("nrel", "all"):
        try:
            rep["nrel"] = bench_nrel(work, s)
            print("NREL:", json.dumps(rep["nrel"]["summary"], ensure_ascii=False))
        except ImportError as e:
            print("NREL 跳过：", e)
    if args.dataset in ("train", "all"):
        from .public.bench import bench_train
        rep["train"] = bench_train(work, s, public_root=args.public_dir)
        print("Train:", json.dumps({k: rep["train"][k] for k in ("n_labels", "promoted", "exam")}, ensure_ascii=False, default=str))
    if args.dataset in ("real", "all"):
        from .public.bench import bench_real
        r = bench_real(work, s, real_root=args.real_dir)
        if r:
            rep["real"] = r
            print("Real:", json.dumps(r["summary"], ensure_ascii=False, default=str))
        elif args.dataset == "real":
            print("Real 跳过：没有 PVDIAG_REAL_DIR/manifest.json（真实站数据不入库，放在本地目录）")
    p = write_report(rep, out)
    print("报告：", p)


def cmd_daily(args):
    """日电量模式：import（巡检报告 xlsx / 长表 csv → 档案+日电量+天气）；diagnose（同机/同通道日电量互比 → suspects）"""
    from . import daily_energy as de
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    if args.action == "import":
        p = Path(args.file)
        if p.suffix.lower() in (".xlsx", ".xls"):
            b = de.read_report_xlsx(p, args.station, args.platform or "", inputs_per_mppt=args.inputs_per_mppt)
        else:
            b = de.read_long_csv(p, args.station, args.platform or "", inputs_per_mppt=args.inputs_per_mppt)
        r = de.load_into_store(store, b)
        sd = b.station_daily
        print(f"{args.station} ← {p.name}：{r['strings']} 路（新增档案 {r['new_strings']}），日电量 {r['energy_rows']} 行，{r['days']} 天"
              f"（晴好 {int((sd['weather_tag'] == 'clear').sum())} 天）；状态 {b.strings['status'].value_counts().to_dict()}")
        print(f"诊断：pvdiag daily diagnose --station {args.station} --start {sd['date'].min()} --end {sd['date'].max()}")
    elif args.action == "diagnose":
        sd = store.station_daily(args.station, args.start or "2000-01-01", args.end or "2100-01-01")
        if sd.empty:
            print("该站没有日电量/天气数据，先 import"); return
        start, end = date.fromisoformat(sd["date"].min()), date.fromisoformat(sd["date"].max())
        r = de.diagnose(store, args.station, start, end, s.thresholds, group_mode=args.group, write=not args.dry_run)
        st = r["strings"]
        print(f"{args.station} {start}~{end}：晴好 {len(r['clear_days'])} 天，分组 {r['group_mode']}，通道三态 {r['state_counts']}")
        g = st[st["grade"].notna()].sort_values("mean_dev")
        print(f"嫌疑 {len(g)} 条：{g['grade'].value_counts().to_dict()}" + ("（--dry-run 未写库）" if args.dry_run else ""))
        if not g.empty:
            cols = ["string_id", "grade", "cause", "mean_dev", "over_dev", "low_days", "n_clear_eval", "lost_kwh"]
            print(g[cols].round(3).to_string(index=False))
        par = [x for x, c in zip(st["string_id"], st["state"]) if c == "connected" and x not in set(g["string_id"])
               and st.set_index("string_id").at[x, "n_clear_eval"] == 0]
        if par:
            print("疑双串并联/档案缺并联数（未参与互比）：", "、".join(par))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            st.to_csv(args.out, index=False, encoding="utf-8-sig"); print("逐串统计：", args.out)


def cmd_soak(args):
    from .sim.soak import SoakConfig, SoakRunner, write_report
    s = _settings(args, with_overrides=False)
    cfg = SoakConfig(days=args.days, n_stations=args.stations, n_inverters=args.inverters, strings_per_inverter=args.strings,
                     seed=args.seed, fault_rate=args.fault_rate, reviewer_error=args.reviewer_error)
    out = Path(args.out or (s.data_dir / "soak"))
    runner = SoakRunner(cfg, out / "world", s)
    runner.build_world()
    print(f"世界已生成：{cfg.n_stations} 站 × {cfg.n_inverters * cfg.strings_per_inverter} 串 × {cfg.days} 天，注入故障 {len(runner.truth)} 个")
    summary = runner.run(progress=lambda d, snap: print(f"  {d} 嫌疑 {snap['suspects']} 误报率 {snap['false_alarm_rate']} 门控 {snap['gate_effective']} "
                                                       f"模型 {snap['model']} 标注 {snap['n_labels']} 覆盖层 {snap['overrides']}"))
    p = write_report(summary, out)
    print(f"召回 {summary['faults_recalled_ab']}/{summary['faults_injected']}，健康串误报 {summary['healthy_false_alarm_ab_strings']}，"
          f"整体误报率 {summary['false_alarm_rate_overall']}，门控 auto {summary['gate']['days_auto']} 天，模型{'已' if summary['model_promoted'] else '未'}上线，"
          f"自调参生效 {summary['autotune']['applied']} 回滚 {summary['autotune']['rolled_back']}，振荡 {summary['autotune']['oscillation']}")
    print("报告：", p)


def cmd_explain(args):
    from .llm import narrative, ask
    s = _settings(args)
    store = Store(s.db_path, s.samples_dir)
    if args.ask:
        r = ask(s, store, args.suspect_id, args.ask)
    else:
        r = narrative(s, store, args.suspect_id, force=args.force)
    print(f"[{r['source']}]\n{r['text']}")


def main(argv=None):
    # 全局选项既可放在子命令前，也可放在子命令后
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--data-dir", default=argparse.SUPPRESS)
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS)
    common.add_argument("--workers", type=int, default=argparse.SUPPRESS, help="按站并行进程数（1 串行，0 = CPU 核数）")
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
    p = add("doctor"); p.add_argument("--vacuum", action="store_true"); p.add_argument("--date"); p.set_defaults(fn=cmd_doctor)
    p = add("tune"); p.add_argument("action", choices=["show", "apply", "rollback", "clear"]); p.add_argument("param", nargs="?")
    p.add_argument("value", nargs="?"); p.add_argument("--note"); p.set_defaults(fn=cmd_tune)
    p = add("gate"); p.add_argument("action", choices=["status", "open", "close"]); p.set_defaults(fn=cmd_gate)
    p = add("model"); p.add_argument("action", choices=["list", "promote", "demote"]); p.add_argument("version", nargs="?"); p.set_defaults(fn=cmd_model)
    p = add("public"); p.add_argument("action", choices=["list", "fetch", "load"]); p.add_argument("name", nargs="?")
    p.add_argument("--plant", type=int); p.add_argument("--public-dir"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_public)
    p = add("explain"); p.add_argument("suspect_id"); p.add_argument("--ask"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_explain)
    p = add("platform"); p.add_argument("action", choices=["register", "test", "auth", "hints"]); p.add_argument("--station", required=True)
    p.add_argument("--platform", choices=["huawei", "sungrow", "ginlong", "csv"]); p.add_argument("--conf", help="platform_config JSON 文件"); p.add_argument("--set", action="append")
    p.add_argument("--name"); p.add_argument("--capacity", type=float); p.add_argument("--region"); p.add_argument("--lat", type=float); p.add_argument("--lon", type=float)
    p.add_argument("--mount"); p.add_argument("--code"); p.add_argument("--start"); p.add_argument("--end")
    p.add_argument("--module-wp", type=float); p.add_argument("--typical-modules", type=int); p.add_argument("--inputs-per-mppt", type=int, default=2)
    p.add_argument("--out"); p.set_defaults(fn=cmd_platform)
    p = add("daily"); p.add_argument("action", choices=["import", "diagnose"]); p.add_argument("--station", required=True)
    p.add_argument("--file"); p.add_argument("--platform"); p.add_argument("--inputs-per-mppt", type=int, default=2)
    p.add_argument("--start"); p.add_argument("--end"); p.add_argument("--group", choices=["auto", "inverter", "channel"], default="auto")
    p.add_argument("--dry-run", action="store_true"); p.add_argument("--out"); p.set_defaults(fn=cmd_daily)
    p = add("soak"); p.add_argument("--days", type=int, default=180); p.add_argument("--stations", type=int, default=2)
    p.add_argument("--inverters", type=int, default=3); p.add_argument("--strings", type=int, default=12); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fault-rate", type=float, default=0.004); p.add_argument("--reviewer-error", type=float, default=0.08); p.add_argument("--out")
    p.set_defaults(fn=cmd_soak)
    p = add("bench"); p.add_argument("--dataset", choices=["kaggle", "nrel", "train", "real", "all"], default="all"); p.add_argument("--out")
    p.add_argument("--public-dir"); p.add_argument("--real-dir"); p.set_defaults(fn=cmd_bench)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
