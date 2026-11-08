#!/usr/bin/env bash
# 容器入口：
#   serve          首次启动自动初始化 + 灌合成演示数据并回放 30 天，然后起复核界面（默认）
#   run            跑一次每日闭环（配合宿主 cron 使用）
#   demo           只做演示数据初始化
#   test           运行测试
#   <任意 pvdiag 子命令>  直接透传，如：pvdiag review list
set -euo pipefail
cd /app
mkdir -p data
if [[ ! -f config.yaml ]]; then
  pvdiag init --data-dir ./data >/dev/null
fi

seed_demo() {
  if [[ ! -f data/.demo_seeded && "${PVDIAG_DEMO:-1}" == "1" ]]; then
    echo "[pvdiag] 首次启动：生成合成电站（36 串、10 种注入故障）并回放 30 天，约 1 分钟..."
    pvdiag simulate --out ./demo --days 60 --load
    pvdiag backfill --start 2026-10-01 --end 2026-10-30 --no-ingest
    touch data/.demo_seeded
    echo "[pvdiag] 演示数据就绪：data/reports/2026-10-30/report.md"
  fi
}

case "${1:-serve}" in
  serve)
    seed_demo
    exec pvdiag serve --host 0.0.0.0 --port "${PORT:-8080}"
    ;;
  demo)
    seed_demo
    ;;
  run)
    shift
    exec pvdiag run "$@"
    ;;
  test)
    exec python -m pytest -q
    ;;
  *)
    exec pvdiag "$@"
    ;;
esac
