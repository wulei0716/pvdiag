#!/usr/bin/env bash
# pvdiag · macOS 一键部署
#   用法：cd <pvdiag 源码目录> && bash deploy/install_mac.sh [--demo] [--port 8080] [--hour 21] [--minute 0]
#   做的事：创建 .venv → 安装依赖 → 初始化 data/ 与 config.yaml → (可选) 灌合成演示数据并回放
#           → 安装两个 launchd 用户代理：每日巡检（默认 21:00）+ 复核 Web 界面（127.0.0.1:端口，常驻）
set -euo pipefail

DEMO=0; PORT=8080; HOUR=21; MINUTE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --demo) DEMO=1; shift;;
    --port) PORT="$2"; shift 2;;
    --hour) HOUR="$2"; shift 2;;
    --minute) MINUTE="$2"; shift 2;;
    *) echo "未知参数 $1"; exit 1;;
  esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "==> 部署目录: $ROOT"

# ---- Python
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    v=$("$c" -c 'import sys; print(sys.version_info[:2] >= (3,10))' 2>/dev/null || echo False)
    if [[ "$v" == "True" ]]; then PY="$c"; break; fi
  fi
done
if [[ -z "$PY" ]]; then
  echo "!! 需要 Python >= 3.10（brew install python@3.12）"; exit 1
fi
echo "==> 使用 $PY ($($PY --version))"

# ---- venv + 依赖
if [[ ! -d .venv ]]; then "$PY" -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -e ".[web,dev]"
echo "==> 依赖安装完成: $(pvdiag --help | head -1)"

# ---- 初始化
mkdir -p data/logs
if [[ ! -f config.yaml ]]; then
  pvdiag init --data-dir ./data
else
  echo "==> 已有 config.yaml，跳过 init"
fi

# ---- 演示数据（可选）
if [[ "$DEMO" == "1" ]]; then
  echo "==> 生成合成电站并回放 30 天（约 1 分钟）..."
  pvdiag simulate --out ./demo --days 60 --load
  pvdiag backfill --start 2026-10-01 --end 2026-10-30 --no-ingest
  echo "==> 演示报告: data/reports/2026-10-30/report.md"
fi

# ---- launchd
AGENTS="$HOME/Library/LaunchAgents"; mkdir -p "$AGENTS"
render() {  # $1 模板 $2 目标
  sed -e "s#__ROOT__#$ROOT#g" -e "s#__PORT__#$PORT#g" -e "s#__HOUR__#$HOUR#g" -e "s#__MINUTE__#$MINUTE#g" \
      -e "s#__HOME__#$HOME#g" "$1" > "$2"
}
for name in com.pvdiag.daily com.pvdiag.web; do
  plist="$AGENTS/$name.plist"
  launchctl bootout "gui/$(id -u)/$name" >/dev/null 2>&1 || true
  render "deploy/$name.plist.template" "$plist"
  launchctl bootstrap "gui/$(id -u)" "$plist"
  launchctl enable "gui/$(id -u)/$name"
done
launchctl kickstart -k "gui/$(id -u)/com.pvdiag.web" || true
sleep 2

cat <<EOF

================ 部署完成 ================
源码目录      : $ROOT
虚拟环境      : $ROOT/.venv
配置          : $ROOT/config.yaml      （阈值/派单/电价都在这里改）
数据与报告    : $ROOT/data/            （reports/<日期>/report.md，outbox/ 为派单草稿）
每日巡检      : launchd com.pvdiag.daily  每天 $(printf '%02d:%02d' "$HOUR" "$MINUTE") 运行 pvdiag run
复核界面      : http://127.0.0.1:$PORT   （launchd com.pvdiag.web 常驻）
日志          : $ROOT/data/logs/daily.log  web.log

常用命令（先 source .venv/bin/activate）：
  pvdiag run                      立即跑一次今天的闭环
  pvdiag review list              复核队列
  pvdiag wo list                  工单
  pvdiag import-profiles ./import 导入真实电站档案（stations/inverters/strings.csv）
  launchctl list | grep pvdiag    查看代理状态
  bash deploy/uninstall_mac.sh    卸载定时任务
EOF
