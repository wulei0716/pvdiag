#!/usr/bin/env bash
# 卸载 pvdiag 的 launchd 定时任务（不删除源码与数据）
set -uo pipefail
for name in com.pvdiag.daily com.pvdiag.web; do
  launchctl bootout "gui/$(id -u)/$name" >/dev/null 2>&1 || true
  rm -f "$HOME/Library/LaunchAgents/$name.plist"
  echo "已移除 $name"
done
echo "源码与 data/ 保留；如需彻底删除：rm -rf <源码目录>"
