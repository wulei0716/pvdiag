# pvdiag 在 macOS 上的部署（MacBook Air / Apple Silicon）

## 一键安装

```bash
cd ~/Projects/pvdiag          # 源码目录（解压 pvdiag-v0.1.zip 后的 pv-string-agents 改名即可）
bash deploy/install_mac.sh --demo --port 8080 --hour 21 --minute 0
```

脚本做的事：
1. 用本机 Python（≥3.10，brew 的 python3.12/3.13 均可）创建 `.venv` 并安装依赖
2. `pvdiag init` 生成 `config.yaml` 与 `data/pvdiag.sqlite`
3. `--demo`：生成合成电站（36 串、10 种故障）并回放 30 天，`data/reports/2026-10-30/report.md` 可直接看效果
4. 安装两个 launchd 用户代理（`~/Library/LaunchAgents/`）：
   - `com.pvdiag.daily`：每天 21:00 运行 `pvdiag run --backfill-days 2`（断点续拉近 2 天 + 全链路），日志 `data/logs/daily.log`
   - `com.pvdiag.web`：常驻复核界面 `http://127.0.0.1:8080`，开机自启、崩溃自拉起，日志 `data/logs/web.log`

> launchd 任务在 Mac 睡眠时不会触发；合盖运行需在「系统设置 → 电池」关闭自动睡眠，或改用 `pmset` 唤醒。运行时间选在 21:00 是为了等全天数据与平台同步（阳光平台当天数据拿不到，系统自动取 T-1）。

## 接入真实电站

```bash
source .venv/bin/activate
mkdir -p import && cp 你的/stations.csv 你的/inverters.csv 你的/strings.csv import/
pvdiag import-profiles ./import
```
数据源三选一（见 README §3.2）：
- **CSV（对接 F:\Hermes 已有拉数脚本）**：电站 `platform=csv`，`extra_json={"platform_config":{"root":"/Users/<你>/pvdata"}}`，把拉数脚本的落地文件同步到 `pvdata/<station_id>/strings/`
- **直连华为/锦浪/阳光**：凭证写入 `stations.extra_json.platform_config`（建议从钥匙串读取后写入，不要明文提交到 git）
- **历史回补**：`pvdiag backfill --start 2026-09-01 --end 2026-11-07`（华为站受配额限制会自动跨天分批，配额用尽时日志提示"明日续拉"）

## 日常操作

| 要做什么 | 命令 / 入口 |
|---|---|
| 看今天结果 | `cat data/reports/$(date +%F)/report.md` 或打开 `http://127.0.0.1:8080` |
| 复核 | 网页「复核队列」→ 确认 / 驳回；或 `pvdiag review list` / `confirm` / `reject` |
| 发工单 | 网页「工单」→ 确认发送（`config.yaml` 里 `dispatch.channel: wecom` + webhook 即推企业微信） |
| 现场回填 | 网页工单页填实际原因/动作 → 系统自动登记事件、3 个晴好天后复查 |
| 改阈值 | 编辑 `config.yaml`，下次 `pvdiag run` 生效 |
| 训练模型 | 标注 ≥60 条后自动按月重训；手动 `pvdiag train` |
| 立即跑一次 | `launchctl kickstart -k gui/$(id -u)/com.pvdiag.daily` 或 `pvdiag run` |
| 查看代理状态 | `launchctl list \| grep pvdiag` |
| 卸载定时任务 | `bash deploy/uninstall_mac.sh` |

## 目录

```
~/Projects/pvdiag/
  .venv/            虚拟环境
  config.yaml       口径配置
  data/
    pvdiag.sqlite   档案/特征/基线/嫌疑/判定/标注/工单/台账
    samples/        5 分钟明细 Parquet（按站/日分区）
    reports/<日期>/ 日报 report.md / report.json
    outbox/         dryrun 派单草稿
    models/         训练好的分类模型
    logs/           daily.log / web.log
  import/           真实电站档案 CSV
  demo/             合成演示数据（可删）
```
