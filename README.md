# pvdiag · 组串发电量异常诊断智能体系统

[![CI](https://github.com/wulei0716/pvdiag/actions/workflows/ci.yml/badge.svg)](https://github.com/wulei0716/pvdiag/actions/workflows/ci.yml)

> 竖向对比(同串历史) → 横向对比(同机互比/站间互比) · 归一化 → AI判定 → 人工复核 → 自动派单 → 闭环自学习
>
> 一句话结论（来自需求表）：判定标准不取"某一天"，而是统计基线——以"晴好天中午窗口(11:30-13:30)、同逆变器各串电流中位数互比"为主力（辐照/温度天然归一），偏离 ≥5% 且连续 2 个晴好天即判异常；先用可解释规则引擎起步，人工复核结果作为标注数据喂养模型，逐步过渡到 AI 自动判定 + 自动派单。

本仓库是该需求的 **v0.1 可运行实现**：10 个阶段 → 10 个智能体 + 每日闭环编排 + CLI + 人工复核 Web 界面，自带合成电站与故障注入，可在没有真实数据时跑通并验证召回/误报。

---

## 1. 架构

```
                 ┌──────────────────────────────────────────────────────────────────┐
  华为北向 ──┐    │  ① IngestAgent   接入/清洗/5分钟明细(Parquet)/日特征库(SQLite)        │
  锦浪云  ──┼──▶ │  ② ProfileAgent  组串档案·分组(逆变器|串长|朝向|倾角)·事件·电压校验      │
  阳光云  ──┤    │  ③ BaselineAgent 晴好天筛选·同机互比滚动基线·机间互比·分段重置·冷启动     │
  CSV(Hermes)┘    │  ④ DetectAgent   规则引擎：曲线签名分型 → A/B/C/D/E 分级 → 案例连续性    │
                 │  ⑤ ClassifyAgent 特征工程·树模型/规则代理·孤立森林兜底·置信度分流·可解释  │
                 │  ⑧ LossAgent     损失电量/电费台账（起算日=首次跌破-3%的晴好日）          │
                 │  ⑥ ReviewAgent   复核队列·标注回写·确认率统计·阈值调优建议               │
                 │  ⑦ WorkOrderAgent 工单模板/优先级/区域合并·dryrun|企业微信·回填·复查归零  │
                 │  ⑨ RegionAgent   区域任务包·积灰签名(慢渐变+雨后恢复)·清洗效果验证        │
                 │  ⑩ MonitorAgent  管道健康·配额/token·模型漂移(PSI)·KPI·自动派单门控       │
                 └──────────────────────────────────────────────────────────────────┘
                                   orchestrator.DailyPipeline（每日 / 历史回放）
                                   → data/reports/<date>/report.{json,md}
                                   → web/（FastAPI 复核界面）  → cli（pvdiag ...）
```

| 需求阶段 | 模块 | 状态 |
|---|---|---|
| 阶段0 需求定义 | `config.py` 全部口径参数化；`models.py` 分级/原因枚举 | ✅ |
| 阶段1 数据接入与治理 | `adapters/`（huawei/ginlong/sungrow/csv）`agents/ingest.py` `features.py` | ✅ 框架+CSV 可用；三家云 API 需用真实凭证联调字段 |
| 阶段2 组串档案 | `agents/profile.py` | ✅ |
| 阶段3 基线建模 | `agents/baseline.py` | ✅ |
| 阶段4 异常检测（规则引擎） | `agents/detect.py` | ✅ |
| 阶段5 AI 判定 | `agents/classify.py` | ✅ 冷启动规则代理 → 标注≥60 自动切树模型 |
| 阶段6 人工复核平台 | `agents/review.py` `web/` | ✅ |
| 阶段7 工单派发与闭环 | `agents/workorder.py` | ✅ dryrun + 企业微信 webhook |
| 阶段8 损失量化 | `agents/loss.py` | ✅ |
| 阶段9 区域检修/清洗 | `agents/region.py` | ✅ |
| 阶段10 闭环自学习与监控 | `agents/monitor.py` `orchestrator.py` | ✅ |

## 2. 快速开始

### 2.0 给测试者：三种零配置方式

| 方式 | 命令 | 得到什么 |
|---|---|---|
| **Docker** | `docker compose up -d` → 打开 http://127.0.0.1:8080 | 首次启动自动生成合成电站（36 串、10 种注入故障）并回放 30 天，复核界面直接可点 |
| **GitHub Codespaces** | 仓库页 Code → Codespaces → Create | 浏览器里全自动装好并打开复核界面，什么都不用装 |
| **本地** | 见下方 2.1 | 同上，另可跑 `pytest` 看召回/误报测试 |

容器内任意命令：`docker compose exec pvdiag pvdiag review list`、`docker compose run --rm pvdiag test`。
接真实数据：把拉数脚本的落地目录挂到 `/import`（见 `docker-compose.yml` 注释）。

### 2.1 本地安装（合成数据 10 分钟跑通）

```bash
pip install -e ".[web,dev]"

pvdiag init --data-dir ./data                       # 生成 config.yaml 与数据库
pvdiag simulate --out ./demo --days 60 --load        # 合成 1 站 3 台逆变器 36 串、60 天、注入 10 种故障并入库
pvdiag backfill --start 2026-10-01 --end 2026-10-30 --no-ingest   # 历史回放：逐日基线/诊断/判定/损失
pvdiag report --date 2026-10-30                      # 当日 Markdown 巡检报告
pvdiag review list                                   # 复核队列
pvdiag wo list                                       # 工单（draft 草稿）
pvdiag serve --port 8080                             # 复核 Web 界面 http://127.0.0.1:8080
pytest -q                                            # 10 个测试：召回/误报、闭环、训练、单元
```

合成故障与系统判定（30 天回放，36 串，全部命中、健康串零误报）：

| 注入故障 | 系统判定 | 分级 |
|---|---|---|
| 断线（全天 0A） | 断线/未接 | A → 直接派单 |
| 旁路二极管（10:00-13:00 电流 67%） | 旁路二极管（凹陷签名） | B → 直接派单 |
| 下午固定遮挡（14:30-17:30，中午窗口正常） | 遮挡（凹陷签名，不依赖中午互比） | B |
| 朝向差异（晴天 -6%，阴天 0） | 朝向差异（阴天消失签名） | C → 核档案不派现场 |
| 局部积灰（慢渐变、雨后恢复） | 组件衰减/积灰 + 区域智能体"积灰签名" | B |
| 电流钉死 8.0A | 数据问题 | A |
| 串长档案错（电压 -9%） | 档案差异/串长（电压异位） | A → 核档案 |
| 已备案遮挡 | 已备案（抑制，不报警） | — |
| -4.5% 阶跃均匀低发 | 组件衰减/积灰 | B |
| 整机 MPPT -8% | MPPT异常（机间互比，互比看不出） | B（整机级） |

## 3. 接入真实电站

### 3.1 档案（阶段2，AI 归因的地基）
准备 `stations.csv / inverters.csv / strings.csv / events.csv`（列名见 `pvdiag/models.py`，`demo/*.csv` 是现成样例），然后：
```bash
pvdiag import-profiles ./import
```
- `strings.csv` 必填：`string_id, station_id, inverter_id, pv_index, n_modules, azimuth, tilt`；强烈建议填 `roof_area, module_wp, module_vmp, module_imp, known_shading, shading_expected_dev`
- `string_id` 约定 `<inverter_id>-pv<N>`，与适配器输出一致；**pv 编号与物理串的对应关系须由现场排布图确认**
- 配置差异串自动分组（逆变器|串长|朝向桶|倾角桶|组件型号），不同配置不混合互比
- 清洗/维修/换串用 `pvdiag event add ...` 登记，基线自该日分段重置

### 3.2 数据（阶段1）
三种方式任选：
1. **复用 F:\Hermes 已有拉数脚本**：把 `pull_range.py` 的落地文件放到 `<root>/<station_id>/strings/`（长表 `string_id,ts,current_a,voltage_v` 或华为宽表 `inverter_id,ts,pv1_i,pv1_u,...` 都认），`inverters/` 放 `inverter_id,ts,pac_kw,status`，电站 `platform=csv`，`extra_json={"platform_config":{"root":"<root>"}}`
2. **直连云平台**：`platform=huawei|ginlong|sungrow`，凭证放 `extra_json.platform_config`（见 `config.example.yaml` 末尾）。华为配额"先算后拉"（`QuotaPlanner`：日限额 = Σ⌈设备数/100⌉+24，每批 ≤4 天，登录计入），token 30 分钟续期；锦浪 pac W→kW；阳光 token 48h、不取当天
3. **程序内直灌**：`IngestAgent().ingest_frames(ctx, station_id, strings_df, inverters_df)`

每日任务：
```bash
pvdiag run --date $(date +%F)          # 拉当天增量 + 全链路；阳光站自动取 T-1
pvdiag run --backfill-days 3           # 断点续拉近 3 天
```
macOS launchd / Windows 计划任务 / cron 任选其一每天 20:00 后执行（需要全天数据）。

### 3.3 控制台（阶段6/7）
`pvdiag serve` 打开控制台（服务端渲染 + 内联 SVG，无任何外网依赖，内网/私有化可用，自动适配深色模式）：

| 页面 | 作用 |
|---|---|
| 看板 `/` | **组串热力图**（行 = 逆变器，列 = pv 编号，颜色 = 当日互比偏差，描边 = 在办案例）+ 待复核 + 工单 + 损失台账 + 流水线状态 |
| 组串热力图 `/strings` | 全站/按站、按日切换；点格子进单串详情 |
| 单串详情 `/string/{id}` | 当日电流曲线 vs 群组中位、45 天偏差趋势、档案、案例、近 10 日签名、事件 |
| 复核队列 `/review` | 按分级/电站筛选，行内确认 / 驳回；详情页"下一条"连续复核 |
| 工单 `/workorders` | 草稿一键发送、现场回填、复查结果 |
| 每日报告 `/reports` | report.json 的结构化渲染，可按日切换 |
| API `/api/queue` `/api/report/{d}` `/api/heatmap/{station}` | 供自有工单系统/下游消费 |

- 复核确认 / 驳回 / 修改原因 → 回写 `reviews` 表成为训练标注
- 工单默认 `draft` 模式：系统生成草稿，人工在界面"确认发送"；`channel: wecom` 配企业微信机器人 webhook 即可推送
- 现场回填实际原因/动作后自动登记事件（基线分段重置），3 个晴好天后自动复查：归零 → `verified`，未归零 → 自动重开
- 误报率连续 1 个月 <10% 且复核样本 ≥30 后，可把 `dispatch.mode` 改为 `auto`（门控在 MonitorAgent 看板里实时显示）

### 3.4 调优（阶段4 块二）
`pvdiag review stats` 给出确认率；ReviewAgent 自动在日报中给阈值调优建议（B 级确认率 <50% → 收紧到 -6%/-7%；现场漏报多 → 放松到 -4%）。改 `config.yaml` 即生效，无需改代码。

## 4. 关键口径实现说明

- **晴好天**：全站中午功率 ≥ 近 60 天中午功率 95 分位的 35%，且中午窗口变异系数 ≤0.15（云影抖动 → cloudy）；阴雨天不参与判定
- **互比偏差** `dev_today = I_串 / median(I_同群组) − 1`，中午窗口中位数抗抖动；`baseline_ratio` = 近 30 个晴好天 ratio 中位数（该串"正常站位"），`dev_vs_own` 捕捉相对自身的阶跃
- **持续度**：`persist_days_5/3` 只在晴好天序列上计数；非晴好天沿用已有判定不新增
- **限发**：逆变器功率"平顶"（连续 ≥6 点相对变化 ≤0.3% 且贴近额定）→ 该时段剔除出签名与损失
- **停机/缺测**：全站 nodata 日不计入 0 电流连续天数（缺数据 ≠ 零发电）
- **冷启动**：晴好天样本 <5 → 只做同日互比，置信度下调；全站集体衰减用机间互比 + 站级趋势提示兜底
- **案例连续性**：同一串的异常跨天是一个案例（suspect_id 固定），分级只升不降；驳回的案例 30 天内不重复报警（除非恶化 ≥5 个百分点或升级）；连续 3 个晴好天回到正常带自愈关闭
- **损失**：仅晴好天累计，`Σ时段 max(0, 群组同时段中位电流×基线比 − 实际电流) × 群组电压 × 0.5h`，限发时段不计，峰谷平电价；起算日取 suspect_id 中的推定起始日

## 5. 目录

```
pvdiag/
  config.py        口径参数（YAML）         storage.py   SQLite + Parquet
  models.py        统一数据模型/枚举        features.py  日特征/天气/平顶
  adapters/        huawei ginlong sungrow csv（QuotaPlanner）
  agents/          ingest profile baseline detect classify loss review workorder region monitor
  orchestrator.py  DailyPipeline（run/backfill/报告）
  sim/synth.py     合成电站 + 故障注入        web/         FastAPI 复核界面（服务端 SVG 图，无外网依赖）
  cli.py           pvdiag 命令行
tests/             端到端召回/误报、闭环、训练、单元
deploy/            macOS 一键安装（launchd 每日巡检 + 常驻复核界面）
docker/ Dockerfile docker-compose.yml   容器化体验/部署
.devcontainer/     GitHub Codespaces 一键环境
.github/workflows/ CI：pytest + 演示回放 + Docker 构建
docs/              示例日报、控制台截图（看板/热力图/嫌疑详情/深色）、示例档案 CSV
```

### 部署到 Mac（本机常驻）

```bash
bash deploy/install_mac.sh --demo     # .venv + 依赖 + 演示数据 + launchd（每日 21:00 巡检，复核界面常驻 :8080）
```
详见 `deploy/部署说明-macOS.md`。

## 6. 待决策事项（需求表第六块，代码已留接口）

1. 派单渠道：已实现 dryrun + 企业微信；钉钉/自有工单系统可实现 `DispatchChannel.send()` 接入
2. 站间对比气象源：`station_daily.irradiance_kwh_m2` 预留，有辐照仪/卫星数据时竖向基线 Y=E/H 自动启用
3. 缺组件参数的站：ProfileAgent 每日报告缺参串；损失用实测 V/I 不依赖铭牌，缺电压时标"待量化"
4. 冷启动口径：`baseline.min_clear_days`，不足只做同日互比
5. 已知遮挡备案：`strings.known_shading / shading_expected_dev`，备案由谁维护需定
6. 复核责任人与频率：`pvdiag review batch --station X` 生成周批清单
7. 数据合规：全部本地 SQLite/Parquet，私有化部署无出站
8. 跟踪支架/双面组件、9. 微逆/优化器：未建模，接入前先盘点设备类型
10. 运行归属：CLI 可在本机 Hermes 定时任务或服务器 cron 下运行，状态/配额/告警都在库里

## 7. 已知限制 / 下一步

- 三家云 API 适配器按公开接口实现，**字段名与限流细节需用 F:\Hermes 现有工具包和真实凭证联调**
- 天气标记无外部气象数据，"雨天"用阴雨日近似；有气象 API 时在 IngestAgent 填 `irradiance/rain`
- 规则阈值是经验起步值；ML 模型需 ≥60 条人工标注后自动启用，首版可用历史人工判定结果灌入 `reviews`
- 性能：36 串×30 天回放约 25 s；400 串规模每日任务约 1 分钟；万串规模建议把基线计算向量化或按站并行
