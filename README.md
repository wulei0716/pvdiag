# pvdiag · 组串发电量异常诊断智能体系统

[![CI](https://github.com/wulei0716/pvdiag/actions/workflows/ci.yml/badge.svg)](https://github.com/wulei0716/pvdiag/actions/workflows/ci.yml)

> 竖向对比(同串历史) → 横向对比(同机互比/站间互比) · 归一化 → AI判定 → 人工复核 → 自动派单 → 闭环自学习
>
> 一句话结论（来自需求表）：判定标准不取"某一天"，而是统计基线——以"晴好天中午窗口(11:30-13:30)、同逆变器各串电流中位数互比"为主力（辐照/温度天然归一），偏离 ≥5% 且连续 2 个晴好天即判异常；先用可解释规则引擎起步，人工复核结果作为标注数据喂养模型，逐步过渡到 AI 自动判定 + 自动派单。

本仓库是该需求的 **v0.4 可运行实现**：10 个阶段 → 11 个智能体 + 每日闭环编排 + CLI + 人工复核 Web 界面，自带合成电站与故障注入，可在没有真实数据时跑通并验证召回/误报。

**v0.4 新增（接真实电站前补齐代码欠缺）**：**日电量模式** `pvdiag daily import|diagnose`——只有"每串每日 kWh"（巡检报告/平台导出宽表）也能跑同机/同通道互比与通道三态（接入/空输入/零电流/未接），并用三座真实屋顶电站的巡检表做了基准（[说明与结果](docs/日电量模式与真实电站基准.md)）；
辐照源可插拔（实测 → NASA POWER → Meteonorm → Open-Meteo → 晴空模型，按序回退+缓存+来源标记）与同区域站间 PR 互比；设备类型（优化器/微逆功率输入、双面组件增益归一、`n_parallel` 并联串数归一、空输入不报断线）；双轴跟踪建模与卡滞物理曲线校核；按站并行 `--workers`。

**v0.3 新增**：连续运行压测 `pvdiag soak`（模拟复核员/主管/现场跑半年，验证自调参/模型/门控的长期行为，[报告](docs/连续运行压测报告.md)）；
诊断能力——季节性朝向物理归一与档案校核（晴空倾斜面模型 + 朝向网格拟合，直接告诉复核员"曲线最像东向 90°"）、跟踪支架卡滞签名、站级积灰量化与清洗回本；
LLM 分析助手（复核页「分析助手」面板 + 追问，`pvdiag explain`，无 key 自动模板）；性能（360 串每日约 2 秒，回放提速 3 倍）；现场回填自动写入标注。

**v0.2 新增（自主运行 + 公开数据考试）**：阈值自调参闭环（影子回放 · 安全区间 · 留痕 · 自动回滚）、模型生命周期（时间留出集考试 · 达标上线 · 线上落后自动降级）、派单门控自动开合（连续达标放开 · 管道异常收回）、数据接入自愈（补缺口 · 重试 · 降级 · 升级告警）、`pvdiag doctor` 自检与关键事件推送、控制台「自主运行」页；并用 **Kaggle 两座真实电站（44 台逆变器）+ NREL 人工标注片段** 做了基准评测，据此新增「间歇停机/掉线」原因、群组噪声门控、冷启动分级上限、半天钉死检测等（[报告](docs/公开数据基准报告.md) · [说明](docs/自主运行与公开数据说明.md)）。

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
                 │  ④' TuningAgent  阈值自调参：复核结果当考题·影子回放·安全区间·留痕·自动回滚   │
                 │  ⑦ WorkOrderAgent 工单模板/优先级/区域合并·dryrun|企业微信·回填·复查归零·  │
                 │                   门控自动开合(gated)                                   │
                 │  ⑨ RegionAgent   区域任务包·积灰签名(慢渐变+雨后恢复)·清洗效果验证        │
                 │  ⑩ MonitorAgent  管道健康·配额/token·模型漂移(PSI)·KPI·健康自检·关键事件推送 │
                 └──────────────────────────────────────────────────────────────────┘
                                   public/（Kaggle 两站 · NREL 标注片段 → pvdiag bench 考试）
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
| 阶段4 异常检测（规则引擎） | `agents/detect.py` `solar.py` `daily_energy.py` | ✅ v0.2：间歇停机/掉线签名、群组噪声门控、冷启动分级上限、半天钉死；v0.3：季节性朝向物理归一+档案校核、跟踪支架卡滞；v0.4：双轴卡滞物理校核、空输入、并联归一、日电量模式 |
| 阶段4·块二 阈值调优 | `agents/tuning.py` | ✅ v0.2：自调参闭环（影子回放 → 达标生效 → 确认率下跌回滚） |
| 阶段5 AI 判定 | `agents/classify.py` | ✅ 冷启动规则代理 → 标注≥60 自动训练；v0.2：时间留出集考试、达标上线、线上落后自动降级 |
| 阶段6 人工复核平台 | `agents/review.py` `web/` | ✅ 控制台新增「自主运行」页 |
| 阶段7 工单派发与闭环 | `agents/workorder.py` | ✅ dryrun + 企业微信 webhook；v0.2：`gated` 门控自动开合 |
| 阶段8 损失量化 | `agents/loss.py` | ✅ |
| 阶段9 区域检修/清洗 | `agents/region.py` `irradiance.py` | ✅ v0.3：站级积灰量化（PR/清洁参考/日损/清洗回本）；v0.4：辐照源回退链、同区域站间 PR 互比 |
| 阶段10 闭环自学习与监控 | `agents/monitor.py` `health.py` `notify.py` `orchestrator.py` | ✅ v0.2：健康自检、关键事件推送、接入自愈 |
| 公开数据基准 | `public/` | ✅ v0.2：Kaggle 两站 + NREL 标注片段，`pvdiag bench` |
| 连续运行压测 | `sim/soak.py` | ✅ v0.3：`pvdiag soak`，模拟复核员/主管/现场 |
| LLM 分析助手 | `llm.py` | ✅ v0.3：案卷 → 结论/依据/建议/不确定性，追问；可关 |

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
pvdiag doctor                                        # 健康自检
pvdiag soak --days 180 --stations 2                  # 连续运行压测（约 4 分钟）→ soak_report.md
pvdiag explain <suspect_id> --ask "现场怎么办"          # 分析助手（设 ANTHROPIC_API_KEY 用大模型，否则模板）
pytest -q                                            # 28 个测试：召回/误报、闭环、训练、自主运行、公开数据、压测、诊断、助手、单元
```

### 2.2 公开数据考试（真实电站）

```bash
pip install -e ".[web,dev,bench]"
pvdiag public fetch kaggle_solar                     # 4 个 CSV（GitHub 镜像，sha256 校验）→ data/public/kaggle_solar
pvdiag bench --dataset all --out ./bench             # Kaggle 两站回放打分 + NREL 标注片段 + 弱标注训练，生成 bench_report.md
pvdiag public load --plant 2 && pvdiag backfill --start 2020-05-15 --end 2020-06-17 --no-ingest && pvdiag serve   # 在控制台里看真实电站
```

| 考题 | 结果（v0.2） |
|---|---|
| Kaggle 44 台逆变器中 4 台持续低发 | 召回 **4/4** |
| 20 台从未掉线的健康机器 | A/B 误报 **0/20**（v0.1 为 4/20） |
| Plant 2 的 18 台间歇掉线机器 | **18/18** 识别为「间歇停机/掉线」（v0.1 全判遮挡） |
| NREL 人工标注钉死日 / 限发日 | 钉死 4/4 零误报；限发 4/4、1 天误标 |
| 弱标注训练（53 条、10 类）时间留出集 | 随机森林 69% vs 规则代理 54% → 达标上线 |

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
| 间歇停机（每天随机 2h 掉到 0） | 间歇停机/掉线（白天整段 0 而同机正常，不是遮挡） | B |
| 半天钉死（10:30 起冻结 3h） | 数据问题（stale_run_min ≥90） | A |
| 实际东向、档案如实登记 | 物理归一后不报警（季节性期望比扣除） | — |
| 实际东向、档案误记南向 | 朝向差异（曲线最像方位 90°，核档案） | C |
| 跟踪电站一排支架卡滞 | 跟踪支架卡滞（清晨/傍晚双侧偏低、中午正常） | B |
| 全站同步积灰 | 站级积灰量化：损失 %、kWh/天、清洗回本天数（互比盲区） | 站级提示 |
| 整机 MPPT -8% | MPPT异常（机间互比，互比看不出） | B（整机级） |

## 3. 接入真实电站

### 3.1 档案（阶段2，AI 归因的地基）
准备 `stations.csv / inverters.csv / strings.csv / events.csv`（列名见 `pvdiag/models.py`，`demo/*.csv` 是现成样例），然后：
```bash
pvdiag import-profiles ./import
```
- `strings.csv` 必填：`string_id, station_id, inverter_id, pv_index, n_modules, azimuth, tilt`；强烈建议填 `roof_area, module_wp, module_vmp, module_imp, known_shading, shading_expected_dev`
- v0.4 新增可选列：`n_parallel`（一路 MPPT 接两串并联填 2，电流按并联数折成单串再互比——否则双串占多数的逆变器里单串会恒显 −50%）、`device_type`（string|optimizer|microinverter，后两者只有功率、不判电压异位）、`module_type`（mono|bifacial，双面按 `thresholds.bifacial_gain` 归一）
- `mppt` 列用于"空输入"判别：华为/阳光每 MPPT 两路输入只接一路时，空的那路读到共压电压、电流恒 0——档案未登记组件数且同 MPPT 兄弟路正常 → 不报断线（0 电流 + 档案有组件数才是真断线）
- `string_id` 约定 `<inverter_id>-pv<N>`，与适配器输出一致；**pv 编号与物理串的对应关系须由现场排布图确认**
- 配置差异串自动分组（逆变器|串长|朝向桶|倾角桶|组件型号），不同配置不混合互比
- 清洗/维修/换串用 `pvdiag event add ...` 登记，基线自该日分段重置

### 3.2 数据（阶段1）
三种方式任选：
1. **复用 F:\Hermes 已有拉数脚本**：把 `pull_range.py` 的落地文件放到 `<root>/<station_id>/strings/`（长表 `string_id,ts,current_a,voltage_v` 或华为宽表 `inverter_id,ts,pv1_i,pv1_u,...` 都认），`inverters/` 放 `inverter_id,ts,pac_kw,status`，电站 `platform=csv`，`extra_json={"platform_config":{"root":"<root>"}}`
2. **直连云平台**：`platform=huawei|ginlong|sungrow`，凭证放 `extra_json.platform_config`（见 `config.example.yaml` 末尾）。华为配额"先算后拉"（`QuotaPlanner`：日限额 = Σ⌈设备数/100⌉+24，每批 ≤4 天，登录计入），token 30 分钟续期；锦浪 pac W→kW；阳光 token 48h、不取当天
3. **程序内直灌**：`IngestAgent().ingest_frames(ctx, station_id, strings_df, inverters_df)`

4. **只有"每串每日 kWh"时（v0.4 日电量模式）**：巡检报告/平台导出的宽表（行 = 逆变器×通道，列 = 日期，另有 状态 列）或长表 CSV
   ```bash
   pvdiag daily import --station XWZ3 --file 月报.xlsx --platform huawei      # 档案 + 日电量 + 天气（概况表的占晴峰%/天气）
   pvdiag daily diagnose --station XWZ3 [--group auto|inverter|channel]      # 同机（或同编号通道跨设备）日电量互比 → suspects
   ```
   没有曲线形状，证据弱于 5 分钟互比：互比最高给 B、晴好天 <5 天上限 D；阴天偏差消失 → C 朝向；与天气无关的恒定比例 → C 档案差异/串长（电压型偏差，电流互比看不见）；
   通道三态：接入 / 空输入（同 MPPT 兄弟路在发电）/ 零电流待核实 / 未接；电量 ≥1.6× 群组中位 → 疑双串并联，不作异常也不进中位数。详见 [docs/日电量模式与真实电站基准.md](docs/日电量模式与真实电站基准.md)

每日任务：
```bash
pvdiag run --date $(date +%F)          # 拉当天增量 + 全链路；阳光站自动取 T-1
pvdiag run --workers 0                 # 多站：按站并行（0 = CPU 核数；或 config runtime.workers / 环境变量 PVDIAG_WORKERS）
pvdiag run --backfill-days 3           # 断点续拉近 3 天
```
macOS launchd / Windows 计划任务 / cron 任选其一每天 20:00 后执行（需要全天数据）。

### 3.3 控制台（阶段6/7）
`pvdiag serve` 打开控制台（服务端渲染 + 内联 SVG，无任何外网依赖，内网/私有化可用，自动适配深色模式）：

| 页面 | 作用 |
|---|---|
| 看板 `/` | **组串热力图**（行 = 逆变器，列 = pv 编号，颜色 = 当日互比偏差，描边 = 在办案例）+ 待复核 + 工单 + 损失台账 + 流水线状态 |
| 自主运行 `/autonomy` | 派单门控状态与开合历史、运行口径与覆盖层、参数变更留痕、模型注册表与考试成绩、健康自检、接入自愈、已推送事件 |
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

### 3.4 调优（阶段4 块二）—— v0.2 起自动
`pvdiag review stats` 给出确认率；TuningAgent 把复核结果当考题，对候选阈值做影子回放，只在安全区间内走一步、考试达标才生效（写运行时覆盖层，`config.yaml` 不动），生效后确认率下跌自动回滚；放松只提议不自动生效。`pvdiag tune show|apply|rollback|clear`，或 `autotune.mode: shadow` 只看不动。

### 3.5 诊断能力（v0.3）
- **季节性朝向**：同机各串朝向/倾角不同时，先按晴空倾斜面辐照把中午电流折算到参考朝向再互比（冬天东向比南向低 25% 是物理不是故障）；
  曲线形状明显像另一个朝向（网格拟合误差 <0.05 且显著优于档案朝向）→ C 级「朝向差异」并给出最像的方位/倾角，核排布图即可，不派现场
- **跟踪支架**：`stations.extra_json.mount: tracker`（水平单轴）或 `tracker_dual`（双轴，v0.4）的电站，某串清晨/傍晚相对同机都 ≤85% → 卡滞候选；单轴要求中午 ≈1，双轴冬季中午也会偏低，改用物理曲线校核（比值曲线 vs "卡在固定倾角"的期望曲线，rmse<0.06 且明显优于平直假设）→ 「跟踪支架卡滞」并给出最像的卡滞倾角
- **辐照源可插拔**（v0.4）：`irradiance.providers` 按序回退 实测 → NASA POWER → Meteonorm → Open-Meteo → 晴空模型；每天来源记在 `station_daily.irradiance_source`，晴空模型只作分母不判天气；同区域同一来源的站做 **站间 PR 互比**（全站一起低发的互比盲区）
- **设备类型**（v0.4）：优化器/微逆只有功率 → 按 `P/U`（无电压按 100V 虚拟电压）折等效电流走同机互比、不判电压异位；双面组件按增益归一；`n_parallel` 并联串数归一
- **站级积灰**：有辐照用 PR、无辐照用中午功率/峰值参考；清洁参考取最近降雨后首个晴好天；输出积灰损失 %、kWh/天、清洗成本与回本天数（`loss.cleaning_cost_per_kwp`）
- **分析助手**：复核页「分析助手」面板把案卷（偏差序列/签名/档案/历史复核/同类案例现场结论/损失/工单）交给大模型写 结论/依据/建议/不确定性，并可追问；模型只能引用案卷事实；无 key 用模板，`llm.enabled: false` 彻底关闭出站

### 3.6 自主运行（v0.2）
- **模型**：标注够了自动训练，按时间留出最近 25% 考试，不低于规则代理/现役才上线；线上一致率落后 10 个百分点自动降级。`pvdiag model list|promote|demote`
- **派单门控**：`dispatch.mode: gated`（默认）——误报率连续 7 个运行日达标自动放开 auto，智能体失败/数据时延/现场漏报立即收回；`draft` 永远草稿。`pvdiag gate status|open|close`
- **接入自愈**：回看 14 天补缺口、重试退避、云平台失败降级读 `fallback_csv_root`、连续失败升级告警
- **自检与推送**：`pvdiag doctor`；关键事件（管道 error / 自检 fail / 门控开合 / 模型上线降级 / 参数生效回滚）推企业微信或写 `outbox/alerts/`，24h 去重
- **现场回填 = 标注**（v0.3）：工单回填的实际原因自动写入复核表（一致 confirm / 不一致 modify / "正常·误报" reject），是最可靠的标注来源
- 控制台 `/autonomy` 一屏看全。详见 [docs/自主运行与公开数据说明.md](docs/自主运行与公开数据说明.md)；长期行为见 [docs/连续运行压测报告.md](docs/连续运行压测报告.md)

## 4. 关键口径实现说明

- **晴好天**：全站中午功率 ≥ 近 60 天中午功率 95 分位的 35%，且中午窗口变异系数 ≤0.15（云影抖动 → cloudy）；阴雨天不参与判定
- **互比偏差** `dev_today = I_串 / median(I_同群组) − 1`，中午窗口中位数抗抖动；`baseline_ratio` = 近 30 个晴好天 ratio 中位数（该串"正常站位"），`dev_vs_own` 捕捉相对自身的阶跃
- **持续度**：`persist_days_5/3` 只在晴好天序列上计数；非晴好天沿用已有判定不新增
- **限发**：逆变器功率"平顶"（连续 ≥6 点相对变化 ≤0.3% 且贴近额定）→ 该时段剔除出签名与损失
- **停机/缺测**：全站 nodata 日不计入 0 电流连续天数（缺数据 ≠ 零发电）
- **群组噪声门控**（v0.2，来自 Kaggle 两站）：每日算群组偏差稳健标准差 σ=1.4826·MAD；当日偏差还要 ≤ -k·σ（k=2.5）才算一次显著低发；σ>4% 的"晴好天"按云影日处理（不计入也不打断持续度）
- **间歇停机/掉线**（v0.2）：白天 ≥1h 本串 ≤ 群组同时段 10% 而群组 ≥ 峰值 30% → 停机签名，近 7 日 ≥2 天判 B；停机日不计入互比持续度；停机不等于遮挡（遮挡不会到 0）
- **冷启动**：晴好天样本 <5 → 只做同日互比，仅凭偏差最高给 D（v0.2），置信度下调；全站集体衰减用机间互比 + 站级趋势提示兜底
- **数据冻结**（v0.2）：日间电流连续 ≥90 分钟一动不动 → 半天钉死，该日不参与互比中位数，连续 2 天判 A 数据问题；冻结机器的功率平顶不标限发
- **案例连续性**：同一串的异常跨天是一个案例（suspect_id 固定），分级只升不降；驳回的案例 30 天内不重复报警（除非恶化 ≥5 个百分点或升级）；连续 3 个晴好天回到正常带自愈关闭
- **损失**：仅晴好天累计，`Σ时段 max(0, 群组同时段中位电流×基线比 − 实际电流) × 群组电压 × 0.5h`，限发时段不计，峰谷平电价；起算日取 suspect_id 中的推定起始日

## 5. 目录

```
pvdiag/
  config.py        口径参数（YAML）         storage.py   SQLite + Parquet（老库自动补列迁移）
  models.py        统一数据模型/枚举        features.py  日特征/天气/平顶/钉死时长
  adapters/        huawei ginlong sungrow csv（QuotaPlanner）
  agents/          ingest profile baseline detect classify loss review tuning workorder region monitor
  orchestrator.py  DailyPipeline（run/backfill/报告；运行口径 = yaml + 自调参覆盖层）
  health.py        健康自检（pvdiag doctor）   notify.py    关键事件推送（企业微信 / outbox，24h 去重）
  solar.py         晴空倾斜面/跟踪 POA：季节性朝向归一、朝向拟合   llm.py   LLM 分析助手（可关，无 key 模板）
  sim/soak.py      连续运行压测（模拟复核员/主管/现场）
  public/          公开数据：datasets 下载校验 · kaggle_solar · nrel · bench（打分/弱标注训练/报告）
  sim/synth.py     合成电站 + 12 种故障注入   web/         FastAPI 复核界面（服务端 SVG 图，无外网依赖）
  cli.py           pvdiag 命令行（含 doctor / tune / gate / model / public / bench）
tests/             端到端召回/误报、闭环、训练、自主运行、公开数据基准、单元
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
2. 站间对比气象源：v0.4 已接 NASA POWER / Meteonorm / Open-Meteo（`irradiance.providers`，默认关闭，打开即用）；有辐照仪数据走 measured 优先
3. 缺组件参数的站：ProfileAgent 每日报告缺参串；损失用实测 V/I 不依赖铭牌，缺电压时标"待量化"
4. 冷启动口径：`baseline.min_clear_days`，不足只做同日互比
5. 已知遮挡备案：`strings.known_shading / shading_expected_dev`，备案由谁维护需定
6. 复核责任人与频率：`pvdiag review batch --station X` 生成周批清单
7. 数据合规：全部本地 SQLite/Parquet，私有化部署无出站
8. 跟踪支架/双面组件、9. 微逆/优化器：v0.4 已建模（`mount` / `module_type` / `device_type`），接入前仍需盘点并写进档案
10. 运行归属：CLI 可在本机 Hermes 定时任务或服务器 cron 下运行，状态/配额/告警都在库里

## 7. 已知限制 / 下一步

- 三家云 API 适配器按公开接口实现，**字段名与限流细节需用 F:\Hermes 现有工具包和真实凭证联调**
- 天气标记无外部气象数据时只看功率，"雨天"用阴雨日近似；有辐照（气象站/卫星）时自动用它校正（Kaggle 基准即如此）
- 规则阈值是经验起步值；v0.2 起由 TuningAgent 按复核结果自动在安全区间内调；ML 模型需 ≥60 条标注后自动训练，考试达标才上线
- 公开数据的"参考标签"是统计得出的弱标注，不是现场确认；弱标注训练验证的是链路和闸门，上生产仍以复核结论为准
- Kaggle 数据是逆变器级，"同机互比"在那里是"同站各逆变器互比"；组串级真实数据的考试需要接入自有电站后用复核结论做
- 性能（v0.3）：72 串×20 天回放 16 s（v0.2 为 47 s）；360 串每日全链路约 2 s；按此线性外推万串级约 1 分钟/天；接入特征计算约 4 ms/串·日
- v0.4 补齐：双轴跟踪、双面组件、微逆/优化器（功率输入）、并联串数、按站并行、辐照源（NASA POWER / Meteonorm / Open-Meteo）与站间互比；云平台适配器真实字段仍待凭证联调
- 日电量模式只有量没有形状：遮挡 / 积灰 / 组件故障分不开（统一 B 待复核），朝向与串长靠"阴天是否消失 / 是否与天气无关"区分；真实站基准见 [docs/日电量模式与真实电站基准.md](docs/日电量模式与真实电站基准.md)
- 季节性朝向模型是简化晴空模型（直射 + 各向同性散射），用于比值不用于绝对发电量；纬度取 stations.lat，缺省 30°
