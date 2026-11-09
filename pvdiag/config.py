"""全局配置：所有判定口径、窗口、阈值集中在这里，通过 YAML 可调。

口径说明（来自需求表）：
- 判定标准不取"某一天"，取统计基线：晴好天中午窗口(11:30-13:30) 同逆变器各串电流中位数互比
- 阈值 -3% / -5% 为起步值，上线后按人工复核确认率调优
- 偏差 ≤ -5% 且连续 ≥ 2 个晴好天判 B 持续低发；全天 0 电流多日 / 电流钉死 / 电压异位判 A 故障
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class Thresholds(BaseModel):
    """偏差分级阈值（负数表示低于群组中位数）"""
    normal_floor: float = -0.03          # 偏差 > -3%  → 正常
    fault_floor: float = -0.05           # 偏差 ≤ -5%  → 持续低发候选
    fault_persist_days: int = 2          # B 级需要连续 ≥2 个晴好天
    observe_persist_days: int = 3        # D 级：-5%~-3% 连续 ≥3 天追踪
    zero_current_a: float = 0.5          # <0.5A 视为未接/夜间
    zero_days: int = 2                   # 全天 0 电流连续天数 → A 故障
    stuck_std_a: float = 0.02            # 日间电流标准差低于此值且均值>0.5A → 电流钉死
    stale_run_min: int = 90              # 日间连续 ≥90 分钟电流一动不动 → 半天钉死（数据问题），该日不参与互比中位数
    voltage_offset: float = 0.08         # 与群组中位电压偏差 > 8% → 电压异位
    voltage_offset_days: int = 2
    dip_ratio: float = 0.85              # 分时段比值 < 0.85 视为凹陷（遮挡/二极管签名）
    dip_min_slots: int = 2               # 凹陷至少持续 2 个半小时槽 (1h)
    dip_normal_outside: float = 0.95     # 凹陷之外比值 ≥0.95 才算"分时段"签名
    orientation_overcast_abs: float = 0.02  # 阴天偏差绝对值 < 2% → 朝向差异候选
    orientation_max_drift: float = 0.012    # 近 10 晴好天偏差 std 超过此值 → 在漂移，不判朝向（积灰/衰减）
    archived_shading_extra: float = -0.05   # 已备案遮挡串，比备案偏差再低 5% 才重新报警
    rainy_streak_caution_days: int = 7      # 连续 ≥7 天无晴好天后的首个晴天 → 低置信
    # ---- 群组噪声门控（来自公开数据 Kaggle 两站的教训：大电站"晴好天"各逆变器互比偏差的稳健标准差可达 2%~7%）
    noise_k: float = 2.5                    # 当日偏差还要 ≤ -k×群组稳健标准差(MAD) 才算"统计显著"的一次低发
    noise_skip: float = 0.04                # 群组稳健标准差 > 4% 的"晴好天"视为云影日：不计入持续度、也不打断持续度
    cold_start_cap_grade: str = "D"         # 冷启动（晴好天样本不足）期间，仅凭互比偏差最高只给到此分级（不给 B）
    outage_min_slots: int = 2               # 白天 ≥2 个半小时槽（1h）整段 0 输出而同机其它串正常 → 间歇停机/掉线签名
    outage_group_ratio: float = 0.30        # 判停机时要求群组曲线 ≥ 当日峰值 30%（排除清晨/傍晚）
    outage_rel_zero: float = 0.10           # 本串 ≤ 群组同时段 10% 即视为"整段为 0"（晴天散射光占比 10%~15%，遮挡不会更低；15 分钟边沿插值也不会）
    tracker_side_ratio: float = 0.85        # 跟踪电站：清晨与傍晚相对群组比值都 ≤0.85 且中午 ≥0.93 → 支架卡滞签名
    outage_noon_skip_frac: float = 0.20     # 中午窗口 ≥20% 样本为 0 的日子不计入互比持续度（停机已有独立签名，别把它算成"低发"）


class BaselineCfg(BaseModel):
    noon_start: str = "11:30"
    noon_end: str = "13:30"
    rolling_clear_days: int = 30         # 滚动基线窗口（晴好天数）
    min_clear_days: int = 5              # 低于此样本数 → 冷启动降级：只做同日互比
    clear_day_ratio: float = 0.35        # 全站中午功率 ≥ 晴天峰值 35% 才算晴好天
    clear_cv_max: float = 0.15           # 中午窗口功率变异系数 ≤0.15 (抗云影抖动)
    peak_lookback_days: int = 60         # 晴天峰值参考 = 近 60 天中午功率 95 分位
    peak_fallback_capacity_ratio: float = 0.70  # 无历史时用装机容量×0.70 兜底
    min_group_size: int = 3              # 互比群组最少串数
    limited_run_slots: int = 6           # 平顶：连续 ≥6 个 5 分钟点功率几乎不变 → 限发
    limited_flat_tol: float = 0.003      # 平顶判定相对变化 ≤0.3%
    slot_minutes: int = 30               # 曲线签名向量粒度（半小时）
    daylight_start: str = "05:00"
    daylight_end: str = "19:30"
    tz_meridian_deg: float = 120.0       # 时区中央经线（Asia/Shanghai = 120°E）：太阳几何用 stations.lon 换算时钟时间 → 太阳时


class Confidence(BaseModel):
    high: float = 0.85                   # ≥ → 直接派单（草稿/自动）
    mid: float = 0.50                    # ≥ → 转人工复核；否则仅观察
    min_labels_for_ml: int = 60          # 标注数不足 → 规则代理模型（冷启动）
    retrain_every_days: int = 30
    # ---- 模型生命周期（自训练 / 自评估 / 自切换 / 自回退）
    retrain_min_new_labels: int = 20     # 距上次训练新增标注 ≥N 条即触发重训（不必等周期）
    holdout_fraction: float = 0.25       # 按时间顺序留出最近 25%（至少 10 条）做"考试"，不用随机切分（避免未来信息泄漏）
    promote_tolerance: float = 0.02      # 挑战者准确率 ≥ 现役 − 2% 且 ≥ 规则代理 − 2% 才上线
    demote_margin: float = 0.10          # 线上：模型与复核结论的一致率比规则代理低 10 个百分点以上 → 自动降级回规则代理
    demote_min_samples: int = 20         # 降级判断至少 20 条近 30 天复核样本
    drift_psi_retrain: float = 0.25      # 偏差分布 PSI 超过此值且有新标注 → 触发重训


class DispatchCfg(BaseModel):
    mode: str = "gated"                  # draft: 永远草稿待人工确认; auto: 门控通过即自动发送; gated: 门控连续达标自动放开、恶化自动收回（默认）
    channel: str = "dryrun"              # dryrun | wecom
    wecom_webhook: Optional[str] = None  # 企业微信群机器人 webhook
    auto_gate_fa_rate_max: float = 0.10  # 误报率连续1个月<10% 才允许 auto
    auto_gate_window_days: int = 30
    auto_gate_min_reviews: int = 30      # 窗口内复核样本至少 30 条
    gate_open_streak_days: int = 7       # gated：门控连续通过 ≥7 个运行日才放开自动派单（滞回，防抖）
    gate_close_field_missed: int = 3     # gated：窗口内现场漏报 ≥3 起立即收回
    gate_close_on_pipeline_fail: bool = True   # gated：当天任一智能体失败 / 数据时延 >2 天 → 收回（数据管道不健康时不自动派人）
    recheck_clear_days: int = 3          # 处理后 N 个晴好天复查
    site_visit_cost_yuan: float = 300.0  # 误派一次现场核查成本（用于优先级与门控）
    merge_same_roof: bool = True         # 同屋面多串合并区域工单


class LossCfg(BaseModel):
    default_price_yuan_per_kwh: float = 0.60
    # 峰谷平时段（小时起止，含起不含止）；可按站覆盖
    tou: Dict[str, List[List[int]]] = Field(default_factory=lambda: {
        "peak": [[10, 12], [14, 17]],
        "flat": [[7, 10], [12, 14], [17, 19]],
        "valley": [[0, 7], [19, 24]],
    })
    tou_price: Dict[str, float] = Field(default_factory=lambda: {
        "peak": 0.90, "flat": 0.60, "valley": 0.30
    })
    expected_fault_days_default: int = 30   # 未修复预计持续天数（优先级公式用）
    cleaning_cost_per_kwp: float = 15.0     # 全站清洗单价（元/kWp），用于积灰清洗回本估算
    repair_cost_by_cause: Dict[str, float] = Field(default_factory=lambda: {
        "断线/未接": 300, "旁路二极管": 500, "遮挡": 400, "朝向差异": 0,
        "MPPT异常": 1500, "组件衰减/积灰": 200, "数据问题": 50, "限发": 0, "间歇停机/掉线": 200, "跟踪支架卡滞": 400, "未知": 300,
    })


class HuaweiQuota(BaseModel):
    """华为北向接口配额：日限额 = Σ⌈设备数/100⌉ + 24（登录也计入），每批 ≤4 天"""
    base_quota: int = 24
    devices_per_unit: int = 100
    batch_days: int = 4
    login_cost: int = 1
    token_ttl_minutes: int = 30


class PlatformCfg(BaseModel):
    huawei: HuaweiQuota = HuaweiQuota()
    ginlong_pac_unit_divisor: float = 1000.0   # 锦浪 pac 单位是 W → ÷1000
    sungrow_token_ttl_hours: int = 48
    sungrow_same_day_available: bool = False  # 阳光不返回当天数据


class AutotuneCfg(BaseModel):
    """自调参闭环：用人工复核结果给规则阈值"考试"，只在安全区间内小步调整，留痕、可回滚。"""
    enabled: bool = True
    mode: str = "apply"                  # apply: 达标自动生效（写 param_overrides，不改 config.yaml）; shadow: 只评估、只建议
    min_labels: int = 20                 # 窗口内"纯偏差型"复核样本至少 20 条才动阈值
    window_days: int = 60                # 评估窗口
    cooldown_days: int = 7               # 两次调整至少间隔 7 天
    min_f1_gain: float = 0.05            # 候选参数在留出的复核样本上 F1 至少提升 5 个百分点
    min_recall_keep: float = 0.80        # 调整后对"已确认"案例的召回不得低于当前的 80%
    step_floor: float = 0.01             # 阈值步长 1 个百分点
    fault_floor_bounds: List[float] = Field(default_factory=lambda: [-0.08, -0.04])
    persist_bounds: List[int] = Field(default_factory=lambda: [2, 4])
    noise_k_bounds: List[float] = Field(default_factory=lambda: [2.0, 4.0])
    rollback_confirm_drop: float = 0.10  # 生效后 B 级确认率比生效前低 ≥10 个百分点（≥10 条新复核）→ 自动回滚
    loosen_on_field_missed: int = 3      # 窗口内现场漏报 ≥3 起 → 提议放松一步（放松无法用已有标注验证，只提议不自动生效）


class IngestCfg(BaseModel):
    """数据接入自愈：缺口自动补拉、失败重试退避、降级到本地文件、连续失败升级告警。"""
    heal_lookback_days: int = 14         # 每次运行回看 14 天，缺哪天补哪天（不只是 backfill_days 窗口）
    max_heal_days_per_run: int = 7       # 单次最多补 7 天（控制配额）
    retries: int = 3                     # 适配器拉取失败重试次数
    backoff_seconds: float = 5.0         # 指数退避起始秒数（5, 10, 20）
    escalate_after_failures: int = 3     # 连续失败 ≥3 次升级为 error 告警并推送
    stale_data_alert_days: int = 2       # 数据时延 >2 天告警


class AlertCfg(BaseModel):
    """关键告警推送（企业微信或 outbox 文件），24h 内相同内容不重复推。"""
    enabled: bool = True
    wecom_webhook: Optional[str] = None  # 为空则沿用 dispatch.wecom_webhook；都为空 → 写 data/outbox/alerts/<date>.md
    min_level: str = "error"             # error | warn：推送级别门槛（门控开合、模型上线/降级、参数生效/回滚 总是推送）
    dedupe_hours: int = 24


class IrradianceCfg(BaseModel):
    """辐照源（按 providers 顺序回退）：
    measured   平台/CSV 实测（含 Meteonorm 导出的小时文件：列 G_Gh）
    nasa_power NASA POWER 日 GHI（ALLSKY_SFC_SW_DWN，kWh/m²/天，免费无 key，滞后数天）
    meteonorm  Meteonorm REST API（需 METEONORM_TOKEN；小时 G_Gh 聚合成日）
    openmeteo  Open-Meteo ERA5 再分析（免费无 key）
    clearsky   晴空模型（永远可用，仅在晴好天作分母有意义）
    出站的提供方都默认关闭（私有化部署不出站）；source=none 完全不填辐照。"""
    source: str = "auto"                 # auto | none
    providers: List[str] = Field(default_factory=lambda: ["measured", "nasa_power", "meteonorm", "openmeteo", "clearsky"])
    nasa_power: bool = False
    nasa_power_base_url: str = "https://power.larc.nasa.gov/api/temporal/daily/point"
    meteonorm: bool = False
    meteonorm_base_url: str = "https://api.meteonorm.com/v1"
    meteonorm_token_env: str = "METEONORM_TOKEN"
    openmeteo: bool = False
    openmeteo_base_url: str = "https://archive-api.open-meteo.com"
    timeout_seconds: int = 30
    cache_days: int = 400                # 外部日辐照缓存在 platform_state，按站保留天数


class LLMCfg(BaseModel):
    """LLM 解释与复核助手（可选）：没有 key 就用模板文案，功能不缺；私有化部署可 enabled=false 彻底关闭出站。"""
    enabled: bool = True
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    base_url: str = "https://api.anthropic.com"
    max_tokens: int = 700
    timeout_seconds: int = 40


class Settings(BaseModel):
    data_dir: Path = Path("./data")
    timezone: str = "Asia/Shanghai"
    thresholds: Thresholds = Thresholds()
    baseline: BaselineCfg = BaselineCfg()
    confidence: Confidence = Confidence()
    dispatch: DispatchCfg = DispatchCfg()
    loss: LossCfg = LossCfg()
    platforms: PlatformCfg = PlatformCfg()
    autotune: AutotuneCfg = AutotuneCfg()
    ingest: IngestCfg = IngestCfg()
    alerts: AlertCfg = AlertCfg()
    llm: LLMCfg = LLMCfg()
    irradiance: IrradianceCfg = IrradianceCfg()

    # ---- 派生路径 ----
    @property
    def db_path(self) -> Path:
        return self.data_dir / "pvdiag.sqlite"

    @property
    def samples_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def outbox_dir(self) -> Path:
        return self.data_dir / "outbox"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    def ensure_dirs(self) -> None:
        for p in (self.data_dir, self.samples_dir, self.reports_dir, self.outbox_dir, self.models_dir):
            p.mkdir(parents=True, exist_ok=True)

    # ---- 读写 ----
    @classmethod
    def load(cls, path: Optional[os.PathLike | str] = None) -> "Settings":
        if path is None:
            path = os.environ.get("PVDIAG_CONFIG", "config.yaml")
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            return cls.model_validate(raw)
        return cls()

    def dump(self, path: os.PathLike | str) -> None:
        data = self.model_dump(mode="json")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def hhmm_to_minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)
