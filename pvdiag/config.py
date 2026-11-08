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
    voltage_offset: float = 0.08         # 与群组中位电压偏差 > 8% → 电压异位
    voltage_offset_days: int = 2
    dip_ratio: float = 0.85              # 分时段比值 < 0.85 视为凹陷（遮挡/二极管签名）
    dip_min_slots: int = 2               # 凹陷至少持续 2 个半小时槽 (1h)
    dip_normal_outside: float = 0.95     # 凹陷之外比值 ≥0.95 才算"分时段"签名
    orientation_overcast_abs: float = 0.02  # 阴天偏差绝对值 < 2% → 朝向差异候选
    archived_shading_extra: float = -0.05   # 已备案遮挡串，比备案偏差再低 5% 才重新报警
    rainy_streak_caution_days: int = 7      # 连续 ≥7 天无晴好天后的首个晴天 → 低置信


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


class Confidence(BaseModel):
    high: float = 0.85                   # ≥ → 直接派单（草稿/自动）
    mid: float = 0.50                    # ≥ → 转人工复核；否则仅观察
    min_labels_for_ml: int = 60          # 标注数不足 → 规则代理模型（冷启动）
    retrain_every_days: int = 30


class DispatchCfg(BaseModel):
    mode: str = "draft"                  # draft: 生成草稿待人工确认; auto: 高置信自动发送
    channel: str = "dryrun"              # dryrun | wecom
    wecom_webhook: Optional[str] = None  # 企业微信群机器人 webhook
    auto_gate_fa_rate_max: float = 0.10  # 误报率连续1个月<10% 才允许 auto
    auto_gate_window_days: int = 30
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
    repair_cost_by_cause: Dict[str, float] = Field(default_factory=lambda: {
        "断线/未接": 300, "旁路二极管": 500, "遮挡": 400, "朝向差异": 0,
        "MPPT异常": 1500, "组件衰减/积灰": 200, "数据问题": 50, "限发": 0, "未知": 300,
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


class Settings(BaseModel):
    data_dir: Path = Path("./data")
    timezone: str = "Asia/Shanghai"
    thresholds: Thresholds = Thresholds()
    baseline: BaselineCfg = BaselineCfg()
    confidence: Confidence = Confidence()
    dispatch: DispatchCfg = DispatchCfg()
    loss: LossCfg = LossCfg()
    platforms: PlatformCfg = PlatformCfg()

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
