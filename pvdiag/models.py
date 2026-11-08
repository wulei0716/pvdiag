"""统一数据模型：三平台（华为/锦浪/阳光）接入后全部映射到这里定义的结构。"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 枚举口径
class Grade(str, Enum):
    A = "A"        # 故障：全天0电流多日 / 电流钉死 / 电压异位 → 直接派单
    B = "B"        # 持续低发：偏差 ≤-5% 连续 ≥2 晴好天 → 待人工复核
    C = "C"        # 疑朝向/档案差异：恒定比例偏差、阴天消失
    D = "D"        # 轻度：-5%~-3% 连续 ≥3 天
    E = "E"        # 观察：首次入表 / 未达持续度
    ARCHIVED = "已备案"   # 已备案遮挡，预期偏差内，不报警


class Cause(str, Enum):
    OPEN_CIRCUIT = "断线/未接"
    BYPASS_DIODE = "旁路二极管"
    SHADING = "遮挡"
    ORIENTATION = "朝向差异"
    MPPT = "MPPT异常"
    DEGRADATION_DUST = "组件衰减/积灰"
    DATA_ISSUE = "数据问题"
    CURTAILMENT = "限发"
    PROFILE_ERROR = "档案差异/串长"
    UNKNOWN = "未知"
    NORMAL = "正常(误报)"


CAUSE_CHOICES: List[str] = [c.value for c in Cause]

REJECT_REASONS: List[str] = ["云影", "限发", "数据脏", "已备案遮挡", "停机/检修", "其他"]


class Action(str, Enum):
    DISPATCH = "直接派单"
    REVIEW = "转人工复核"
    OBSERVE = "仅观察"


class SuspectStatus(str, Enum):
    OPEN = "open"
    REVIEWING = "reviewing"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    DISPATCHED = "dispatched"
    CLOSED = "closed"


class WorkOrderStatus(str, Enum):
    DRAFT = "draft"
    SENT = "sent"
    DONE = "done"            # 现场回填完成，待复查
    VERIFIED = "verified"    # 复查偏差归零
    REOPENED = "reopened"


class Weather(str, Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"    # 阴雨
    NODATA = "nodata"


class EventType(str, Enum):
    CLEANING = "cleaning"
    REPAIR = "repair"
    REWIRE = "rewire"
    INVERTER_REPLACE = "inverter_replace"
    MODULE_REPLACE = "module_replace"
    SHADING_REGISTERED = "shading_registered"
    MAINTENANCE = "maintenance"


# ---------------------------------------------------------------- 档案
class StationInfo(BaseModel):
    station_id: str
    name: str = ""
    platform: str = "csv"            # huawei | ginlong | sungrow | csv
    capacity_kwp: Optional[float] = None
    region: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    tariff: Dict[str, Any] = Field(default_factory=dict)   # 覆盖 LossCfg 的电价口径
    extra: Dict[str, Any] = Field(default_factory=dict)


class InverterInfo(BaseModel):
    inverter_id: str
    station_id: str
    name: str = ""
    model: str = ""
    rated_ac_kw: Optional[float] = None
    dc_kwp: Optional[float] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class StringProfile(BaseModel):
    """组串档案——AI 归因的地基。pv_index 与物理串对应关系须由现场排布图确认。"""
    string_id: str
    station_id: str
    inverter_id: str
    mppt: Optional[int] = None
    pv_index: Optional[int] = None            # pv1~pv36
    n_modules: Optional[int] = None
    module_model: str = ""
    module_wp: Optional[float] = None
    module_vmp: Optional[float] = None
    module_imp: Optional[float] = None
    azimuth: Optional[float] = None           # 朝向（度，南=180）
    tilt: Optional[float] = None
    roof_area: str = ""                       # 屋面/区域物理位置
    known_shading: bool = False               # 已备案固定遮挡
    shading_expected_dev: Optional[float] = None  # 备案遮挡的预期偏差（负数）
    shading_note: str = ""
    group_key: str = ""                       # 由档案智能体派生
    notes: str = ""

    @property
    def dc_wp(self) -> Optional[float]:
        if self.n_modules and self.module_wp:
            return self.n_modules * self.module_wp
        return None


class StringEvent(BaseModel):
    scope: str                                # string | inverter | station
    target_id: str
    event_type: EventType
    event_date: date
    notes: str = ""


# ---------------------------------------------------------------- 判定结果
class Signature(BaseModel):
    """全天曲线签名分型结果"""
    zero_all_day: bool = False
    stuck: bool = False
    voltage_offset: Optional[float] = None    # 与群组中位电压相对偏差
    dip_depth: Optional[float] = None         # 凹陷深度 (1 - 最低比值)
    dip_start: Optional[str] = None           # HH:MM
    dip_end: Optional[str] = None
    outside_dip_ratio: Optional[float] = None
    constant_offset: bool = False             # 全天恒定比例偏差
    ratio_mean: Optional[float] = None
    ratio_std: Optional[float] = None
    overcast_dev: Optional[float] = None      # 近期阴天偏差（朝向判定）
    orientation_like: bool = False
    inverter_level: bool = False              # 整机问题（机间互比）
    limited: bool = False
    notes: List[str] = Field(default_factory=list)


class Suspect(BaseModel):
    suspect_id: str
    string_id: Optional[str]
    inverter_id: str
    station_id: str
    date: date
    scope: str = "string"                     # string | inverter
    grade: str
    rule_cause: str
    dev_today: Optional[float] = None
    persist_days: int = 0
    signature: Signature = Signature()
    low_confidence: bool = False
    archived_shading: bool = False
    status: str = SuspectStatus.OPEN.value


class Verdict(BaseModel):
    suspect_id: str
    model_version: str
    cause: str
    confidence: float
    action: str
    explanation: str
    proba: Dict[str, float] = Field(default_factory=dict)
    novelty_score: Optional[float] = None


class WorkOrder(BaseModel):
    wo_id: str
    station_id: str
    roof_area: str = ""
    suspect_ids: List[str]
    string_ids: List[str]
    cause: str
    symptom: str
    checklist: List[str]
    loss_kwh_per_day: float = 0.0
    loss_kwh_total: float = 0.0
    loss_yuan_total: float = 0.0
    priority: float = 0.0
    status: str = WorkOrderStatus.DRAFT.value
    channel: str = "dryrun"
    created_at: datetime = Field(default_factory=datetime.now)
