"""存储层：
- SQLite：档案、日特征、基线、嫌疑、判定、标注、工单、损失台账、流水线日志、告警
- Parquet：组串/逆变器 5 分钟明细，按 station_id/date 分区（每站每天约 5.5MB，10站×365天≈20GB/年）
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS stations (
    station_id TEXT PRIMARY KEY, name TEXT, platform TEXT, capacity_kwp REAL,
    region TEXT, lat REAL, lon REAL, tariff_json TEXT, extra_json TEXT
);
CREATE TABLE IF NOT EXISTS inverters (
    inverter_id TEXT PRIMARY KEY, station_id TEXT, name TEXT, model TEXT,
    rated_ac_kw REAL, dc_kwp REAL, extra_json TEXT
);
CREATE TABLE IF NOT EXISTS strings (
    string_id TEXT PRIMARY KEY, station_id TEXT, inverter_id TEXT, mppt INTEGER, pv_index INTEGER,
    n_modules INTEGER, module_model TEXT, module_wp REAL, module_vmp REAL, module_imp REAL,
    azimuth REAL, tilt REAL, roof_area TEXT, known_shading INTEGER DEFAULT 0, device_type TEXT, module_type TEXT, n_parallel INTEGER,
    shading_expected_dev REAL, shading_note TEXT, group_key TEXT, notes TEXT, updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_strings_inv ON strings(inverter_id);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, scope TEXT, target_id TEXT, event_type TEXT,
    event_date TEXT, notes TEXT, created_at TEXT
);
CREATE TABLE IF NOT EXISTS daily_features (
    string_id TEXT, date TEXT, n_samples INTEGER, n_daytime_expected INTEGER, n_missing INTEGER,
    noon_median_i REAL, noon_median_v REAL, noon_n INTEGER, noon_zero_frac REAL, day_max_i REAL, day_energy_kwh REAL,
    daytime_i_std REAL, zero_all_day INTEGER, stuck INTEGER, stale_run_min INTEGER, i_profile_json TEXT, v_profile_json TEXT,
    PRIMARY KEY (string_id, date)
);
CREATE INDEX IF NOT EXISTS idx_df_date ON daily_features(date);
CREATE TABLE IF NOT EXISTS inverter_daily (
    inverter_id TEXT, date TEXT, pac_noon_median REAL, pac_peak REAL, pac_noon_cv REAL,
    limited_flag INTEGER, limited_slots_json TEXT, offline_flag INTEGER, pac_profile_json TEXT,
    PRIMARY KEY (inverter_id, date)
);
CREATE TABLE IF NOT EXISTS station_daily (
    station_id TEXT, date TEXT, noon_power REAL, peak_ref REAL, noon_cv REAL, weather_tag TEXT,
    irradiance_kwh_m2 REAL, rain INTEGER, irradiance_source TEXT, PRIMARY KEY (station_id, date)
);
CREATE TABLE IF NOT EXISTS baselines (
    string_id TEXT, date TEXT, segment_start TEXT, group_key TEXT, group_n INTEGER, group_level TEXT,
    group_median_i REAL, ratio_today REAL, baseline_ratio REAL, baseline_n INTEGER,
    dev_today REAL, dev_vs_own REAL, dev_roll7 REAL, persist_days_5 INTEGER, persist_days_3 INTEGER,
    first_breach_date TEXT, v_dev REAL, overcast_dev REAL, n_clear_recent INTEGER, cold_start INTEGER,
    weather_tag TEXT, limited INTEGER, group_noise REAL, orient_exp_ratio REAL, PRIMARY KEY (string_id, date)
);
CREATE INDEX IF NOT EXISTS idx_bl_date ON baselines(date);
CREATE TABLE IF NOT EXISTS inverter_baselines (
    inverter_id TEXT, date TEXT, norm_noon REAL, station_median_norm REAL, station_n INTEGER, inv_dev REAL,
    persist_days INTEGER, weather_tag TEXT, PRIMARY KEY (inverter_id, date)
);
CREATE TABLE IF NOT EXISTS signatures (
    string_id TEXT, date TEXT, weather_tag TEXT, ratio_mean REAL, ratio_std REAL, am_pm_skew REAL,
    dip_depth REAL, dip_start TEXT, dip_end TEXT, outside_dip_ratio REAL, voltage_offset REAL, outage_slots INTEGER, tracker_flag INTEGER,
    PRIMARY KEY (string_id, date)
);
CREATE TABLE IF NOT EXISTS suspects (
    suspect_id TEXT PRIMARY KEY, string_id TEXT, inverter_id TEXT, station_id TEXT, date TEXT, first_date TEXT,
    scope TEXT, grade TEXT, rule_cause TEXT, dev_today REAL, persist_days INTEGER,
    signature_json TEXT, low_confidence INTEGER, archived_shading INTEGER, status TEXT, created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_susp_date ON suspects(date);
CREATE INDEX IF NOT EXISTS idx_susp_string ON suspects(string_id);
CREATE TABLE IF NOT EXISTS suspect_daily (
    suspect_id TEXT, date TEXT, grade TEXT, rule_cause TEXT, dev_today REAL, persist_days INTEGER, low_confidence INTEGER,
    PRIMARY KEY (suspect_id, date)
);
CREATE TABLE IF NOT EXISTS verdicts (
    suspect_id TEXT PRIMARY KEY, model_version TEXT, cause TEXT, confidence REAL, action TEXT,
    explanation TEXT, proba_json TEXT, novelty_score REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT, suspect_id TEXT, reviewer TEXT, decision TEXT,
    true_cause TEXT, reject_reason TEXT, note TEXT, reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS workorders (
    wo_id TEXT PRIMARY KEY, station_id TEXT, roof_area TEXT, suspect_ids_json TEXT, string_ids_json TEXT,
    cause TEXT, symptom TEXT, checklist_json TEXT, loss_kwh_per_day REAL, loss_kwh_total REAL,
    loss_yuan_total REAL, priority REAL, status TEXT, channel TEXT, dispatch_result TEXT,
    created_at TEXT, sent_at TEXT, closed_at TEXT, field_cause TEXT, field_action TEXT, field_note TEXT,
    field_photos_json TEXT, recheck_due TEXT, recheck_result TEXT
);
CREATE TABLE IF NOT EXISTS loss_ledger (
    string_id TEXT, date TEXT, suspect_id TEXT, lost_kwh REAL, lost_yuan REAL, quantified INTEGER,
    PRIMARY KEY (string_id, date)
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT, agent TEXT, status TEXT,
    started_at TEXT, finished_at TEXT, metrics_json TEXT, message TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT, level TEXT, source TEXT,
    message TEXT, created_at TEXT, resolved INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS model_registry (
    model_version TEXT PRIMARY KEY, trained_at TEXT, n_labels INTEGER, classes_json TEXT,
    metrics_json TEXT, path TEXT, active INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS platform_state (
    key TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT
);
CREATE TABLE IF NOT EXISTS param_changes (
    change_id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT, param TEXT, old_value TEXT, new_value TEXT,
    reason TEXT, metrics_json TEXT, status TEXT, created_at TEXT, rolled_back_at TEXT
);
CREATE TABLE IF NOT EXISTS param_overrides (
    param TEXT PRIMARY KEY, value_json TEXT, updated_at TEXT, source TEXT
);
CREATE TABLE IF NOT EXISTS health_checks (
    check_id INTEGER PRIMARY KEY AUTOINCREMENT, run_date TEXT, name TEXT, status TEXT, detail TEXT, created_at TEXT
);
"""

# 老库升级：按表逐列补齐（SQLite 的 CREATE TABLE IF NOT EXISTS 不会加新列）
_MIGRATIONS = {
    "daily_features": {"stale_run_min": "INTEGER", "noon_zero_frac": "REAL"},
    "baselines": {"group_noise": "REAL", "orient_exp_ratio": "REAL"},
    "station_daily": {"irradiance_source": "TEXT"},
    "strings": {"device_type": "TEXT", "module_type": "TEXT", "n_parallel": "INTEGER"},
    "signatures": {"outage_slots": "INTEGER", "tracker_flag": "INTEGER"},
    "model_registry": {"status": "TEXT", "holdout_json": "TEXT"},
    "pipeline_runs": {"attempt": "INTEGER"},
}


def _d(x) -> str:
    """date/datetime/str → ISO 日期字符串"""
    if isinstance(x, datetime):
        return x.date().isoformat()
    if isinstance(x, date):
        return x.isoformat()
    return str(x)[:10]


class Store:
    def __init__(self, db_path: Path | str, samples_dir: Path | str):
        self.db_path = Path(db_path)
        self.samples_dir = Path(samples_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """自愈：老数据库自动补齐新列，不需要人工迁移脚本"""
        for table, cols in _MIGRATIONS.items():
            have = set(self.table_columns(table))
            for col, typ in cols.items():
                if col not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
        self._conn.commit()

    def quick_check(self) -> str:
        """SQLite 完整性快速检查：返回 'ok' 或错误描述"""
        try:
            return str(self._conn.execute("PRAGMA quick_check").fetchone()[0])
        except Exception as e:  # pragma: no cover
            return f"error: {e}"

    # ------------------------------------------------------------ 基础 SQL
    @contextmanager
    def tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def execute(self, sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
        with self.tx() as c:
            return c.execute(sql, params)

    def executemany(self, sql: str, rows: Iterable[Sequence[Any]]) -> None:
        with self.tx() as c:
            c.executemany(sql, rows)

    def query(self, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self._conn, params=list(params))

    def scalar(self, sql: str, params: Sequence[Any] = ()) -> Any:
        cur = self._conn.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else None

    def upsert_df(self, table: str, df: pd.DataFrame) -> int:
        """INSERT OR REPLACE 整个 DataFrame（列名需与表一致；多余列忽略）"""
        if df is None or df.empty:
            return 0
        cols = [c for c in df.columns if c in self.table_columns(table)]
        df = df[cols].copy()
        for c in cols:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
            elif df[c].dtype == object:
                df[c] = df[c].map(lambda v: v.isoformat() if isinstance(v, (date, datetime)) else v)
            elif df[c].dtype == bool:
                df[c] = df[c].astype(int)
        placeholders = ",".join("?" * len(cols))
        sql = f"INSERT OR REPLACE INTO {table} ({','.join(cols)}) VALUES ({placeholders})"
        rows = [tuple(None if (isinstance(v, float) and pd.isna(v)) else (v.item() if hasattr(v, "item") else v)
                      for v in r) for r in df.itertuples(index=False, name=None)]
        self.executemany(sql, rows)
        return len(rows)

    def table_columns(self, table: str) -> List[str]:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        return [r[1] for r in cur.fetchall()]

    # ------------------------------------------------------------ 平台状态 (token/配额)
    def set_state(self, key: str, value: Any) -> None:
        self.execute("INSERT OR REPLACE INTO platform_state VALUES (?,?,?)",
                     (key, json.dumps(value, ensure_ascii=False, default=str), datetime.now().isoformat()))

    def get_state(self, key: str, default: Any = None) -> Any:
        v = self.scalar("SELECT value_json FROM platform_state WHERE key=?", (key,))
        return json.loads(v) if v is not None else default

    # ------------------------------------------------------------ 5 分钟明细 (Parquet)
    def _part_path(self, kind: str, station_id: str, d: date | str) -> Path:
        p = self.samples_dir / kind / station_id
        p.mkdir(parents=True, exist_ok=True)
        return p / f"{_d(d)}.parquet"

    def write_samples(self, kind: str, station_id: str, df: pd.DataFrame) -> int:
        """kind: 'string' (string_id, ts, current_a, voltage_v) | 'inverter' (inverter_id, ts, pac_kw, status)"""
        if df.empty:
            return 0
        df = df.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        n = 0
        for d, part in df.groupby(df["ts"].dt.date):
            path = self._part_path(kind, station_id, d)
            if path.exists():   # 断点续拉：与已有数据合并去重
                old = pd.read_parquet(path)
                part = pd.concat([old, part], ignore_index=True)
                key = ["string_id", "ts"] if kind == "string" else ["inverter_id", "ts"]
                part = part.drop_duplicates(subset=key, keep="last")
            part.sort_values("ts").to_parquet(path, index=False)
            n += len(part)
        return n

    def read_samples(self, kind: str, station_id: str, d: date | str) -> pd.DataFrame:
        path = self.samples_dir / kind / station_id / f"{_d(d)}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def available_dates(self, kind: str, station_id: str) -> List[str]:
        p = self.samples_dir / kind / station_id
        if not p.exists():
            return []
        return sorted(f.stem for f in p.glob("*.parquet"))

    # ------------------------------------------------------------ 常用读取
    def stations(self) -> pd.DataFrame:
        return self.query("SELECT * FROM stations ORDER BY station_id")

    def inverters(self, station_id: Optional[str] = None) -> pd.DataFrame:
        if station_id:
            return self.query("SELECT * FROM inverters WHERE station_id=? ORDER BY inverter_id", (station_id,))
        return self.query("SELECT * FROM inverters ORDER BY inverter_id")

    def strings(self, station_id: Optional[str] = None) -> pd.DataFrame:
        if station_id:
            return self.query("SELECT * FROM strings WHERE station_id=? ORDER BY inverter_id, pv_index", (station_id,))
        return self.query("SELECT * FROM strings ORDER BY station_id, inverter_id, pv_index")

    def events(self) -> pd.DataFrame:
        return self.query("SELECT * FROM events ORDER BY event_date")

    def daily_features(self, station_id: str, start: date | str, end: date | str) -> pd.DataFrame:
        return self.query(
            """SELECT f.* FROM daily_features f JOIN strings s ON s.string_id=f.string_id
               WHERE s.station_id=? AND f.date BETWEEN ? AND ? ORDER BY f.date""",
            (station_id, _d(start), _d(end)))

    def station_daily(self, station_id: str, start: date | str, end: date | str) -> pd.DataFrame:
        return self.query("SELECT * FROM station_daily WHERE station_id=? AND date BETWEEN ? AND ? ORDER BY date",
                          (station_id, _d(start), _d(end)))

    def baselines(self, station_id: str, start: date | str, end: date | str) -> pd.DataFrame:
        return self.query(
            """SELECT b.* FROM baselines b JOIN strings s ON s.string_id=b.string_id
               WHERE s.station_id=? AND b.date BETWEEN ? AND ? ORDER BY b.date""",
            (station_id, _d(start), _d(end)))

    def log_run(self, run_date: date | str, agent: str, status: str, started: datetime,
                metrics: Optional[Dict[str, Any]] = None, message: str = "") -> None:
        self.execute(
            "INSERT INTO pipeline_runs (run_date, agent, status, started_at, finished_at, metrics_json, message) "
            "VALUES (?,?,?,?,?,?,?)",
            (_d(run_date), agent, status, started.isoformat(), datetime.now().isoformat(),
             json.dumps(metrics or {}, ensure_ascii=False, default=str), message))

    def alert(self, run_date: date | str, level: str, source: str, message: str) -> None:
        self.execute("INSERT INTO alerts (run_date, level, source, message, created_at) VALUES (?,?,?,?,?)",
                     (_d(run_date), level, source, message, datetime.now().isoformat()))

    def close(self) -> None:
        self._conn.close()
