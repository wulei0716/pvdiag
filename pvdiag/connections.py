from datetime import date, timedelta

def input_evidence(store, station_id: str, d: date, zero_current_a: float = 0.5):
    """Return unused input IDs and each channel's last observed production date."""
    today = d.isoformat()
    last_production = dict(store.query(
        "SELECT f.string_id, MAX(f.date) AS last_date FROM daily_features f "
        "JOIN strings s ON s.string_id=f.string_id WHERE s.station_id=? AND f.date<? "
        "AND f.day_energy_kwh>0 AND f.noon_median_i>=? AND f.stuck=0 GROUP BY f.string_id",
        (station_id, today, zero_current_a)).itertuples(index=False, name=None))
    recent = store.daily_features(station_id, d - timedelta(days=7), d)
    observed_zero = recent[
        (recent["zero_all_day"] == 1) & (recent["day_energy_kwh"] == 0)
        & (recent["stuck"] == 0) & (recent["n_daytime_expected"] > 0)
        & (recent["n_missing"] <= recent["n_daytime_expected"] * 0.25)
    ]
    unused_by_history = set(observed_zero.groupby("string_id")["date"].nunique().loc[lambda n: n == 8].index)
    profiles = store.strings(station_id)
    documented = (profiles["connection_status"] == "connected") | (profiles["n_modules"].fillna(0) > 0)
    inferred = profiles["string_id"].isin(unused_by_history - last_production.keys()) & ~documented
    unused = set(profiles.loc[inferred | (profiles["connection_status"] == "unused"), "string_id"])
    return unused, last_production


def zero_output_context(store, station_id: str, day: str, zero_current_a: float):
    """Summarize same-inverter voltage and repeated zero channels by model."""
    frame = store.query(
        "SELECT t.string_id, t.inverter_id, t.pv_index, i.model, f.zero_all_day, f.stuck, "
        "f.noon_median_v, f.noon_median_i, f.noon_low_current_v, f.noon_low_current_v_n "
        "FROM daily_features f JOIN strings t USING(string_id) JOIN inverters i USING(inverter_id) "
        "WHERE t.station_id=? AND f.date=? AND COALESCE(t.connection_status,'unknown')!='unused'",
        (station_id, day),
    )
    if frame.empty:
        return {}
    producing = frame[(frame.noon_median_i >= zero_current_a) & (frame.stuck == 0)
                      & (frame.zero_all_day == 0) & (frame.noon_median_v > 0)]
    peers = producing.groupby("inverter_id").noon_median_v.agg(["median", "count"])
    sampled = frame[(frame.stuck == 0) & frame.model.notna() & frame.model.ne("")].copy()
    sampled["zero_inverter"] = sampled.inverter_id.where(sampled.zero_all_day == 1)
    patterns = sampled.groupby(["model", "pv_index"]).agg(
        reporting=("inverter_id", "nunique"), zero=("zero_inverter", "nunique"))
    result = {}
    for row in frame[(frame.zero_all_day == 1) & (frame.stuck == 0)].itertuples():
        evidence = {}
        if row.inverter_id in peers.index and row.noon_low_current_v_n >= 3:
            peer = peers.loc[row.inverter_id]
            if peer["count"] >= 2 and row.noon_low_current_v >= 0:
                evidence["voltage_ratio"] = round(float(row.noon_low_current_v / peer["median"]), 3)
                evidence["voltage_peer_count"] = int(peer["count"])
        key = (row.model, row.pv_index)
        if key in patterns.index and patterns.loc[key, "reporting"] >= 2:
            pattern = patterns.loc[key]
            evidence["same_model_channel_zero"] = int(pattern["zero"])
            evidence["same_model_channel_reporting"] = int(pattern["reporting"])
        if evidence:
            result[row.string_id] = evidence
    return result
