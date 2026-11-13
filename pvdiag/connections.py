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
