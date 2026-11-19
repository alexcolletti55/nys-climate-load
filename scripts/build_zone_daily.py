"""Merge cleaned load and weather into analysis-ready tables for the study zones.

Inputs  (data/cleaned/, from build_nyiso_hourly.py and clean_weather.py):
    nyiso_hourly_load.parquet, weather_hourly.parquet
Outputs (data/processed/):
    zone_hourly.parquet   one row per zone per hour (UTC): load + hour-averaged weather
    zone_daily.parquet    one row per zone per LOCAL calendar day: peak/energy + daily weather
    zone_daily.csv        same as zone_daily.parquet, for opening in Excel

Run from the repository root:
    python scripts/build_zone_daily.py

Key choices (all documented so they can be stated in the paper's methods):
  * Zone-station pairs: J (NYC) = LaGuardia, K (Long Island) = Islip, C (Central) = Syracuse.
  * NYISO time stamps are hour-BEGINNING, so the load for hour H..H+1 is paired with the
    average of the weather observations at H and H+1 (or whichever one exists).
  * Days are local (America/New_York) calendar days. A day's peak needs >= 20 hours of
    load; daily energy needs every hour of that day; daily weather needs >= 18 hours.
  * Relative humidity is recomputed from temperature and dewpoint (Magnus formula), and
    heat index uses the NWS algorithm (Rothfusz regression with NWS adjustments).
  * cdd_c / hdd_c use a fixed 18.3 C (65 F) base for description only; the regression
    step fits its own balance points.
  * Known major outages are FLAGGED (exclude=True), not deleted, so the choice is visible
    and reversible. COVID lockdown months are flagged separately (covid_lockdown) so the
    modeling step can decide whether to drop them.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

CLEANED = Path("data/cleaned")
PROCESSED = Path("data/processed")
LOCAL_TZ = "America/New_York"

ZONES = {  # zone -> (GHCN station ID, station name)
    "J": ("USW00014732", "LaGuardia"),
    "K": ("USW00004781", "Islip"),
    "C": ("USW00014771", "Syracuse"),
}

# Known major outages: (first local date, last local date, zones affected, reason).
# Rows in these windows are flagged exclude=True. Dates checked against the load data:
# e.g. Long Island daily-mean load fell from ~2,200 MW to 615 MW on 2012-10-30 (Sandy).
OUTAGES = [
    ("2003-08-14", "2003-08-17", {"J", "K", "C"}, "2003 Northeast blackout"),
    ("2011-08-27", "2011-09-02", {"J", "K"}, "Hurricane Irene"),
    ("2012-10-29", "2012-11-12", {"J", "K"}, "Superstorm Sandy"),
]
COVID = ("2020-03-16", "2020-06-30")

BASE_C = 18.3  # 65 F, descriptive degree-day base only
MIN_LOAD_HOURS_FOR_PEAK = 20
MIN_WEATHER_HOURS = 18


def rh_from_t_td(t_c, td_c):
    """Relative humidity (%) from temperature and dewpoint (deg C), Magnus formula."""
    a, b = 17.625, 243.04
    return 100 * np.exp(a * td_c / (b + td_c)) / np.exp(a * t_c / (b + t_c))


def heat_index_c(t_c, rh):
    """NWS heat index (deg C) from temperature (deg C) and RH (%)."""
    t = t_c * 9 / 5 + 32
    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    full = (-42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
            - 0.00683783 * t**2 - 0.05481717 * rh**2 + 0.00122874 * t**2 * rh
            + 0.00085282 * t * rh**2 - 0.00000199 * t**2 * rh**2)
    low_rh = (rh < 13) & (t >= 80) & (t <= 112)
    full = np.where(low_rh, full - ((13 - rh) / 4) * np.sqrt(np.clip((17 - np.abs(t - 95)) / 17, 0, None)), full)
    high_rh = (rh > 85) & (t >= 80) & (t <= 87)
    full = np.where(high_rh, full + ((rh - 85) / 10) * ((87 - t) / 5), full)
    hi_f = np.where((simple + t) / 2 >= 80, full, simple)
    return (hi_f - 32) * 5 / 9


def zone_hourly(load: pd.DataFrame, wx: pd.DataFrame, zone: str) -> pd.DataFrame:
    station, name = ZONES[zone]
    y = load[zone]
    y = y.loc[y.first_valid_index():]                       # J and K start Feb 2005

    w = (wx[wx["station"] == station]
         .drop_duplicates("time_utc").set_index("time_utc")[["temp_c", "dewpoint_c"]])
    full = pd.date_range(y.index.min(), y.index.max() + pd.Timedelta("1h"), freq="h", tz="UTC")
    w = w.reindex(full)
    nxt = w.shift(-1)
    w_avg = ((w + nxt) / 2).fillna(w).fillna(nxt).reindex(y.index)   # hour-beginning average

    h = pd.DataFrame({"zone": zone, "station": station, "station_name": name,
                      "load_mw": y, "temp_c": w_avg["temp_c"], "dewpoint_c": w_avg["dewpoint_c"]})
    h["rh_pct"] = rh_from_t_td(h["temp_c"], h["dewpoint_c"]).clip(upper=100)
    h["heat_index_c"] = heat_index_c(h["temp_c"], h["rh_pct"])
    h.index.name = "time_utc"
    return h.reset_index()


def flags_for(dates: pd.Series, zone: str) -> pd.DataFrame:
    exclude = pd.Series(False, index=dates.index)
    reason = pd.Series("", index=dates.index)
    for start, end, zones, why in OUTAGES:
        if zone in zones:
            m = dates.between(pd.Timestamp(start), pd.Timestamp(end))
            exclude |= m
            reason[m] = why
    covid = dates.between(pd.Timestamp(COVID[0]), pd.Timestamp(COVID[1]))
    return pd.DataFrame({"exclude": exclude, "exclude_reason": reason, "covid_lockdown": covid})


def zone_daily(h: pd.DataFrame) -> pd.DataFrame:
    h = h.copy()
    local = h["time_utc"].dt.tz_convert(LOCAL_TZ)
    h["date"] = local.dt.tz_localize(None).dt.normalize()
    h["hour_local"] = local.dt.hour
    g = h.groupby("date")

    d = pd.DataFrame({
        "n_hours": g.size(),
        "n_load_hours": g["load_mw"].count(),
        "energy_mwh": g["load_mw"].sum(min_count=1),
        "n_weather_hours": g["temp_c"].count(),
        "tmax_c": g["temp_c"].max(),
        "tmin_c": g["temp_c"].min(),
        "tmean_c": g["temp_c"].mean(),
        "dewpoint_mean_c": g["dewpoint_c"].mean(),
        "dewpoint_max_c": g["dewpoint_c"].max(),
        "heat_index_max_c": g["heat_index_c"].max(),
    })

    valid = h.dropna(subset=["load_mw"])
    pk = valid.loc[valid.groupby("date")["load_mw"].idxmax(),
                   ["date", "load_mw", "hour_local", "temp_c", "dewpoint_c", "heat_index_c"]]
    pk = pk.set_index("date")
    pk.columns = ["peak_mw", "peak_hour_local", "temp_at_peak_c", "dewpoint_at_peak_c",
                  "heat_index_at_peak_c"]
    d = d.join(pk)

    # Completeness rules
    peak_cols = ["peak_mw", "peak_hour_local", "temp_at_peak_c", "dewpoint_at_peak_c", "heat_index_at_peak_c"]
    d.loc[d["n_load_hours"] < MIN_LOAD_HOURS_FOR_PEAK, peak_cols] = np.nan
    d.loc[d["n_load_hours"] < d["n_hours"], "energy_mwh"] = np.nan
    wx_cols = ["tmax_c", "tmin_c", "tmean_c", "dewpoint_mean_c", "dewpoint_max_c", "heat_index_max_c"]
    d.loc[d["n_weather_hours"] < MIN_WEATHER_HOURS, wx_cols] = np.nan

    # Continuous daily index so "yesterday" really is the previous day
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D", name="date"))
    d["tmean_lag1_c"] = d["tmean_c"].shift(1)
    d["tmax_lag1_c"] = d["tmax_c"].shift(1)
    d["cdd_c"] = (d["tmean_c"] - BASE_C).clip(lower=0)
    d["hdd_c"] = (BASE_C - d["tmean_c"]).clip(lower=0)

    # Calendar
    hol = USFederalHolidayCalendar().holidays(d.index.min(), d.index.max())
    d["year"] = d.index.year
    d["month"] = d.index.month
    d["dow"] = d.index.day_of_week          # 0 = Monday
    d["is_weekend"] = d["dow"] >= 5
    d["is_holiday"] = d.index.isin(hol)

    d = d.reset_index()
    zone = h["zone"].iloc[0]
    d.insert(1, "zone", zone)
    d.insert(2, "station", h["station"].iloc[0])
    d = pd.concat([d, flags_for(d["date"], zone)], axis=1)
    return d


def main() -> None:
    load = pd.read_parquet(CLEANED / "nyiso_hourly_load.parquet")
    wx = pd.read_parquet(CLEANED / "weather_hourly.parquet")

    hourly_parts, daily_parts = [], []
    for zone in ZONES:
        h = zone_hourly(load, wx, zone)
        d = zone_daily(h)
        local_date = h["time_utc"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
        h = pd.concat([h, flags_for(local_date, zone)], axis=1)
        hourly_parts.append(h)
        daily_parts.append(d)

        usable = d[~d["exclude"]].dropna(subset=["peak_mw", "tmax_c"])
        print(f"Zone {zone} ({ZONES[zone][1]}): {d['date'].min():%Y-%m-%d} to {d['date'].max():%Y-%m-%d}, "
              f"{len(d):,} days, {len(usable):,} usable (peak + weather, not excluded), "
              f"{int(d['exclude'].sum())} days flagged as outages")

    hourly = pd.concat(hourly_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(PROCESSED / "zone_hourly.parquet", index=False)
    daily.to_parquet(PROCESSED / "zone_daily.parquet", index=False)
    daily.to_csv(PROCESSED / "zone_daily.csv", index=False, float_format="%.2f")
    print(f"Saved {PROCESSED / 'zone_hourly.parquet'} ({len(hourly):,} rows), "
          f"{PROCESSED / 'zone_daily.parquet'} and zone_daily.csv ({len(daily):,} rows)")


if __name__ == "__main__":
    main()
