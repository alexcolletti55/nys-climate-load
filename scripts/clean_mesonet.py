"""Clean NYS Mesonet 5-minute data into hourly tables (deg C, UTC).

Input  (never modified):
    data/raw/nysm/{STATION}.csv      5-minute observations as delivered by the NYS Mesonet
Outputs:
    data/cleaned/mesonet_hourly/{STATION}.parquet   one row per station-hour:
        time_utc (start of hour), station, temp_c, temp_max_c, temp_min_c, dewpoint_c,
        rh_pct, wind_ms, solar_wm2, n_obs_temp
    data/cleaned/mesonet_stations.csv   station, latitude, longitude, elevation_m,
        first/last valid hour, percent of hours complete (2016-2025)

Run from the repository root (takes several minutes; re-running skips stations whose
output is newer than the raw file):
    python scripts/clean_mesonet.py
Read all stations later with:  pd.read_parquet("data/cleaned/mesonet_hourly")

Method:
  * Hourly values average the twelve 5-minute readings from HH:00 to HH:55 and are labeled
    with the hour's START, matching NYISO's hour-beginning load (no shifting needed).
  * An hour needs >= 9 of its 12 readings (75%) for a variable to be kept; otherwise NaN.
  * Same physical limits as the NOAA cleaning (clean_weather.py): temperature -45..45 C,
    dewpoint -60..35 C and not more than 1 C above temperature, RH 0..100 %.
  * Data are cut off at 2025-12-31 23:00 UTC to match the NYISO and NOAA records
    (the Mesonet delivery runs to 2026-11-03).
  * NYS Mesonet data may not be redistributed: outputs stay in data/ (ignored by git).
    Cite Brotzge et al. (2020) and include the NYS Mesonet acknowledgment.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw/nysm")
OUT_DIR = Path("data/cleaned/mesonet_hourly")
STATIONS_OUT = Path("data/cleaned/mesonet_stations.csv")

END_UTC = pd.Timestamp("2026-01-01", tz="UTC")       # exclusive
START_COMPLETENESS = pd.Timestamp("2016-01-01", tz="UTC")
MIN_OBS_PER_HOUR = 9                                 # of 12 five-minute readings

TEMP_LIMITS = (-45.0, 45.0)
DEWPOINT_LIMITS = (-60.0, 35.0)
DEWPOINT_MAX_ABOVE_TEMP = 1.0

COLS = {  # raw column -> short name
    "station": "station",
    "time": "time",
    "latitude [degrees_north]": "latitude",
    "longitude [degrees_east]": "longitude",
    "elevation [m]": "elevation_m",
    "temp_2m [degC]": "temp_c",
    "relative_humidity [percent]": "rh_pct",
    "dewpoint [degC]": "dewpoint_c",
    "avg_wind_speed_merge [m/s]": "wind_ms",
    "solar_insolation [W/m^2]": "solar_wm2",
}


def clean_station(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, usecols=list(COLS), low_memory=False).rename(columns=COLS)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S UTC", utc=True)
    df = df[df["time"] < END_UTC]

    meta = {"station": path.stem,
            "latitude": df["latitude"].dropna().iloc[0] if df["latitude"].notna().any() else np.nan,
            "longitude": df["longitude"].dropna().iloc[0] if df["longitude"].notna().any() else np.nan,
            "elevation_m": df["elevation_m"].dropna().iloc[0] if df["elevation_m"].notna().any() else np.nan}

    # Physical plausibility (same limits as clean_weather.py)
    t, td = df["temp_c"], df["dewpoint_c"]
    bad_t = t.notna() & ((t < TEMP_LIMITS[0]) | (t > TEMP_LIMITS[1]))
    bad_td = td.notna() & ((td < DEWPOINT_LIMITS[0]) | (td > DEWPOINT_LIMITS[1])
                           | (td > t + DEWPOINT_MAX_ABOVE_TEMP))
    meta["removed_temp"] = int(bad_t.sum())
    meta["removed_dewpoint"] = int((bad_td | (bad_t & td.notna())).sum())
    df.loc[bad_t, "temp_c"] = np.nan
    df.loc[bad_t | bad_td, ["dewpoint_c", "rh_pct"]] = np.nan
    df.loc[(df["rh_pct"] < 0) | (df["rh_pct"] > 100), "rh_pct"] = np.nan

    # Hourly aggregation, labeled by start of hour
    df["time_utc"] = df["time"].dt.floor("h")
    g = df.groupby("time_utc")
    h = pd.DataFrame({
        "temp_c": g["temp_c"].mean(),
        "temp_max_c": g["temp_c"].max(),
        "temp_min_c": g["temp_c"].min(),
        "dewpoint_c": g["dewpoint_c"].mean(),
        "rh_pct": g["rh_pct"].mean(),
        "wind_ms": g["wind_ms"].mean(),
        "solar_wm2": g["solar_wm2"].mean(),
        "n_obs_temp": g["temp_c"].count(),
    })
    counts = {"temp": g["temp_c"].count(), "dew": g["dewpoint_c"].count(), "rh": g["rh_pct"].count(),
              "wind": g["wind_ms"].count(), "solar": g["solar_wm2"].count()}
    h.loc[counts["temp"] < MIN_OBS_PER_HOUR, ["temp_c", "temp_max_c", "temp_min_c"]] = np.nan
    h.loc[counts["dew"] < MIN_OBS_PER_HOUR, "dewpoint_c"] = np.nan
    h.loc[counts["rh"] < MIN_OBS_PER_HOUR, "rh_pct"] = np.nan
    h.loc[counts["wind"] < MIN_OBS_PER_HOUR, "wind_ms"] = np.nan
    h.loc[counts["solar"] < MIN_OBS_PER_HOUR, "solar_wm2"] = np.nan
    h = h[h[["temp_c", "dewpoint_c", "wind_ms", "solar_wm2"]].notna().any(axis=1)]

    float_cols = ["temp_c", "temp_max_c", "temp_min_c", "dewpoint_c", "rh_pct", "wind_ms", "solar_wm2"]
    h[float_cols] = h[float_cols].astype("float32").round(2)
    h["n_obs_temp"] = h["n_obs_temp"].astype("int8")
    h.insert(0, "station", path.stem)
    h = h.reset_index()

    valid = h.loc[h["temp_c"].notna(), "time_utc"]
    meta["first_valid_utc"] = valid.min() if len(valid) else pd.NaT
    meta["last_valid_utc"] = valid.max() if len(valid) else pd.NaT
    if len(valid):
        start = max(valid.min(), START_COMPLETENESS)
        expected = (END_UTC - start) / pd.Timedelta("1h")
        meta["pct_hours_complete"] = round(100 * (valid >= start).sum() / expected, 1)
    else:
        meta["pct_hours_complete"] = 0.0
    return h, meta


def main() -> None:
    files = sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No Mesonet CSVs in {RAW_DIR}.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    old_meta = pd.read_csv(STATIONS_OUT) if STATIONS_OUT.exists() else pd.DataFrame()
    metas = []
    for i, f in enumerate(files, 1):
        out = OUT_DIR / f"{f.stem}.parquet"
        if out.exists() and out.stat().st_mtime > f.stat().st_mtime and not old_meta.empty \
                and f.stem in set(old_meta["station"]):
            metas.append(old_meta[old_meta["station"] == f.stem].iloc[0].to_dict())
            print(f"[{i}/{len(files)}] {f.stem}: skip (already cleaned)")
            continue
        h, meta = clean_station(f)
        h.to_parquet(out, index=False)
        metas.append(meta)
        print(f"[{i}/{len(files)}] {f.stem}: {len(h):,} hours, "
              f"{meta['pct_hours_complete']}% complete, removed {meta['removed_temp']} temp / "
              f"{meta['removed_dewpoint']} dewpoint readings")

    stations = pd.DataFrame(metas).sort_values("station")
    stations.to_csv(STATIONS_OUT, index=False)
    print(f"\nSaved {len(stations)} stations to {OUT_DIR}/ and {STATIONS_OUT}")
    low = stations[stations["pct_hours_complete"] < 90]
    if len(low):
        print(f"Stations under 90% complete: {', '.join(low['station'])}")


if __name__ == "__main__":
    main()
