"""Clean NOAA weather downloads into analysis-ready tables (all temperatures in deg C, times in UTC).

Inputs  (from download_noaa.py):
    data/raw/ghcnh/{STATION}_{YEAR}.psv     hourly GHCNh files
    data/raw/ghcnd/{STATION}_daily.csv      daily Tmax/Tmin (deg F from the NOAA API)
Outputs (data/cleaned/):
    data/cleaned/weather_hourly.parquet   one row per station per hour:
        time_utc, station, temp_c, dewpoint_c, rh_pct, wetbulb_c, minutes_off_hour
    data/cleaned/weather_daily.parquet    one row per station per day: date, station, tmax_c, tmin_c
    data/cleaned/weather_hourly_completeness.csv   % of hours with valid temperature per station-year

Run from the repository root:
    python scripts/clean_weather.py
"""
import calendar
from pathlib import Path

import pandas as pd

HOURLY_DIR = Path("data/raw/ghcnh")
DAILY_DIR = Path("data/raw/ghcnd")
OUT_DIR = Path("data/cleaned")

VARS = {  # GHCNh column -> output column
    "temperature": "temp_c",
    "dew_point_temperature": "dewpoint_c",
    "relative_humidity": "rh_pct",
    "wet_bulb_temperature": "wetbulb_c",
}

# GHCNh quality codes (inherited from ISD). Keep values that passed QC or were
# accepted/corrected; drop suspect (2, 6) and erroneous (3, 7) values. Any code not
# listed here is also dropped and reported, so nothing unexpected slips through.
GOOD_QC = {"0", "1", "4", "5", "9", "A", "C", "I", "M", "P", "R", "U"}
DERIVED = {"relative_humidity", "wet_bulb_temperature"}

# Physical plausibility limits, deg C. A few impossible values passed NOAA's QC
# (e.g. a 50 C dewpoint at LaGuardia on 1997-12-30, -75.6 C at Syracuse on 2009-01-02).
TEMP_LIMITS = (-45.0, 45.0)
DEWPOINT_LIMITS = (-60.0, 35.0)
DEWPOINT_MAX_ABOVE_TEMP = 1.0   # dewpoint cannot exceed air temperature (1 C allows rounding)

# Routine hourly reports are preferred over specials when two reports fall in the same hour.
ROUTINE = {"FM15", "SAO", "SYSA", "FM12", "SYMT", "AUTO", "SY-MT", "SY-SA"}


def _qc(code: pd.Series) -> pd.Series:
    """Normalize QC codes: 5.0 -> '5', NaN -> ''."""
    s = code.astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s.where(code.notna(), "")


def clean_hourly_file(path: Path, rejected: dict) -> pd.DataFrame:
    usecols = ["STATION", "DATE", "temperature_Report_Type"]
    for v in VARS:
        usecols += [v, f"{v}_Quality_Code"]
    df = pd.read_csv(path, sep="|", usecols=usecols, low_memory=False)

    # Blank out values that failed QC, variable by variable. RH and wet bulb are derived
    # by NCEI from temperature/dewpoint and usually carry no QC code, so blank is OK for them.
    for v in VARS:
        qc = _qc(df[f"{v}_Quality_Code"])
        ok = GOOD_QC | ({""} if v in DERIVED else set())
        bad = df[v].notna() & ~qc.isin(ok)
        for code, n in qc[bad].value_counts().items():
            rejected[(v, code)] = rejected.get((v, code), 0) + int(n)
        df.loc[bad, v] = pd.NA

    # Physical plausibility checks on values that passed NOAA's QC codes.
    t = pd.to_numeric(df["temperature"], errors="coerce")
    td = pd.to_numeric(df["dew_point_temperature"], errors="coerce")
    bad_t = t.notna() & ((t < TEMP_LIMITS[0]) | (t > TEMP_LIMITS[1]))
    bad_td = td.notna() & ((td < DEWPOINT_LIMITS[0]) | (td > DEWPOINT_LIMITS[1])
                           | (td > t + DEWPOINT_MAX_ABOVE_TEMP))
    for v, mask in (("temperature", bad_t), ("dew_point_temperature", bad_td | bad_t)):
        n = int((mask & df[v].notna()).sum())
        if n:
            rejected[(v, "physical limit")] = rejected.get((v, "physical limit"), 0) + n
        df.loc[mask, v] = pd.NA

    # Derived humidity values are only meaningful if both inputs passed QC.
    inputs_bad = df["temperature"].isna() | df["dew_point_temperature"].isna()
    df.loc[inputs_bad, list(DERIVED)] = pd.NA

    df = df[df["temperature"].notna()].copy()
    if df.empty:
        return df

    # Assign each report to the nearest top of the hour (e.g. 00:51 -> 01:00).
    t = pd.to_datetime(df["DATE"], utc=True)
    df["time_utc"] = t.dt.round("h")
    df["minutes_off_hour"] = ((t - df["time_utc"]).dt.total_seconds() / 60).round().astype(int)
    df["_special"] = ~df["temperature_Report_Type"].astype(str).str.strip().isin(ROUTINE)

    out = df.rename(columns={"STATION": "station", **VARS})
    out = out[["time_utc", "station", *VARS.values(), "minutes_off_hour", "_special"]]
    for c in VARS.values():
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def one_per_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one report per station-hour: closest to the top of the hour, routine before special.
    Done after combining years so a 23:51 report on Dec 31 doesn't collide with the next file."""
    df = df.assign(_dist=df["minutes_off_hour"].abs())
    df = (df.sort_values(["station", "time_utc", "_dist", "_special"])
            .drop_duplicates(["station", "time_utc"], keep="first"))
    return df.drop(columns=["_dist", "_special"])


def build_hourly() -> None:
    files = sorted(HOURLY_DIR.glob("*.psv"))
    if not files:
        print(f"No hourly files in {HOURLY_DIR}; skipping hourly.")
        return
    rejected: dict = {}
    parts = []
    for i, f in enumerate(files, 1):
        parts.append(clean_hourly_file(f, rejected))
        if i % 25 == 0 or i == len(files):
            print(f"  hourly: {i}/{len(files)} files")
    hourly = one_per_hour(pd.concat(parts, ignore_index=True))
    # A 23:51 report on Dec 31 of the last year rounds into the next year; drop that spillover.
    last_year = max(int(f.stem.split("_")[-1]) for f in files)
    hourly = hourly[hourly["time_utc"] < pd.Timestamp(f"{last_year + 1}-01-01", tz="UTC")]

    if rejected:
        print("Values dropped by QC code (variable, code): count")
        for k, n in sorted(rejected.items()):
            print(f"  {k}: {n}")

    # Completeness: share of hours in each station-year with a valid temperature.
    hourly["year"] = hourly["time_utc"].dt.year
    counts = hourly.groupby(["station", "year"]).size().rename("hours_with_temp").reset_index()
    counts["hours_in_year"] = counts["year"].map(lambda y: 8784 if calendar.isleap(y) else 8760)
    counts["pct_complete"] = (100 * counts["hours_with_temp"] / counts["hours_in_year"]).round(1)
    hourly = hourly.drop(columns="year")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_parquet(OUT_DIR / "weather_hourly.parquet", index=False)
    counts.to_csv(OUT_DIR / "weather_hourly_completeness.csv", index=False)
    print(f"Saved {OUT_DIR / 'weather_hourly.parquet'}: {len(hourly):,} station-hours")
    low = counts[counts["pct_complete"] < 90]
    print(f"Station-years under 90% complete: {len(low)} (see weather_hourly_completeness.csv)")


def build_daily() -> None:
    files = sorted(DAILY_DIR.glob("*_daily.csv"))
    if not files:
        print(f"No daily files in {DAILY_DIR}; skipping daily.")
        return
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df = df.rename(columns={"STATION": "station", "DATE": "date"})
    df["date"] = pd.to_datetime(df["date"])
    # NOAA API returned deg F (units=standard); convert to deg C for consistency.
    df["tmax_c"] = ((pd.to_numeric(df["TMAX"], errors="coerce") - 32) * 5 / 9).round(2)
    df["tmin_c"] = ((pd.to_numeric(df["TMIN"], errors="coerce") - 32) * 5 / 9).round(2)
    bad = df["tmax_c"] < df["tmin_c"]
    if bad.any():
        print(f"Dropping {bad.sum()} days with Tmax < Tmin")
        df.loc[bad, ["tmax_c", "tmin_c"]] = pd.NA
    daily = df[["date", "station", "tmax_c", "tmin_c"]].sort_values(["station", "date"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT_DIR / "weather_daily.parquet", index=False)
    span = daily.groupby("station")["date"].agg(["min", "max", "count"])
    print(f"Saved {OUT_DIR / 'weather_daily.parquet'}\n{span}")


if __name__ == "__main__":
    build_daily()
    build_hourly()
