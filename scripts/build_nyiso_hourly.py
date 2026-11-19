"""Combine NYISO daily CSVs into one hourly table: one row per hour, one column per zone (MW).

Run after download_nyiso.py:
    python scripts/build_nyiso_hourly.py

Output: data/cleaned/nyiso_hourly_load.parquet
    index  time_utc  start of the hour, UTC (NYISO "Time Stamp" is hour-beginning local time;
                     each daily file runs 00:00-23:00, and EST/EDT labels resolve DST)
    columns A-K      NYISO zones (A=WEST ... J=N.Y.C., K=LONGIL)
            JK       NYC + Long Island; uses the combined "N.Y.C._LONGIL" zone NYISO reported
                     until Jan 2005, so JK runs back to 2001 while J and K start Feb 2005
            NYCA     statewide total (sum of A-K, or A-I + JK before 2005)
Every hour from the first to the last timestamp is present; hours NYISO did not report
(e.g. the Aug 2003 blackout) and zero readings are NaN.

Bad-value screening (added 2026-10-27): values above 2x the zone's 99.9th percentile
(impossible spikes) or below 40% of the surrounding 2-day median (dropouts, DST reporting
artifacts) are set to NaN. Every removed value is logged to
data/cleaned/nyiso_removed_values.csv. Unusually HIGH but plausible values (e.g.
early-season heat waves) are deliberately kept.
NOTE: this screening does NOT catch multi-day outages (Superstorm Sandy, Hurricane Irene),
because those drag the rolling median down with them. Known major outages are flagged
instead in scripts/build_zone_daily.py.
"""
from pathlib import Path

import pandas as pd

CSV_DIR = Path("data/raw/nyiso/csv")
OUT = Path("data/cleaned/nyiso_hourly_load.parquet")
REMOVED_LOG = Path("data/cleaned/nyiso_removed_values.csv")

SPIKE_FACTOR = 2.0        # remove values above SPIKE_FACTOR x the zone's 99.9th percentile
DROPOUT_FRACTION = 0.4    # remove values below this fraction of the centered 49-hour median
MEDIAN_WINDOW_H = 49

ZONES = {
    "WEST": "A", "GENESE": "B", "CENTRL": "C", "NORTH": "D", "MHK VL": "E",
    "CAPITL": "F", "HUD VL": "G", "MILLWD": "H", "DUNWOD": "I",
    "N.Y.C.": "J", "LONGIL": "K", "N.Y.C._LONGIL": "JK_combined",
}


def main() -> None:
    files = sorted(CSV_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSVs in {CSV_DIR}. Run download_nyiso.py first.")
    print(f"Reading {len(files):,} daily files...")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    # Local hour-beginning time -> UTC using the EST/EDT label (unambiguous across DST).
    offset_h = df["Time Zone"].map({"EST": 5, "EDT": 4})
    if offset_h.isna().any():
        raise SystemExit(f"Unexpected Time Zone labels: {df.loc[offset_h.isna(), 'Time Zone'].unique()}")
    local = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M:%S")
    df["time_utc"] = (local + pd.to_timedelta(offset_h, unit="h")).dt.tz_localize("UTC")

    unknown = set(df["Name"]) - set(ZONES)
    if unknown:
        print("Warning: unknown zone names ignored:", unknown)
    df["zone"] = df["Name"].map(ZONES)

    wide = df.pivot_table(index="time_utc", columns="zone", values="Integrated Load", aggfunc="mean")
    wide.columns.name = None

    # Zero load is a reporting error, not real; make every hour explicit.
    wide = wide.mask(wide <= 0)
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h", tz="UTC", name="time_utc")
    n_absent = len(full.difference(wide.index))
    wide = wide.reindex(full)

    # Screen each zone for impossible spikes and dropouts before building totals.
    # Example caught: zone D (North) = 15,160 MW on 2001-10-24 vs a typical ~700 MW.
    removed = []
    for z in [c for c in wide.columns]:
        s = wide[z]
        cap = SPIKE_FACTOR * s.quantile(0.999)
        med = s.rolling(MEDIAN_WINDOW_H, center=True, min_periods=12).median()
        spike = s > cap
        drop = s < DROPOUT_FRACTION * med
        for kind, mask in (("spike", spike), ("dropout", drop)):
            if mask.any():
                removed.append(pd.DataFrame({"zone": z, "type": kind, "load_mw": s[mask],
                                             "median_49h_mw": med[mask].round(1)}))
        wide.loc[spike | drop, z] = float("nan")
    if removed:
        log = pd.concat(removed).rename_axis("time_utc").reset_index()
        REMOVED_LOG.parent.mkdir(parents=True, exist_ok=True)
        log.to_csv(REMOVED_LOG, index=False)
        print("Values removed by screening (zone, type):")
        print(log.groupby(["zone", "type"]).size().to_string())
        print(f"Full list saved to {REMOVED_LOG}")

    wide["JK"] = (wide["J"] + wide["K"]).fillna(wide["JK_combined"])
    single = [z for z in "ABCDEFGHI"]
    wide["NYCA"] = wide[single].sum(axis=1, min_count=len(single)) + wide["JK"]
    wide = wide[[*"ABCDEFGHIJK", "JK", "NYCA"]]

    print(f"{len(wide):,} hours, {wide.index.min()} to {wide.index.max()}")
    print(f"Hours NYISO did not report: {n_absent}; zero readings set to missing: "
          f"{int((df['Integrated Load'] <= 0).sum())} values")
    print("Missing hours per column:\n", wide.isna().sum().to_string())
    print("J and K start:", wide["J"].first_valid_index())

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
