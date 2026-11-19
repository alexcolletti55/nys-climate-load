"""Data-quality check: are the airport dewpoint records trustworthy enough for the load model?

Motivation: in figures/climate_trends.png, hours with dewpoint >= 70 F drop sharply at both
LaGuardia and Islip around 2006-2012 and then recover. A dip shared by two stations that
recovers could be real weather or an instrument problem (ASOS dewpoint sensors are a known
source of artificial steps; Brown and DeGaetano 2013). This script does not change any data;
it measures how consistent the stations are so the load model can use only trustworthy years.

Inputs (data/cleaned/): weather_hourly.parquet (NOAA), mesonet_hourly/, mesonet_stations.csv
Outputs:
    figures/dewpoint_check.png
    results/dewpoint_check_summary.csv   mean differences by period
    results/dewpoint_check_monthly.csv   monthly difference series behind the figure

Three checks:
  (a) Summer (June-August) mean dewpoint anomalies at each NOAA station, 1973-2025, relative
      to each station's 1997-2025 mean. Real weather moves the stations together.
  (b) Monthly LaGuardia - Central Park and Islip - Central Park dewpoint differences, with each
      calendar month's average difference removed (deseasonalized), 1996-2025 (Central Park's
      hourly record is complete from 1996). An instrument problem shows up as a step or a
      temporary dip in the difference that the reference station does not share.
  (c) Monthly airport minus nearby NYS Mesonet dewpoint, 2017-2025: LaGuardia vs mean of
      Queens (QUEE) and Manhattan (MANH); Islip vs Stony Brook (STON); Central Park vs MANH.
      Monthly means avoid any mismatch between airport hourly reports and Mesonet hour averages.
A month counts only if >= 90% of its hours have dewpoint at both stations being compared.

Run from the repository root:
    python scripts/check_dewpoint.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CLEANED = Path("data/cleaned")
RESULTS = Path("results")
FIGURES = Path("figures")

NOAA = {"USW00014732": "LaGuardia", "USW00004781": "Islip", "USW00094728": "Central Park",
        "USW00014735": "Albany", "USW00014771": "Syracuse", "USW00014733": "Buffalo"}
UPSTATE = ["Albany", "Syracuse", "Buffalo"]
MESONET_PAIRS = {  # airport -> nearby Mesonet stations (distance in km)
    "LaGuardia": ["QUEE", "MANH"],     # 7.4, 7.2 km
    "Islip": ["STON"],                 # 14.2 km
    "Central Park": ["MANH"],          # 1.3 km
}
PERIODS = [(1996, 2005), (2006, 2012), (2013, 2025)]
MIN_FRAC = 0.90
# First three slots of a colorblind-validated categorical palette, plus neutral gray
COLORS = {"LaGuardia": "#2a78d6", "Islip": "#eb6834", "Central Park": "#1baf7a",
          "Upstate mean": "#52514e"}


def monthly_means(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Monthly mean dewpoint per station, with the count of valid hours."""
    d = df.dropna(subset=["dewpoint_c"]).copy()
    d["month"] = d["time_utc"].dt.tz_convert(None).dt.to_period("M")
    m = d.groupby([key, "month"])["dewpoint_c"].agg(["mean", "size"]).reset_index()
    m["hours_in_month"] = m["month"].dt.days_in_month * 24
    m.loc[m["size"] < MIN_FRAC * m["hours_in_month"], "mean"] = np.nan
    return m.pivot(index="month", columns=key, values="mean")


def main() -> None:
    noaa = pd.read_parquet(CLEANED / "weather_hourly.parquet", columns=["time_utc", "station", "dewpoint_c"])
    noaa["name"] = noaa["station"].map(NOAA)
    noaa_m = monthly_means(noaa, "name")

    meso_ids = sorted({s for v in MESONET_PAIRS.values() for s in v})
    meso = pd.concat([pd.read_parquet(CLEANED / "mesonet_hourly" / f"{s}.parquet",
                                      columns=["time_utc", "station", "dewpoint_c"]) for s in meso_ids])
    meso_m = monthly_means(meso, "station")

    # (a) summer anomalies
    summer = noaa_m[noaa_m.index.month.isin([6, 7, 8])]
    jja = summer.groupby(summer.index.year).agg(lambda x: x.mean() if x.notna().sum() == 3 else np.nan)
    anom = jja - jja.loc[1997:2025].mean()
    anom["Upstate mean"] = anom[UPSTATE].mean(axis=1, skipna=False)

    # (b) deseasonalized monthly differences vs Central Park
    diffs = pd.DataFrame(index=noaa_m.index)
    for st in ["LaGuardia", "Islip"]:
        d = noaa_m[st] - noaa_m["Central Park"]
        d = d[d.index.year >= 1996]
        d = d - d.groupby(d.index.month).transform("mean")
        diffs[f"{st} - Central Park"] = d

    # (c) airport minus nearby Mesonet
    meso_diffs = pd.DataFrame(index=noaa_m.index)
    for airport, mids in MESONET_PAIRS.items():
        ref = meso_m[mids].mean(axis=1, skipna=False)
        meso_diffs[f"{airport} - {'/'.join(mids)}"] = noaa_m[airport] - ref
    meso_diffs = meso_diffs[meso_diffs.index.year >= 2017]

    # Summary table
    rows = []
    for col in diffs:
        for y0, y1 in PERIODS:
            x = diffs[col][(diffs.index.year >= y0) & (diffs.index.year <= y1)].dropna()
            rows.append({"comparison": col + " (deseasonalized)", "period": f"{y0}-{y1}",
                         "n_months": len(x), "mean_diff_c": round(x.mean(), 2), "std_c": round(x.std(), 2)})
    for col in meso_diffs:
        x = meso_diffs[col].dropna()
        rows.append({"comparison": col, "period": "2017-2025", "n_months": len(x),
                     "mean_diff_c": round(x.mean(), 2), "std_c": round(x.std(), 2)})
        for y in sorted(set(x.index.year)):
            xy = x[x.index.year == y]
            rows.append({"comparison": col, "period": str(y), "n_months": len(xy),
                         "mean_diff_c": round(xy.mean(), 2), "std_c": round(xy.std(), 2)})
    summary = pd.DataFrame(rows)

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    summary.to_csv(RESULTS / "dewpoint_check_summary.csv", index=False)
    pd.concat([diffs, meso_diffs], axis=1).dropna(how="all").round(3).to_csv(
        RESULTS / "dewpoint_check_monthly.csv", index_label="month")

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    ax = axes[0]
    for col in ["LaGuardia", "Islip", "Central Park", "Upstate mean"]:
        s = anom[col].dropna()
        ax.plot(s.index, s.values, color=COLORS[col], lw=2 if col != "Upstate mean" else 1.5,
                ls="-" if col != "Upstate mean" else "--", label=col)
    ax.set_title("(a) Summer (JJA) mean dewpoint anomaly vs. 1997-2025 mean (°C)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left")

    ax = axes[1]
    for col, c in zip(diffs.columns, ["LaGuardia", "Islip"]):
        s = diffs[col].dropna()
        ax.plot(s.index.to_timestamp(), s.values, color=COLORS[c], lw=0.8, alpha=0.35)
        ax.plot(s.index.to_timestamp(), s.rolling(12, min_periods=9).mean(), color=COLORS[c], lw=2, label=col)
    ax.axhline(0, color="#9a9994", lw=1)
    ax.axvspan(pd.Timestamp("2006-01-01"), pd.Timestamp("2012-12-31"), color="#e4e3df", alpha=0.6, lw=0)
    ax.set_title("(b) Monthly dewpoint difference vs. Central Park, seasonal cycle removed (°C); "
                 "thick = 12-month mean, shaded = 2006-2012", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    ax = axes[2]
    for col, c in zip(meso_diffs.columns, MESONET_PAIRS):
        s = meso_diffs[col].dropna()
        ax.plot(s.index.to_timestamp(), s.values, color=COLORS[c], lw=1.5, marker="o", ms=3, label=col)
    ax.axhline(0, color="#9a9994", lw=1)
    ax.set_title("(c) Monthly mean dewpoint, airport minus nearby NYS Mesonet station(s) (°C)", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")

    for ax in axes:
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9a9994")
        ax.tick_params(colors="#52514e", labelsize=8)
    fig.text(0.5, 0.005, "Sources: NOAA GHCNh; NYS Mesonet (Brotzge et al. 2020). Months need >= 90% valid hours.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIGURES / "dewpoint_check.png", dpi=200, facecolor="white")
    plt.close(fig)

    print(summary[summary["period"].str.contains("-")].to_string(index=False))
    print(f"\nSaved {FIGURES / 'dewpoint_check.png'}, {RESULTS / 'dewpoint_check_summary.csv'}, "
          f"{RESULTS / 'dewpoint_check_monthly.csv'}")


if __name__ == "__main__":
    main()
