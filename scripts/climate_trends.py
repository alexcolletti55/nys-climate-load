"""Research Question 1: how have the weather drivers of New York electricity load changed?

Computes annual climate indicators at each NOAA station and tests them for trends.

Inputs  (data/cleaned/, from clean_weather.py):
    weather_daily.parquet    daily Tmax/Tmin, 1950-2025 (Islip from late 1963)
    weather_hourly.parquet   hourly temperature/dewpoint, 1973-2025
Outputs (small, derived from public NOAA data, so they are tracked by git):
    results/climate_annual_metrics.csv   one row per station-year
    results/climate_trends.csv           Mann-Kendall test + Sen's slope per station/metric/period
    figures/climate_trends.png           annual indicators for the three study-zone stations

Run from the repository root:
    python scripts/climate_trends.py

Indicators (per calendar year):
  From daily Tmax/Tmin (Tmean = (Tmax + Tmin) / 2):
    cdd_c          cooling degree days, base 18.3 C (65 F), in C-days (x1.8 for F-days)
    hdd_c          heating degree days, base 18.3 C
    hot_days       days with Tmax >= 90 F (32.2 C)
    warm_nights    days with Tmin >= 68 F (20 C); the ETCCDI "tropical nights" index (TR20)
    tmean_c        annual mean of daily Tmean
  From hourly dewpoint:
    humid_hours    hours with dewpoint >= 70 F (21.1 C), a common "oppressive humidity" threshold
    jja_dewpoint_c mean June-August dewpoint
Completeness rules: a year needs >= 345 days with both Tmax and Tmin (counts and sums are
scaled to a full year), or >= 90% of hours (>= 90% of June-August hours for jja_dewpoint_c).

Trend test: Mann-Kendall (pymannkendall.original_test), with Sen's slope reported per decade.
Periods:
  1964-2025  common period for daily indicators (Islip's daily record begins late 1963)
  1973-2025  full period for hourly (dewpoint) indicators
  1997-2025  recent period for all indicators. For dewpoint this also avoids the mid-1990s
             switch to automated ASOS sensors, which can create artificial steps in humidity
             records (Brown and DeGaetano 2013).
A period needs >= 20 valid years to be tested. Limitation: the original Mann-Kendall test
assumes independent years; mild year-to-year autocorrelation can overstate significance.
"""
import calendar
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymannkendall as mk

CLEANED = Path("data/cleaned")
RESULTS = Path("results")
FIGURES = Path("figures")

STATIONS = {  # ID -> (name, NYISO zone it represents or "-")
    "USW00014732": ("LaGuardia", "J"),
    "USW00004781": ("Islip", "K"),
    "USW00014771": ("Syracuse", "C"),
    "USW00094728": ("Central Park", "J (long record)"),
    "USW00014733": ("Buffalo", "A"),
    "USW00014735": ("Albany", "F"),
}
BASE_C = 18.3
MIN_DAYS = 345
MIN_HOUR_FRAC = 0.90
MIN_YEARS_FOR_TEST = 20

DAILY_METRICS = ["cdd_c", "hdd_c", "hot_days", "warm_nights", "tmean_c"]
HOURLY_METRICS = ["humid_hours", "jja_dewpoint_c"]
PERIODS = {
    "daily": [(1964, 2025), (1997, 2025)],
    "hourly": [(1973, 2025), (1997, 2025)],
}
LABELS = {
    "cdd_c": "Cooling degree days (°C-days)",
    "hdd_c": "Heating degree days (°C-days)",
    "hot_days": "Days with Tmax ≥ 90 °F",
    "warm_nights": "Nights with Tmin ≥ 68 °F",
    "humid_hours": "Hours with dewpoint ≥ 70 °F",
    "jja_dewpoint_c": "Summer (JJA) mean dewpoint (°C)",
}
# Figure: the three study-zone stations. Colors are the first three slots of a
# colorblind-validated categorical palette (blue, orange, aqua).
FIG_STATIONS = [("USW00014732", "#2a78d6"), ("USW00004781", "#eb6834"), ("USW00014771", "#1baf7a")]


def daily_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.dropna(subset=["tmax_c", "tmin_c"]).copy()
    d["year"] = d["date"].dt.year
    d["tmean_c"] = (d["tmax_c"] + d["tmin_c"]) / 2
    tmax_f = (d["tmax_c"] * 9 / 5 + 32).round()
    tmin_f = (d["tmin_c"] * 9 / 5 + 32).round()
    d["cdd"] = (d["tmean_c"] - BASE_C).clip(lower=0)
    d["hdd"] = (BASE_C - d["tmean_c"]).clip(lower=0)
    d["hot"] = tmax_f >= 90
    d["warm_night"] = tmin_f >= 68
    g = d.groupby(["station", "year"])
    a = pd.DataFrame({
        "n_days": g.size(),
        "cdd_c": g["cdd"].sum(), "hdd_c": g["hdd"].sum(),
        "hot_days": g["hot"].sum(), "warm_nights": g["warm_night"].sum(),
        "tmean_c": g["tmean_c"].mean(),
    }).reset_index()
    days_in_year = a["year"].map(lambda y: 366 if calendar.isleap(int(y)) else 365).to_numpy()
    scale = days_in_year / a["n_days"]
    for c in ["cdd_c", "hdd_c", "hot_days", "warm_nights"]:
        a[c] = a[c] * scale
    a.loc[a["n_days"] < MIN_DAYS, DAILY_METRICS] = np.nan
    return a


def hourly_metrics(hourly: pd.DataFrame) -> pd.DataFrame:
    h = hourly.dropna(subset=["dewpoint_c"]).copy()
    h["year"] = h["time_utc"].dt.year
    h["humid"] = h["dewpoint_c"] >= 21.1
    h["jja"] = h["time_utc"].dt.month.isin([6, 7, 8])
    g = h.groupby(["station", "year"])
    a = pd.DataFrame({"n_hours": g.size(), "humid_hours": g["humid"].sum()}).reset_index()
    hours_in_year = a["year"].map(lambda y: 8784 if calendar.isleap(int(y)) else 8760).to_numpy()
    a["humid_hours"] = a["humid_hours"] * hours_in_year / a["n_hours"]
    a.loc[a["n_hours"] < MIN_HOUR_FRAC * hours_in_year, "humid_hours"] = np.nan

    j = h[h["jja"]].groupby(["station", "year"])["dewpoint_c"].agg(["mean", "size"]).reset_index()
    j.loc[j["size"] < MIN_HOUR_FRAC * 92 * 24, "mean"] = np.nan
    a = a.merge(j.rename(columns={"mean": "jja_dewpoint_c", "size": "n_jja_hours"}),
                on=["station", "year"], how="left")
    return a


def trend_table(annual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for station, (name, zone) in STATIONS.items():
        s = annual[annual["station"] == station].set_index("year")
        for kind, metrics in (("daily", DAILY_METRICS), ("hourly", HOURLY_METRICS)):
            for metric in metrics:
                for y0, y1 in PERIODS[kind]:
                    x = s.loc[y0:y1, metric].dropna() if metric in s else pd.Series(dtype=float)
                    row = {"station": station, "station_name": name, "zone": zone, "metric": metric,
                           "period": f"{y0}-{y1}", "n_years": len(x),
                           "period_mean": round(x.mean(), 2) if len(x) else np.nan}
                    if len(x) >= MIN_YEARS_FOR_TEST:
                        r = mk.original_test(x.values)
                        row.update({"sen_slope_per_decade": round(r.slope * 10, 3),
                                    "mk_tau": round(r.Tau, 3), "mk_p": round(r.p, 4),
                                    "trend": r.trend})
                    else:
                        row.update({"sen_slope_per_decade": np.nan, "mk_tau": np.nan,
                                    "mk_p": np.nan, "trend": "too few years"})
                    rows.append(row)
    return pd.DataFrame(rows)


def make_figure(annual: pd.DataFrame, trends: pd.DataFrame, out: Path) -> None:
    metrics = ["cdd_c", "hdd_c", "hot_days", "warm_nights", "humid_hours", "jja_dewpoint_c"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True)
    for ax, metric in zip(axes.flat, metrics):
        kind = "hourly" if metric in HOURLY_METRICS else "daily"
        y0, y1 = PERIODS[kind][0]
        for station, color in FIG_STATIONS:
            s = annual[(annual["station"] == station)].set_index("year")[metric].loc[y0:y1].dropna()
            ax.plot(s.index, s.values, color=color, lw=1, alpha=0.55)
            t = trends[(trends["station"] == station) & (trends["metric"] == metric)
                       & (trends["period"] == f"{y0}-{y1}")].iloc[0]
            if np.isfinite(t["sen_slope_per_decade"]):
                # Sen line through the median point; solid = significant (p < 0.05), dotted = not
                xm, ym = np.median(s.index), np.median(s.values)
                xs = np.array([s.index.min(), s.index.max()])
                ys = ym + t["sen_slope_per_decade"] / 10 * (xs - xm)
                ax.plot(xs, ys, color=color, lw=2.5, ls="-" if t["mk_p"] < 0.05 else ":")
        ax.set_title(LABELS[metric], fontsize=10, loc="left", color="#0b0b0b")
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9a9994")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.tick_params(labelbottom=True)   # show years on every panel
        ax.set_xticks(range(1970, 2030, 10))
    # Two legends: station colors on the first row, trend-line meaning on the second.
    station_handles = [plt.Line2D([], [], color=c, lw=2.5, label=f"{STATIONS[s][0]} (zone {STATIONS[s][1]})")
                       for s, c in FIG_STATIONS]
    style_handles = [plt.Line2D([], [], color="#0b0b0b", lw=2.5, ls="-",
                                label="Solid trend line = significant trend (Mann-Kendall p < 0.05)"),
                     plt.Line2D([], [], color="#0b0b0b", lw=2.5, ls=":",
                                label="Dotted trend line = NOT significant (p \u2265 0.05)")]
    leg1 = fig.legend(handles=station_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
                      ncol=3, frameon=False, fontsize=9.5)
    fig.add_artist(leg1)
    fig.legend(handles=style_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2,
               frameon=True, edgecolor="#e4e3df", fontsize=9.5, handlelength=3,
               title="Trend lines (Sen's slope)", title_fontsize=9.5)
    fig.text(0.5, 0.01, "Thin lines: annual values. Thick lines: Sen's slope trend. Daily indicators 1964-2025; "
             "dewpoint indicators 1973-2025. Source: NOAA GHCN-Daily and GHCNh.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    daily = pd.read_parquet(CLEANED / "weather_daily.parquet")
    hourly = pd.read_parquet(CLEANED / "weather_hourly.parquet", columns=["time_utc", "station", "dewpoint_c"])

    annual = daily_metrics(daily).merge(hourly_metrics(hourly), on=["station", "year"], how="outer")
    annual.insert(1, "station_name", annual["station"].map(lambda s: STATIONS.get(s, ("?", ""))[0]))
    annual = annual.sort_values(["station", "year"])
    trends = trend_table(annual)

    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    annual.round(2).to_csv(RESULTS / "climate_annual_metrics.csv", index=False)
    trends.to_csv(RESULTS / "climate_trends.csv", index=False)
    make_figure(annual, trends, FIGURES / "climate_trends.png")

    show = trends[trends["station_name"].isin(["LaGuardia", "Islip", "Syracuse"])]
    show = show[show["period"].isin(["1964-2025", "1973-2025"])]
    print(show.pivot_table(index="metric", columns="station_name", values="sen_slope_per_decade").round(2)
          .to_string())
    print("\n(values = Sen's slope per decade; see results/climate_trends.csv for p-values and other periods)")
    print(f"Saved {RESULTS / 'climate_annual_metrics.csv'}, {RESULTS / 'climate_trends.csv'}, "
          f"{FIGURES / 'climate_trends.png'}")


if __name__ == "__main__":
    main()
