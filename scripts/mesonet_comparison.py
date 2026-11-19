"""Robustness check: is one airport station a good enough stand-in for a zone's weather?

Fits the same load models twice on the SAME days (2018-2025), once with the single airport
station (as in load_model.py) and once with the average of the NYS Mesonet stations in the zone.

Inputs:  data/processed/zone_daily.parquet, data/cleaned/mesonet_hourly/, data/cleaned/weather_hourly.parquet
Outputs:
    results/mesonet_comparison.csv       sensitivities (full 2018-2025 fit) and out-of-sample
                                         accuracy (train 2018-2022, test 2023-2025), airport vs. Mesonet
    results/mesonet_zone_stations.csv    which Mesonet stations represent each zone
    figures/mesonet_comparison.png
    figures/heatwave_snapshot.png        station temperatures across NYC and Long Island at the
                                         hour of NYC's three highest peaks, 2018-2025

Run from the repository root (reuses functions from scripts/load_model.py):
    python scripts/mesonet_comparison.py

Method:
  * Zone station sets: J = the five borough stations; K = all three Long Island stations
    (Nassau: WANT; Suffolk: STON, SOUT); C = six stations within ~55 km of Syracuse and west of
    Oneida County (an APPROXIMATION of NYISO's Central zone, whose boundaries are not county lines).
  * Mesonet zone hourly value = mean of the zone's stations, kept if >= 60% of them report.
    Mesonet hours are averages over the hour labeled by its start, which already matches NYISO's
    hour-beginning load. Daily values need >= 18 hours (same rule as build_zone_daily.py).
  * Both versions use identical days: 2018-2025, outages and COVID lockdown excluded, both
    weather sources complete. Models M1 and M5 and the balance-point search are the same as in
    load_model.py (M3 is not used here). Absolute accuracy for zone J is still affected by the
    post-COVID level shift, but both versions share it, so the comparison is fair.
  * Heat-wave snapshot: temperature at each NYC/Long Island Mesonet station and airport at the
    local hour of zone J's daily peak, on zone J's three highest-load days. Airport values average
    the observations at the start and end of the hour, as in build_zone_daily.py.
"""
from math import ceil
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_model as L  # noqa: E402

CLEANED = Path("data/cleaned")
MESO_DIR = CLEANED / "mesonet_hourly"
RESULTS = Path("results")
FIGURES = Path("figures")
LOCAL_TZ = "America/New_York"

ZONE_STATIONS = {
    "J": ["BKLN", "BRON", "MANH", "QUEE", "STAT"],
    "K": ["WANT", "STON", "SOUT"],
    "C": ["FAYE", "CSQR", "JORD", "TULL", "OSWE", "SCIP"],   # approximate Central zone
}
AIRPORTS = {"USW00014732": "LaGuardia", "USW00094728": "Central Park", "USW00004781": "Islip"}
MIN_STATION_FRAC = 0.6
YEARS, TRAIN, TEST = (2018, 2025), (2018, 2022), (2023, 2025)
MODELS = ["M1", "M5"]
VERSION_STYLE = {"Airport": ("#52514e", "o"), "Mesonet": ("#6250d6", "s")}


def mesonet_zone_daily(zone: str) -> pd.DataFrame:
    stations = ZONE_STATIONS[zone]
    h = pd.concat([pd.read_parquet(MESO_DIR / f"{s}.parquet", columns=["time_utc", "temp_c", "dewpoint_c"])
                   for s in stations])
    g = h.groupby("time_utc")
    z = pd.DataFrame({"temp_c": g["temp_c"].mean(), "dewpoint_c": g["dewpoint_c"].mean(),
                      "n_t": g["temp_c"].count(), "n_d": g["dewpoint_c"].count()})
    need = ceil(MIN_STATION_FRAC * len(stations))
    z.loc[z["n_t"] < need, "temp_c"] = np.nan
    z.loc[z["n_d"] < need, "dewpoint_c"] = np.nan
    local_day = z.index.tz_convert(LOCAL_TZ).tz_localize(None).normalize()
    g = z.groupby(local_day)
    d = pd.DataFrame({"tmean_c": g["temp_c"].mean(), "dewpoint_mean_c": g["dewpoint_c"].mean(),
                      "n_t": g["temp_c"].count(), "n_d": g["dewpoint_c"].count()})
    d.loc[d["n_t"] < 18, "tmean_c"] = np.nan
    d.loc[d["n_d"] < 18, "dewpoint_mean_c"] = np.nan
    d = d.asfreq("D")
    d["tmean_lag1_c"] = d["tmean_c"].shift(1)
    d.index.name = "date"
    return d[["tmean_c", "tmean_lag1_c", "dewpoint_mean_c"]].reset_index()


def zone_versions(zd: pd.DataFrame, zone: str) -> dict:
    base = zd[(zd["zone"] == zone) & zd["year"].between(*YEARS) & ~zd["exclude"] & ~zd["covid_lockdown"]].copy()
    base["trend"] = base["year"] - 2015 + (base["date"].dt.day_of_year - 1) / 365.25
    wx_cols = ["tmean_c", "tmean_lag1_c", "dewpoint_mean_c"]
    meso = mesonet_zone_daily(zone).rename(columns={c: f"{c}_m" for c in wx_cols})
    base = base.merge(meso, on="date", how="left")
    ok = base[["peak_mw", *wx_cols, *[f"{c}_m" for c in wx_cols]]].notna().all(axis=1)
    base = base[ok].reset_index(drop=True)
    airport = base.copy()
    mesonet = base.copy()
    for c in wx_cols:
        mesonet[c] = base[f"{c}_m"]
    out = {}
    for name, df in (("Airport", airport), ("Mesonet", mesonet)):
        df = df.copy()
        df["hi_mean_c"] = df["tmean_c"]          # placeholder: M3 is not used in this comparison
        df["hi_mean_lag1_c"] = df["tmean_lag1_c"]
        out[name] = df
    return out


def evaluate(versions: dict, zone: str) -> list[dict]:
    rows = []
    for version, df in versions.items():
        train = df[df["year"].between(*TRAIN)].reset_index(drop=True)
        test = df[df["year"].between(*TEST)].reset_index(drop=True)
        top = test["peak_mw"] >= test["peak_mw"].quantile(0.95)
        bp_train = L.select_balance_points(train)
        bp_full = L.select_balance_points(df)
        jja = df["month"].isin([6, 7, 8])
        t_mean, summer_peak = float(df.loc[jja, "tmean_c"].mean()), float(df.loc[jja, "peak_mw"].mean())
        for model in MODELS:
            # out-of-sample
            tc, th, td0 = bp_train[model]
            Xtr = L.design(train, model, tc, th, td0)
            res_tr = sm.OLS(train["peak_mw"].to_numpy(), Xtr).fit()
            err = test["peak_mw"].to_numpy() - res_tr.predict(L.design(test, model, tc, th, td0)[Xtr.columns]).to_numpy()
            # full-sample sensitivity
            res, names = L.fit(df, model, bp_full[model])
            est, lo, hi, pct, pct_ci = L.sustained(res, names, summer_peak, t_mean, bp_full[model][0])
            rows.append({
                "zone": zone, "version": version, "model": model, "n_days": len(df),
                "stations": ", ".join(ZONE_STATIONS[zone]) if version == "Mesonet" else L.ZONES[zone],
                "cool_balance_c": bp_full[model][0], "heat_balance_c": bp_full[model][1],
                "dewpoint_threshold_c": bp_full[model][2],
                "cooling_mw_per_c": round(est, 1), "ci95_low": round(lo, 1), "ci95_high": round(hi, 1),
                "cooling_pct_of_summer_peak_per_c": round(pct, 2), "pct_ci95_halfwidth": round(pct_ci, 2),
                "r2_full": round(res.rsquared, 3),
                "test_rmse_mw": round(np.sqrt(np.mean(err ** 2)), 1),
                "test_rmse_pct": round(100 * np.sqrt(np.mean(err ** 2)) / test["peak_mw"].mean(), 2),
                "test_top5_rmse_mw": round(np.sqrt(np.mean(err[top] ** 2)), 1),
                "test_top5_rmse_pct": round(100 * np.sqrt(np.mean(err[top] ** 2)) / test.loc[top, "peak_mw"].mean(), 2),
                "test_top5_bias_mw": round(np.mean(err[top]), 1),
            })
    return rows


def fig_comparison(res: pd.DataFrame, out: Path) -> None:
    r = res[res["model"] == "M5"]
    panels = [("cooling_pct_of_summer_peak_per_c", "Cooling sensitivity, M5 (% of summer peak per °C)", True),
              ("test_rmse_pct", "Test error, all days (RMSE, % of mean peak)", False),
              ("test_top5_rmse_pct", "Test error, top 5% days (RMSE, % of their mean)", False)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
    for ax, (col, title, ci) in zip(axes, panels):
        for i, zone in enumerate(L.ZONES):
            for k, version in enumerate(["Airport", "Mesonet"]):
                row = r[(r["zone"] == zone) & (r["version"] == version)].iloc[0]
                color, marker = VERSION_STYLE[version]
                y = i + (k - 0.5) * 0.3
                if ci:
                    ax.errorbar(row[col], y, xerr=row["pct_ci95_halfwidth"], fmt="none", ecolor=color, lw=1.5)
                ax.scatter(row[col], y, color=color, marker=marker, s=45, zorder=3,
                           label=version if i == 0 else None)
        ax.set_yticks(range(len(L.ZONES)), [f"Zone {z}" for z in L.ZONES])
        ax.set_title(title, loc="left", fontsize=9.5)
        L.style(ax)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.5, 0.01, "Same days for both versions (2018-2025). Sensitivity: full-sample fit with 95% CI; "
             "test error: trained 2018-2022, tested 2023-2025. Lower error is better.",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def fig_snapshot(zd: pd.DataFrame, out: Path) -> pd.DataFrame:
    j = zd[(zd["zone"] == "J") & zd["year"].between(*YEARS) & ~zd["exclude"]].dropna(subset=["peak_mw"])
    days = j.nlargest(3, "peak_mw")
    stations = pd.read_csv(CLEANED / "mesonet_stations.csv").set_index("station")
    meso_ids = ZONE_STATIONS["J"] + ZONE_STATIONS["K"]
    meso = pd.concat([pd.read_parquet(MESO_DIR / f"{s}.parquet", columns=["time_utc", "station", "temp_c"])
                      for s in meso_ids])
    air = pd.read_parquet(CLEANED / "weather_hourly.parquet", columns=["time_utc", "station", "temp_c"])
    air = air[air["station"].isin(AIRPORTS)]
    air_lonlat = {"USW00014732": (-73.880, 40.779), "USW00094728": (-73.969, 40.779), "USW00004781": (-73.102, 40.794)}

    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, (_, day) in zip(axes, days.sort_values("date").iterrows()):
        hour_local = pd.Timestamp(day["date"]) + pd.Timedelta(hours=int(day["peak_hour_local"]))
        hour_utc = hour_local.tz_localize(LOCAL_TZ).tz_convert("UTC")
        for s in meso_ids:
            v = meso[(meso["station"] == s) & (meso["time_utc"] == hour_utc)]["temp_c"]
            if len(v) and np.isfinite(v.iloc[0]):
                lon, lat = stations.loc[s, "longitude"], stations.loc[s, "latitude"]
                zone = "J" if s in ZONE_STATIONS["J"] else "K"
                ax.scatter(lon, v.iloc[0], color=L.ZONE_COLORS[zone], s=40, zorder=3)
                ax.annotate(s, (lon, v.iloc[0]), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#52514e")
                rows.append({"date": day["date"].date(), "hour_local": int(day["peak_hour_local"]), "site": s,
                             "type": "Mesonet", "zone": zone, "longitude": lon, "temp_c": round(float(v.iloc[0]), 1)})
        for sid, name in AIRPORTS.items():
            a = air[(air["station"] == sid) & air["time_utc"].isin([hour_utc, hour_utc + pd.Timedelta("1h")])]["temp_c"]
            if len(a):
                lon = air_lonlat[sid][0]
                t = float(a.mean())
                ax.scatter(lon, t, color="#0b0b0b", marker="D", s=40, zorder=4)
                ax.annotate(name, (lon, t), textcoords="offset points", xytext=(-6, -3), ha="right",
                            fontsize=7, color="#0b0b0b", fontweight="bold")
                rows.append({"date": day["date"].date(), "hour_local": int(day["peak_hour_local"]), "site": name,
                             "type": "Airport", "zone": "J" if sid != "USW00004781" else "K",
                             "longitude": lon, "temp_c": round(t, 1)})
        ax.set_title(f"{day['date']:%b %d, %Y}, {int(day['peak_hour_local'])}:00  "
                     f"(NYC peak {day['peak_mw']:,.0f} MW)", loc="left", fontsize=9.5)
        ax.set_xlabel("Longitude (west → east)", fontsize=9, color="#52514e")
        L.style(ax)
    axes[0].set_ylabel("Temperature at NYC peak hour (°C)", fontsize=9, color="#52514e")
    handles = [plt.Line2D([], [], color=L.ZONE_COLORS["J"], marker="o", ls="", label="Mesonet, NYC (zone J)"),
               plt.Line2D([], [], color=L.ZONE_COLORS["K"], marker="o", ls="", label="Mesonet, Long Island (zone K)"),
               plt.Line2D([], [], color="#0b0b0b", marker="D", ls="", label="Airport / Central Park")]
    axes[0].legend(handles=handles, frameon=False, fontsize=7.5, loc="lower left")
    fig.text(0.5, 0.005, "NYC's three highest-load days, 2018-2025. Sources: NYS Mesonet (Brotzge et al. 2020), "
             "NOAA GHCNh, NYISO.", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return pd.DataFrame(rows)


def main() -> None:
    zd = pd.read_parquet(L.PROCESSED / "zone_daily.parquet")
    rows = []
    for zone in L.ZONES:
        rows += evaluate(zone_versions(zd, zone), zone)
        print(f"Zone {zone}: done")
    res = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    res.to_csv(RESULTS / "mesonet_comparison.csv", index=False)
    stations = pd.read_csv(CLEANED / "mesonet_stations.csv")
    zs = pd.concat([stations[stations["station"].isin(v)].assign(zone=z) for z, v in ZONE_STATIONS.items()])
    zs[["zone", "station", "latitude", "longitude", "elevation_m", "first_valid_utc", "pct_hours_complete"]].to_csv(
        RESULTS / "mesonet_zone_stations.csv", index=False)
    fig_comparison(res, FIGURES / "mesonet_comparison.png")
    snap = fig_snapshot(zd, FIGURES / "heatwave_snapshot.png")

    pd.set_option("display.width", 220)
    print(res[["zone", "version", "model", "n_days", "cooling_mw_per_c", "ci95_low", "ci95_high",
               "cooling_pct_of_summer_peak_per_c", "test_rmse_pct", "test_top5_rmse_pct", "test_top5_bias_mw"]]
          .to_string(index=False))
    spread = snap.groupby("date")["temp_c"].agg(["min", "max"])
    print("\nHeat-wave snapshot, temperature spread across sites (°C):")
    print((spread.assign(range_c=spread["max"] - spread["min"])).round(1).to_string())
    print(f"\nSaved results/mesonet_comparison.csv, results/mesonet_zone_stations.csv, "
          f"figures/mesonet_comparison.png, figures/heatwave_snapshot.png")


if __name__ == "__main__":
    main()
