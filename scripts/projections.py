"""Research Question 3: how much would peak load rise under +1, +2, +3 C of warming?

"Delta method": take each zone's real daily weather from past years, warm every day by
dT, and run it through the load models fitted in load_model.py, with today's grid held fixed.

Inputs:  data/processed/zone_daily.parquet (+ zone_hourly.parquet via load_model.load_data)
Outputs:
    results/projections_summary.csv   per zone, model, and warming level
    figures/projected_peaks.png       % change in the 1-in-10-year peak vs. warming
    figures/peak_frequency.png        days per year above today's 1-in-10-year peak

Run from the repository root (reuses functions from scripts/load_model.py):
    python scripts/projections.py

Method:
  * Base weather: every day of 2005-2025 (zones J, K) or 2002-2025 (zone C) with complete
    weather, i.e. 21-24 real "weather years", including outage days (their weather is valid).
  * Warming: dT = 0, 1, 2, 3 C added to today's and yesterday's daily mean temperature.
    Dewpoint is raised so relative humidity stays constant (Magnus formula).
  * Today's grid: models M1 (straight cooling line) and M5 (curved + dewpoint) fitted on
    2015-2025 exactly as in load_model.py; calendar effects use each date's own month,
    weekday and holidays; the time trend is fixed at mid-2025, so results isolate weather.
  * Metrics per weather year: the annual maximum daily peak, summarized as the median year
    and the 1-in-10-year peak (90th percentile of annual maxima); the average number of days
    per year above today's (dT = 0) 1-in-10-year peak; and the median winter (Dec-Feb) maximum.
  * Uncertainty: 1,000 draws of model coefficients from their Newey-West covariance; 95%
    ranges are the 2.5th-97.5th percentiles of the draws (changes are computed within each
    draw). M5 is the central estimate; M1 shows sensitivity to the curve's shape.
  * Supply-side context (illustrative): transmission capacity falls about 0.38% per 1 F
    (= 0.68% per 1 C) of warming (Meier et al. 2024, NYS Climate Impacts Assessment Ch. 6).
Caveats: uniform warming (heat waves may warm more); today's technology (no air-conditioning
growth or building/vehicle electrification, so the projected WINTER decrease does not reflect
New York's electrified future); +3 C extrapolates beyond the hottest observed days.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_model as L  # noqa: E402

RESULTS = Path("results")
FIGURES = Path("figures")
WARMING = [0.0, 1.0, 2.0, 3.0]
MODELS = ["M5", "M1"]
N_DRAWS = 1000
SEED = 2026
WEATHER_YEARS = {"J": (2005, 2025), "K": (2005, 2025), "C": (2002, 2025)}
TREND_2025 = 2025 - 2015 + 0.5
TRANSMISSION_PCT_PER_C = -0.38 * 1.8   # Meier et al. (2024): -0.38% per 1 F


def warm_dewpoint(t, td, dt):
    """New dewpoint after warming t by dt with relative humidity held constant (Magnus)."""
    a, b = 17.625, 243.04
    gamma = a * td / (b + td) - a * t / (b + t) + a * (t + dt) / (b + t + dt)
    return b * gamma / (a - gamma)


def base_weather(zone: str) -> pd.DataFrame:
    d = pd.read_parquet(L.PROCESSED / "zone_daily.parquet")
    y0, y1 = WEATHER_YEARS[zone]
    d = d[(d["zone"] == zone) & d["year"].between(y0, y1)]
    d = d.dropna(subset=["tmean_c", "tmean_lag1_c", "dewpoint_mean_c"]).copy()
    d["trend"] = TREND_2025
    full_years = d.groupby("year").size().loc[lambda s: s >= 300].index
    return d[d["year"].isin(full_years)].reset_index(drop=True)


def scenario(base: pd.DataFrame, dt: float) -> pd.DataFrame:
    s = base.copy()
    s["dewpoint_mean_c"] = warm_dewpoint(base["tmean_c"], base["dewpoint_mean_c"], dt)
    s["tmean_c"] = base["tmean_c"] + dt
    s["tmean_lag1_c"] = base["tmean_lag1_c"] + dt
    return s


def metrics(pred: np.ndarray, years: np.ndarray, months: np.ndarray, threshold: float):
    """pred: (n_days, n_draws). Returns a dict of arrays over draws (index 0 = point estimate)."""
    df = pd.DataFrame(pred)
    annual_max = df.groupby(years).max()
    winter = np.isin(months, [12, 1, 2])
    winter_max = df[winter].groupby(years[winter]).max()
    return {"median_annual_peak": annual_max.median().to_numpy(),
            "peak_1in10": annual_max.quantile(0.9).to_numpy(),
            "median_winter_peak": winter_max.median().to_numpy(),
            "days_above_today_1in10": (df > threshold).groupby(years).sum().mean().to_numpy()}


def main() -> None:
    rng = np.random.default_rng(SEED)
    data = L.load_data()
    rows = []
    for zone in L.ZONES:
        fit_d = data[(data["zone"] == zone) & data["year"].between(*L.FIT_YEARS)].reset_index(drop=True)
        bp = L.select_balance_points(fit_d)
        base = base_weather(zone)
        years, months = base["year"].to_numpy(), base["month"].to_numpy()
        for model in MODELS:
            res, names = L.fit(fit_d, model, bp[model])
            beta, cov = np.asarray(res.params), np.asarray(res.cov_params())
            # column 0 = point estimate, columns 1..N = coefficient draws
            draws = np.column_stack([beta, rng.multivariate_normal(beta, cov, size=N_DRAWS).T])
            preds = {dt: L.design(scenario(base, dt), model, *bp[model])[names].to_numpy() @ draws
                     for dt in WARMING}
            threshold = float(pd.Series(preds[0.0][:, 0]).groupby(years).max().quantile(0.9))
            results = {dt: metrics(p, years, months, threshold) for dt, p in preds.items()}
            r0 = results[0.0]
            for dt in WARMING:
                r = results[dt]
                row = {"zone": zone, "zone_name": L.ZONES[zone], "model": model, "warming_c": dt,
                       "weather_years": f"{years.min()}-{years.max()}"}
                for key in ["median_annual_peak", "peak_1in10", "median_winter_peak"]:
                    row[f"{key}_mw"] = round(r[key][0], 1)
                    row[f"{key}_mw_lo"] = round(np.percentile(r[key][1:], 2.5), 1)
                    row[f"{key}_mw_hi"] = round(np.percentile(r[key][1:], 97.5), 1)
                v = r["days_above_today_1in10"]
                row["days_above_today_1in10"] = round(v[0], 2)
                row["days_above_today_1in10_lo"] = round(np.percentile(v[1:], 2.5), 2)
                row["days_above_today_1in10_hi"] = round(np.percentile(v[1:], 97.5), 2)
                for key in ["peak_1in10", "median_winter_peak"]:
                    ch = r[key] - r0[key]
                    pct = 100 * ch / r0[key]
                    row[f"{key}_change_mw"] = round(ch[0], 1)
                    row[f"{key}_change_pct"] = round(pct[0], 2)
                    row[f"{key}_change_pct_lo"] = round(np.percentile(pct[1:], 2.5), 2)
                    row[f"{key}_change_pct_hi"] = round(np.percentile(pct[1:], 97.5), 2)
                row["transmission_capacity_change_pct"] = round(TRANSMISSION_PCT_PER_C * dt, 2)
                rows.append(row)
        print(f"Zone {zone}: done ({len(base):,} base days, {years.min()}-{years.max()})")

    out = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    out.to_csv(RESULTS / "projections_summary.csv", index=False)
    figures(out)

    pd.set_option("display.width", 220)
    show = out[["zone", "model", "warming_c", "peak_1in10_mw", "peak_1in10_change_mw", "peak_1in10_change_pct",
                "peak_1in10_change_pct_lo", "peak_1in10_change_pct_hi", "days_above_today_1in10",
                "median_winter_peak_change_pct", "transmission_capacity_change_pct"]]
    print(show.to_string(index=False))
    print(f"\nSaved {RESULTS / 'projections_summary.csv'}, {FIGURES / 'projected_peaks.png'}, "
          f"{FIGURES / 'peak_frequency.png'}")


def figures(out: pd.DataFrame) -> None:
    # 1) % change in the 1-in-10-year peak
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, zone in zip(axes, L.ZONES):
        color = L.ZONE_COLORS[zone]
        for model, ls, alpha, label in (("M5", "-", 0.18, "M5: curved + dewpoint (central)"),
                                        ("M1", "--", 0.08, "M1: straight line (conservative)")):
            s = out[(out["zone"] == zone) & (out["model"] == model)]
            ax.fill_between(s["warming_c"], s["peak_1in10_change_pct_lo"], s["peak_1in10_change_pct_hi"],
                            color=color, alpha=alpha, lw=0)
            ax.plot(s["warming_c"], s["peak_1in10_change_pct"], color=color, ls=ls, lw=2, marker="o", ms=4,
                    label=label)
        s5 = out[(out["zone"] == zone) & (out["model"] == "M5")]
        ax.plot(s5["warming_c"], s5["transmission_capacity_change_pct"], color="#52514e", lw=1.5, ls=":",
                label="Transmission capacity (Meier et al. 2024 rate)")
        ax.axhline(0, color="#9a9994", lw=1)
        last = s5.iloc[-1]
        ax.annotate(f"+{last['peak_1in10_change_mw']:,.0f} MW", (3, last["peak_1in10_change_pct"]),
                    textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8, color="#0b0b0b")
        ax.set_title(f"Zone {zone}: {L.ZONES[zone]}", loc="left", fontsize=10)
        ax.set_xticks(WARMING, [f"+{w:.0f} °C" for w in WARMING])
        L.style(ax)
    axes[0].set_ylabel("Change in 1-in-10-year peak load (%)", fontsize=9, color="#52514e")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.text(0.5, 0.005, "Today's grid held fixed; past weather years warmed uniformly (constant RH). "
             "Bands: 95% range from 1,000 coefficient draws.", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "projected_peaks.png", dpi=200, facecolor="white")
    plt.close(fig)

    # 2) frequency of extreme days
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for zone in L.ZONES:
        color = L.ZONE_COLORS[zone]
        for model, ls in (("M5", "-"), ("M1", "--")):
            s = out[(out["zone"] == zone) & (out["model"] == model)]
            if model == "M5":
                ax.fill_between(s["warming_c"], s["days_above_today_1in10_lo"], s["days_above_today_1in10_hi"],
                                color=color, alpha=0.10, lw=0)
            ax.plot(s["warming_c"], s["days_above_today_1in10"], color=color, ls=ls,
                    lw=2 if model == "M5" else 1.2, marker="o" if model == "M5" else None, ms=4,
                    label=f"Zone {zone}: {L.ZONES[zone]}" if model == "M5" else None)
    ax.plot([], [], color="#52514e", ls="--", lw=1.2, label="M1 (straight line), same colors")
    ax.set_xticks(WARMING, [f"+{w:.0f} °C" for w in WARMING])
    ax.set_title("Days per year above today's 1-in-10-year peak load", loc="left", fontsize=10)
    ax.set_ylabel("Days per year", fontsize=9, color="#52514e")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    L.style(ax)
    fig.text(0.5, 0.005, "Solid: M5 (curved + dewpoint) with 95% range; dashed: M1.", ha="center",
             fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIGURES / "peak_frequency.png", dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
