"""Research Question 2: how sensitive is daily peak load to temperature and humidity?

Inputs  (data/processed/, from build_zone_daily.py): zone_daily.parquet, zone_hourly.parquet
Outputs:
    results/load_model_coefficients.csv   full-sample (2015-2025) fits with 95% CIs
    results/load_model_validation.csv     out-of-sample accuracy, train 2015-2021 / test 2022-2025
    results/load_sensitivity_over_time.csv  M1 cooling sensitivity in rolling 5-year windows
    figures/load_vs_temperature.png       daily peak vs. temperature with fitted M1 curve
    figures/model_comparison.png          out-of-sample error of M1-M3
    figures/sensitivity_over_time.png     rolling-window cooling sensitivity

Run from the repository root:
    python scripts/load_model.py

Model (one per zone: J = NYC/LaGuardia, K = Long Island/Islip, C = Central/Syracuse):
    peak_d = a + b_cool*max(0, C_d - Tc) + b_lag*max(0, C_{d-1} - Tc) + b_heat*max(0, Th - T_d)
             [+ b_humid*max(0, Td_d - Td0)*1(T_d > Tc)]   (M2 only)
             + month + day-of-week + holiday + linear time trend + error
  T  = daily mean temperature (C); Td = daily mean dewpoint (C); "yesterday" terms capture
       heat building up in buildings during multi-day heat waves.
  M1: C = T                (temperature only)
  M2: M1 + humidity term   (dewpoint above Td0, applied on cooling days only)
  M3: C = daily mean NWS heat index on the cooling side (heat side still uses T)
  M4: M1 + b_cool2*max(0, T_d - Tc)^2   (curved cooling response; added 2026-11-18 because
      M1 under-predicted the hottest test days by ~800 MW in zone J)
  M5: M4 + M2's humidity term            (curved + dewpoint)
  Balance points Tc, Th (and Td0) are chosen by grid search on the fitting data only:
  M1 searches Tc in 14-24 C and Th in 6-18 C (Th <= Tc); M2 keeps M1's Tc/Th and searches
  Td0 in 8-24 C; M3 keeps M1's Th and searches its own Tc in 14-26 C (heat-index scale);
  M4 uses M1's balance points and M5 uses M2's.
Sample: 2015-2025 days not flagged as outages (exclude) or COVID lockdown, with complete
  weather. Out-of-sample test: balance points and coefficients fit on 2015-2021 only, then
  scored on 2022-2025 overall and on the top 5% highest-load test days.
Sustained cooling sensitivity = d(peak)/dT when today AND yesterday are 1 C warmer:
  b_cool + b_lag (+ 2*b_cool2*(T - Tc) for M4/M5, so it grows with temperature). Reported at
  the zone's mean summer (JJA) daily mean temperature and at its 95th percentile ("hot day"),
  in MW/C and as % of the zone's mean summer daily peak.
Uncertainty: Newey-West (HAC, 7-day lag) standard errors because errors are correlated
  from one day to the next.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

PROCESSED = Path("data/processed")
RESULTS = Path("results")
FIGURES = Path("figures")
LOCAL_TZ = "America/New_York"

ZONES = {"J": "NYC (LaGuardia)", "K": "Long Island (Islip)", "C": "Central (Syracuse)"}
ZONE_COLORS = {"J": "#2a78d6", "K": "#eb6834", "C": "#1baf7a"}   # same as climate_trends.png
FIT_YEARS = (2015, 2025)
TRAIN_YEARS = (2015, 2021)
TEST_YEARS = (2022, 2025)
TC_GRID = np.arange(14, 24.01, 0.5)
TH_GRID = np.arange(6, 18.01, 0.5)
TD0_GRID = np.arange(8, 24.01, 1.0)   # widened after first test: K hit the 20 C edge
TC_HI_GRID = np.arange(14, 26.01, 0.5)
HAC_LAGS = 7
WINDOW_YEARS = 5


# ---------------------------------------------------------------- data
def load_data() -> pd.DataFrame:
    d = pd.read_parquet(PROCESSED / "zone_daily.parquet")
    h = pd.read_parquet(PROCESSED / "zone_hourly.parquet", columns=["time_utc", "zone", "heat_index_c"])
    h["date"] = h["time_utc"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None).dt.normalize()
    hi = h.groupby(["zone", "date"])["heat_index_c"].agg(["mean", "count"]).reset_index()
    hi.loc[hi["count"] < 18, "mean"] = np.nan
    hi = hi.rename(columns={"mean": "hi_mean_c"}).drop(columns="count")
    d = d.merge(hi, on=["zone", "date"], how="left").sort_values(["zone", "date"])
    d["hi_mean_lag1_c"] = d.groupby("zone")["hi_mean_c"].shift(1)   # zone_daily has no date gaps
    d["trend"] = d["year"] - 2015 + (d["date"].dt.day_of_year - 1) / 365.25
    need = ["peak_mw", "tmean_c", "tmean_lag1_c", "dewpoint_mean_c", "hi_mean_c", "hi_mean_lag1_c"]
    keep = d[need].notna().all(axis=1) & ~d["exclude"] & ~d["covid_lockdown"]
    return d[keep].reset_index(drop=True)


def calendar_matrix(d: pd.DataFrame) -> pd.DataFrame:
    month = pd.get_dummies(d["month"].astype(int), prefix="m", drop_first=True, dtype=float)
    dow = pd.get_dummies(d["dow"].astype(int), prefix="dow", drop_first=True, dtype=float)
    month = month.reindex(columns=[f"m_{i}" for i in range(2, 13)], fill_value=0.0)
    dow = dow.reindex(columns=[f"dow_{i}" for i in range(1, 7)], fill_value=0.0)
    cal = pd.concat([month, dow], axis=1)
    cal["holiday"] = d["is_holiday"].astype(float).values
    cal["trend"] = d["trend"].values
    return cal


def weather_matrix(d: pd.DataFrame, model: str, tc: float, th: float, td0: float | None) -> pd.DataFrame:
    t, t_lag = d["tmean_c"], d["tmean_lag1_c"]
    c, c_lag = (d["hi_mean_c"], d["hi_mean_lag1_c"]) if model == "M3" else (t, t_lag)
    w = pd.DataFrame({"cool": (c - tc).clip(lower=0), "cool_lag": (c_lag - tc).clip(lower=0),
                      "heat": (th - t).clip(lower=0)})
    if model in ("M4", "M5"):
        w["cool_sq"] = w["cool"] ** 2
    if model in ("M2", "M5"):
        w["humid"] = (d["dewpoint_mean_c"] - td0).clip(lower=0) * (t > tc)
    return w


def design(d, model, tc, th, td0=None) -> pd.DataFrame:
    X = pd.concat([weather_matrix(d, model, tc, th, td0).reset_index(drop=True),
                   calendar_matrix(d).reset_index(drop=True)], axis=1)
    X.insert(0, "const", 1.0)
    return X


# ---------------------------------------------------------------- fitting
def rmse_of(d, model, tc, th, td0=None) -> float:
    X = design(d, model, tc, th, td0).to_numpy()
    y = d["peak_mw"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(np.sqrt(np.mean((y - X @ beta) ** 2)))


def select_balance_points(d: pd.DataFrame) -> dict:
    """Grid search on the given data only. Returns balance points for M1, M2, M3."""
    best = min(((rmse_of(d, "M1", tc, th), tc, th) for tc in TC_GRID for th in TH_GRID if th <= tc))
    _, tc, th = best
    td0 = min((rmse_of(d, "M2", tc, th, td0), td0) for td0 in TD0_GRID)[1]
    tc_hi = min((rmse_of(d, "M3", c, th), c) for c in TC_HI_GRID if c >= th)[1]
    return {"M1": (tc, th, None), "M2": (tc, th, td0), "M3": (tc_hi, th, None),
            "M4": (tc, th, None), "M5": (tc, th, td0)}


def fit(d, model, params):
    tc, th, td0 = params
    X = design(d, model, tc, th, td0)
    return sm.OLS(d["peak_mw"].to_numpy(), X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}), X.columns


def sustained(res, names, denom, t_ref=None, tc=None):
    """Sustained cooling sensitivity with HAC 95% CI, in MW/C and % of denom.
    For curved models the slope depends on temperature, so it is evaluated at t_ref."""
    names = list(names)
    g = np.zeros(len(names))
    g[names.index("cool")] = 1.0
    g[names.index("cool_lag")] = 1.0
    if "cool_sq" in names and t_ref is not None:
        g[names.index("cool_sq")] = 2 * max(0.0, t_ref - tc)
    params, cov = np.asarray(res.params), np.asarray(res.cov_params())
    est = float(g @ params)
    se = float(np.sqrt(g @ cov @ g))
    return est, est - 1.96 * se, est + 1.96 * se, 100 * est / denom, 100 * 1.96 * se / denom


# ---------------------------------------------------------------- analyses
def coefficients(data) -> pd.DataFrame:
    rows = []
    for zone in ZONES:
        d = data[(data["zone"] == zone) & data["year"].between(*FIT_YEARS)].reset_index(drop=True)
        bp = select_balance_points(d)
        summer_peak = d.loc[d["month"].isin([6, 7, 8]), "peak_mw"].mean()
        winter_peak = d.loc[d["month"].isin([12, 1, 2]), "peak_mw"].mean()
        jja_t = d.loc[d["month"].isin([6, 7, 8]), "tmean_c"]
        t_mean, t_hot = float(jja_t.mean()), float(jja_t.quantile(0.95))
        for model, params in bp.items():
            res, names = fit(d, model, params)
            est, lo, hi, pct, pct_ci = sustained(res, names, summer_peak, t_mean, params[0])
            h_est, h_lo, h_hi, h_pct, _ = sustained(res, names, summer_peak, t_hot, params[0])
            k = list(names).index("heat")
            row = {"zone": zone, "zone_name": ZONES[zone], "model": model, "n_days": len(d),
                   "cool_balance_c": params[0], "heat_balance_c": params[1], "dewpoint_threshold_c": params[2],
                   "sustained_cooling_mw_per_c": round(est, 1), "ci95_low": round(lo, 1), "ci95_high": round(hi, 1),
                   "sustained_cooling_pct_of_summer_peak_per_c": round(pct, 2),
                   "pct_ci95_halfwidth": round(pct_ci, 2),
                   "summer_mean_temp_c": round(t_mean, 1), "hot_day_temp_c": round(t_hot, 1),
                   "hot_day_cooling_mw_per_c": round(h_est, 1), "hot_day_ci95_low": round(h_lo, 1),
                   "hot_day_ci95_high": round(h_hi, 1), "hot_day_pct_of_summer_peak_per_c": round(h_pct, 2),
                   "heating_mw_per_c": round(np.asarray(res.params)[k], 1),
                   "heating_pct_of_winter_peak_per_c": round(100 * np.asarray(res.params)[k] / winter_peak, 2),
                   "mean_summer_peak_mw": round(summer_peak, 0), "r2": round(res.rsquared, 3)}
            if model in ("M4", "M5"):
                q = list(names).index("cool_sq")
                row["cool_sq_mw_per_c2"] = round(np.asarray(res.params)[q], 2)
                row["cool_sq_p_value"] = round(np.asarray(res.pvalues)[q], 4)
            if model in ("M2", "M5"):
                m = list(names).index("humid")
                row["humid_mw_per_c_dewpoint"] = round(np.asarray(res.params)[m], 1)
                row["humid_p_value"] = round(np.asarray(res.pvalues)[m], 4)
            rows.append(row)
    return pd.DataFrame(rows)


def validation(data) -> pd.DataFrame:
    rows = []
    for zone in ZONES:
        z = data[data["zone"] == zone]
        train = z[z["year"].between(*TRAIN_YEARS)].reset_index(drop=True)
        test = z[z["year"].between(*TEST_YEARS)].reset_index(drop=True)
        bp = select_balance_points(train)
        top = test["peak_mw"] >= test["peak_mw"].quantile(0.95)
        for model, (tc, th, td0) in bp.items():
            Xtr = design(train, model, tc, th, td0)
            res = sm.OLS(train["peak_mw"].to_numpy(), Xtr).fit()
            pred = res.predict(design(test, model, tc, th, td0)[Xtr.columns])
            err = test["peak_mw"].to_numpy() - pred.to_numpy()
            mean_load = test["peak_mw"].mean()
            rows.append({"zone": zone, "model": model, "n_test_days": len(test),
                         "rmse_mw": round(np.sqrt(np.mean(err ** 2)), 1),
                         "rmse_pct": round(100 * np.sqrt(np.mean(err ** 2)) / mean_load, 2),
                         "mape_pct": round(100 * np.mean(np.abs(err) / test["peak_mw"].to_numpy()), 2),
                         "top5_rmse_mw": round(np.sqrt(np.mean(err[top] ** 2)), 1),
                         "top5_bias_mw": round(np.mean(err[top]), 1),
                         "cool_balance_c": tc, "heat_balance_c": th, "dewpoint_threshold_c": td0})
    return pd.DataFrame(rows)


def sensitivity_over_time(data, coef) -> pd.DataFrame:
    rows = []
    for zone in ZONES:
        z = data[data["zone"] == zone]
        c = coef[(coef["zone"] == zone) & (coef["model"] == "M1")].iloc[0]
        params = (c["cool_balance_c"], c["heat_balance_c"], None)
        first = int(z.groupby("year").size().loc[lambda s: s >= 300].index.min())
        for start in range(first, FIT_YEARS[1] - WINDOW_YEARS + 2):
            w = z[z["year"].between(start, start + WINDOW_YEARS - 1)].reset_index(drop=True)
            res, names = fit(w, "M1", params)
            summer_peak = w.loc[w["month"].isin([6, 7, 8]), "peak_mw"].mean()
            est, lo, hi, pct, pct_ci = sustained(res, names, summer_peak)
            rows.append({"zone": zone, "window": f"{start}-{start + WINDOW_YEARS - 1}",
                         "center_year": start + WINDOW_YEARS // 2, "n_days": len(w),
                         "sustained_cooling_mw_per_c": round(est, 1), "ci95_low": round(lo, 1),
                         "ci95_high": round(hi, 1), "pct_of_summer_peak_per_c": round(pct, 2),
                         "pct_ci95_halfwidth": round(pct_ci, 2)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def style(ax):
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9994")
    ax.tick_params(colors="#52514e", labelsize=8)


def fig_load_vs_temperature(data, coef, out):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, zone in zip(axes, ZONES):
        d = data[(data["zone"] == zone) & data["year"].between(*FIT_YEARS)].reset_index(drop=True)
        c = coef[(coef["zone"] == zone) & (coef["model"] == "M1")].iloc[0]
        params = (c["cool_balance_c"], c["heat_balance_c"], None)
        ax.scatter(d["tmean_c"], d["peak_mw"], s=4, color=ZONE_COLORS[zone], alpha=0.25, lw=0)
        # Partial dependence: set today's and yesterday's temperature to T for every day, average predictions
        grid = np.linspace(d["tmean_c"].quantile(0.002), d["tmean_c"].quantile(0.998), 80)
        for model, color, ls, label in (("M1", "#0b0b0b", "-", "M1: straight cooling line"),
                                        ("M4", "#6250d6", "--", "M4: curved cooling")):
            res, names = fit(d, model, params)
            curve = []
            for t in grid:
                dd = d.copy()
                dd["tmean_c"] = t
                dd["tmean_lag1_c"] = t
                curve.append(res.predict(design(dd, model, *params)[names]).mean())
            ax.plot(grid, curve, color=color, lw=2, ls=ls, label=label)
        for x in params[:2]:
            ax.axvline(x, color="#9a9994", lw=1, ls=":")
        ax.set_title(f"Zone {zone}: {ZONES[zone]}", loc="left", fontsize=10)
        ax.set_xlabel("Daily mean temperature (°C)", fontsize=9, color="#52514e")
        style(ax)
    axes[0].set_ylabel("Daily peak load (MW)", fontsize=9, color="#52514e")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(0.5, 0.005, "Points: days 2015-2025 (outages and COVID lockdown excluded). Dotted lines: fitted "
             "heating and cooling balance points. Sources: NYISO, NOAA GHCNh.", ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def fig_model_comparison(val, out):
    shades = {"M1": "#0b0b0b", "M2": "#52514e", "M3": "#9a9994", "M4": "#6250d6", "M5": "#6250d6"}
    markers = {"M1": "o", "M2": "s", "M3": "^", "M4": "D", "M5": "P"}
    models = list(shades)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, col, title in zip(axes, ["rmse_pct", "top5_rmse_mw"],
                              ["All test days: RMSE (% of mean peak)", "Top 5% load days: RMSE (MW)"]):
        for i, zone in enumerate(ZONES):
            for k, model in enumerate(models):
                v = val[(val["zone"] == zone) & (val["model"] == model)][col].iloc[0]
                ax.scatter(v, i + (k - 2) * 0.14, color=shades[model], marker=markers[model], s=45,
                           label=model if i == 0 else None, zorder=3)
        ax.set_yticks(range(len(ZONES)), [f"Zone {z}" for z in ZONES])
        ax.set_title(title, loc="left", fontsize=10)
        style(ax)
        ax.grid(axis="x", color="#e4e3df", lw=0.6)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
                   title="M1 temp\nM2 +dewpoint\nM3 heat index\nM4 curved\nM5 curved+dewpoint",
                   title_fontsize=7)
    fig.text(0.5, 0.01, "Trained on 2015-2021, tested on 2022-2025 (lower is better).",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def fig_sensitivity(sens, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    for zone in ZONES:
        s = sens[sens["zone"] == zone]
        ax.fill_between(s["center_year"], s["pct_of_summer_peak_per_c"] - s["pct_ci95_halfwidth"],
                        s["pct_of_summer_peak_per_c"] + s["pct_ci95_halfwidth"],
                        color=ZONE_COLORS[zone], alpha=0.15, lw=0)
        ax.plot(s["center_year"], s["pct_of_summer_peak_per_c"], color=ZONE_COLORS[zone], lw=2,
                marker="o", ms=4, label=f"Zone {zone}: {ZONES[zone]}")
    ax.set_title("Cooling sensitivity of daily peak load, rolling 5-year windows (% of summer peak per °C)",
                 loc="left", fontsize=10)
    ax.set_xlabel("Center year of 5-year window", fontsize=9, color="#52514e")
    ax.set_xticks(range(2004, 2025, 2))
    ax.legend(frameon=False, fontsize=8)
    style(ax)
    fig.text(0.5, 0.005, "Model M1 with full-sample balance points; shaded = 95% CI (Newey-West).",
             ha="center", fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    data = load_data()
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    coef = coefficients(data)
    val = validation(data)
    sens = sensitivity_over_time(data, coef)
    coef.to_csv(RESULTS / "load_model_coefficients.csv", index=False)
    val.to_csv(RESULTS / "load_model_validation.csv", index=False)
    sens.to_csv(RESULTS / "load_sensitivity_over_time.csv", index=False)
    fig_load_vs_temperature(data, coef, FIGURES / "load_vs_temperature.png")
    fig_model_comparison(val, FIGURES / "model_comparison.png")
    fig_sensitivity(sens, FIGURES / "sensitivity_over_time.png")

    pd.set_option("display.width", 200)
    print("Full-sample fits (2015-2025):")
    print(coef[["zone", "model", "cool_balance_c", "heat_balance_c", "dewpoint_threshold_c",
                "sustained_cooling_mw_per_c", "hot_day_cooling_mw_per_c",
                "sustained_cooling_pct_of_summer_peak_per_c", "hot_day_pct_of_summer_peak_per_c",
                "r2"]].to_string(index=False))
    print("\nOut-of-sample (train 2015-2021, test 2022-2025):")
    print(val[["zone", "model", "rmse_mw", "rmse_pct", "mape_pct", "top5_rmse_mw", "top5_bias_mw"]]
          .to_string(index=False))
    print(f"\nSaved results/ (3 tables) and figures/ (3 figures)")


if __name__ == "__main__":
    main()
