"""Download NOAA GHCNh (hourly) and GHCN-Daily data for stations representing NYISO zones.

Run from the repository root:
    python scripts/download_noaa.py

Files are saved under data/raw/ (ignored by git). Re-running skips files that already exist,
so you can stop and restart safely.
"""
from pathlib import Path
import time

import requests

# Station ID -> description. IDs follow NOAA's "USW000" + WBAN pattern.
# Islip and JFK were verified; confirm others in the GHCNh station list if a station
# comes back "missing" for every year:
# https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/doc/ghcnh-station-list.txt
STATIONS = {
    "USW00014732": "LaGuardia (Zone J)",
    "USW00094728": "Central Park (Zone J, long climate record)",
    "USW00004781": "Islip MacArthur (Zone K)",
    "USW00014771": "Syracuse (Zone C)",
    "USW00014733": "Buffalo (Zone A)",
    "USW00014735": "Albany (Zone F)",
}

HOURLY_YEARS = range(1973, 2026)          # GHCNh: temperature + dewpoint + RH
DAILY_START, DAILY_END = "1950-01-01", "2025-12-31"   # GHCN-Daily: Tmax/Tmin

RAW = Path("data/raw")
HOURLY_DIR = RAW / "ghcnh"
DAILY_DIR = RAW / "ghcnd"

HOURLY_URL = ("https://www.ncei.noaa.gov/oa/global-historical-climatology-network/"
              "hourly/access/by-year/{y}/psv/GHCNh_{s}_{y}.psv")
DAILY_URL = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
             "&stations={s}&startDate={a}&endDate={b}&dataTypes=TMAX,TMIN"
             "&units=standard&format=csv")


def fetch(url: str, out: Path) -> str:
    """Download url to out unless it already exists. Returns a status string."""
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=180)
        except requests.RequestException as exc:
            status = f"error ({exc.__class__.__name__})"
            time.sleep(5)
            continue
        if r.status_code == 200:
            out.write_bytes(r.content)
            return "ok"
        if r.status_code == 404:
            return "missing (404)"
        status = f"http {r.status_code}"
        time.sleep(5)
    return status


def main() -> None:
    HOURLY_DIR.mkdir(parents=True, exist_ok=True)
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    log = []

    for sid, name in STATIONS.items():
        print(f"\n=== {sid}  {name} ===")

        status = fetch(DAILY_URL.format(s=sid, a=DAILY_START, b=DAILY_END),
                       DAILY_DIR / f"{sid}_daily.csv")
        print(f"  daily {DAILY_START[:4]}-{DAILY_END[:4]}: {status}")
        log.append((sid, "daily", status))

        for y in HOURLY_YEARS:
            status = fetch(HOURLY_URL.format(s=sid, y=y), HOURLY_DIR / f"{sid}_{y}.psv")
            if status != "skip":
                print(f"  hourly {y}: {status}")
                time.sleep(0.5)  # be polite to NOAA's server
            log.append((sid, y, status))

    missing = [(s, y) for s, y, st in log if st.startswith(("missing", "error", "http"))]
    print(f"\nDone. {len(missing)} files not downloaded.")
    for s, y in missing:
        print(f"  {s} {y}")


if __name__ == "__main__":
    main()
