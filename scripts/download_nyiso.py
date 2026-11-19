"""Download NYISO Integrated Real-Time Actual Load (P-58C), hourly by zone.

Monthly zip archives from https://mis.nyiso.com/public/P-58Clist.htm
Run from the repository root:
    python scripts/download_nyiso.py
"""
from pathlib import Path
import time
import zipfile

import requests

START_YEAR, END_YEAR = 2001, 2025
URL = "https://mis.nyiso.com/public/csv/palIntegrated/{y}{m:02d}01palIntegrated_csv.zip"

ZIP_DIR = Path("data/raw/nyiso/zips")
CSV_DIR = Path("data/raw/nyiso/csv")


def main() -> None:
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    missing = []

    for y in range(START_YEAR, END_YEAR + 1):
        for m in range(1, 13):
            out = ZIP_DIR / f"{y}{m:02d}_palIntegrated.zip"
            if not (out.exists() and out.stat().st_size > 0):
                r = requests.get(URL.format(y=y, m=m), timeout=120)
                if r.status_code != 200:
                    print(f"{y}-{m:02d}: http {r.status_code}")
                    missing.append(f"{y}-{m:02d}")
                    continue
                out.write_bytes(r.content)
                print(f"{y}-{m:02d}: ok")
                time.sleep(0.5)
            # Each monthly zip holds one CSV per day
            with zipfile.ZipFile(out) as z:
                z.extractall(CSV_DIR)

    print(f"\nDone. Missing months: {missing or 'none'}")
    print(f"Daily CSVs are in {CSV_DIR}/ -- next run: python scripts/build_nyiso_hourly.py")


if __name__ == "__main__":
    main()
