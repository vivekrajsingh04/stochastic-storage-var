"""
Auto-refresh: detect and ingest the newest published Backblaze quarter.
=======================================================================
Backblaze publishes Drive Stats quarterly, ~4-6 weeks after each quarter
ends. This script (run monthly by .github/workflows/refresh-data.yml):

  1. Works out the newest quarter that should be published by now
  2. Compares against what the deployed dashboard payload already covers
  3. If newer data should exist, verifies the ZIP is actually live
  4. Downloads the latest 4 quarters and rebuilds the fleet tables

Prints exactly one status line the workflow keys off:
  UP-TO-DATE          payload already covers the newest published quarter
  NOT-PUBLISHED-YET   quarter expected but ZIP not on Backblaze's CDN yet
  UPDATED <quarters>  fleet tables rebuilt — run the pipeline next

Usage:  python data/auto_refresh.py
"""

import json
import os
from datetime import date
from urllib.request import Request, urlopen

from download_backblaze import BASE_URL, DATA_DIR, aggregate, download, iter_zips

REPO = os.path.dirname(DATA_DIR)
PAYLOAD = os.path.join(REPO, "StorageIQ_Dashboard", "dashboard_data.json")
PUBLICATION_LAG_DAYS = 40   # Backblaze's typical release delay
N_QUARTERS = 4              # rolling window aggregated into one series

QUARTER_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}


def quarter_end(q: int, year: int) -> date:
    m, d = QUARTER_END[q]
    return date(year, m, d)


def published_quarters(today: date):
    """Newest-first (quarter, year) pairs whose data should be public by now."""
    q, y = (today.month - 1) // 3 + 1, today.year
    out = []
    while len(out) < N_QUARTERS + 4:
        q -= 1
        if q == 0:
            q, y = 4, y - 1
        if (today - quarter_end(q, y)).days >= PUBLICATION_LAG_DAYS:
            out.append((q, y))
    return out


def coverage_end() -> date | None:
    if not os.path.exists(PAYLOAD):
        return None
    try:
        with open(PAYLOAD) as f:
            return date.fromisoformat(json.load(f)["meta"]["end"])
    except Exception:
        return None


def zip_is_live(quarter: str) -> bool:
    """Cheap availability probe: request the first byte of the quarterly ZIP."""
    req = Request(BASE_URL.format(quarter=quarter),
                  headers={"User-Agent": "storageiq-refresh", "Range": "bytes=0-0"})
    try:
        with urlopen(req, timeout=30) as r:
            return r.status in (200, 206)
    except Exception:
        return False


def main():
    today = date.today()
    quarters = published_quarters(today)
    newest_q, newest_y = quarters[0]
    newest_name = f"Q{newest_q}_{newest_y}"

    covered = coverage_end()
    if covered and quarter_end(newest_q, newest_y) <= covered:
        print(f"UP-TO-DATE (payload covers through {covered}, newest is {newest_name})")
        return

    if not zip_is_live(newest_name):
        print(f"NOT-PUBLISHED-YET ({newest_name} expected but not on CDN)")
        return

    window = [f"Q{q}_{y}" for q, y in quarters[:N_QUARTERS]][::-1]  # oldest first
    zpaths = [download(q, DATA_DIR) for q in window]
    aggregate(iter_zips(zpaths),
              os.path.join(DATA_DIR, "fleet_daily.csv"),
              os.path.join(DATA_DIR, "model_stats.csv"))
    for z in zpaths:
        os.remove(z)
    print(f"UPDATED {','.join(window)}")


if __name__ == "__main__":
    main()
