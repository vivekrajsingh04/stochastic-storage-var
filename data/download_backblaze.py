"""
Backblaze Drive Stats — downloader & aggregator
================================================
Downloads real-world hard-drive telemetry published quarterly by Backblaze
(https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
and aggregates the raw per-drive daily snapshots into the two fleet-level
tables consumed by the StorageIQ pipeline:

    data/fleet_daily.csv    one row per calendar day (fleet aggregates)
    data/model_stats.csv    one row per drive model (drive-days, failures, AFR)

Raw schema (one CSV per day inside each quarterly ZIP):
    date, serial_number, model, capacity_bytes, failure,
    smart_1_normalized, smart_1_raw, ... smart_255_raw

Usage:
    python data/download_backblaze.py --quarter Q1_2025
    python data/download_backblaze.py --quarter Q4_2024 --keep-raw
    # multiple quarters aggregate into one continuous series:
    python data/download_backblaze.py --quarter Q2_2024,Q3_2024,Q4_2024,Q1_2025
    # or aggregate a folder of daily CSVs you already extracted:
    python data/download_backblaze.py --raw-dir /path/to/extracted

A single quarter is ~1 GB zipped / ~10 GB extracted (~250-300k drives/day).
Aggregation streams file-by-file, so peak memory stays low.
"""

import argparse
import io
import os
import sys
import zipfile
from glob import glob
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

BASE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_{quarter}.zip"
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# SMART attributes with strong failure signal (Backblaze's own shortlist)
SMART_COLS = {
    "smart_5_raw":   "Realloc_Sectors",       # reallocated sector count
    "smart_9_raw":   "Power_On_Hours",
    "smart_187_raw": "Uncorrectable_Errors",  # reported uncorrectable
    "smart_188_raw": "Command_Timeouts",
    "smart_194_raw": "Temperature_C",
    "smart_197_raw": "Pending_Sectors",       # current pending sector count
    "smart_198_raw": "Offline_Uncorrectable",
}
BASE_COLS = ["date", "serial_number", "model", "capacity_bytes", "failure"]


def download(quarter: str, dest: str, retries: int = 8) -> str:
    """Stream the quarterly ZIP to disk. Resumes a partial download via HTTP
    Range requests and retries on stalls (60 s socket timeout)."""
    url = BASE_URL.format(quarter=quarter)
    zpath = os.path.join(dest, f"data_{quarter}.zip")
    part = zpath + ".part"
    if os.path.exists(zpath):
        print(f"✔ {zpath} already downloaded")
        return zpath
    print(f"⬇ Downloading {url}\n  (quarterly archives are ~1 GB — this takes a while)")

    for attempt in range(retries):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": "storageiq-loader"}
        if done:
            headers["Range"] = f"bytes={done}-"
            print(f"\n  resuming from {done / 1e6:,.0f} MB")
        try:
            with urlopen(Request(url, headers=headers), timeout=60) as r:
                if done and r.status != 206:  # server ignored Range → restart
                    done = 0
                total = done + int(r.headers.get("Content-Length") or 0)
                with open(part, "ab" if done else "wb") as f:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            sys.stdout.write(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB")
                            sys.stdout.flush()
            if total and done < total:
                raise IOError(f"connection closed early at {done}/{total} bytes")
            os.rename(part, zpath)
            print("\n✔ download complete")
            return zpath
        except Exception as e:
            print(f"\n  ⚠ {e} — retry {attempt + 1}/{retries}")
    raise SystemExit(f"Download failed after {retries} attempts: {url}")


def _daily_aggregate(df: pd.DataFrame) -> dict:
    """Collapse one day of per-drive rows into a single fleet-level row."""
    row = {
        "Timestamp":   df["date"].iloc[0],
        "Drive_Count": len(df),
        "Capacity_PB": df["capacity_bytes"].clip(lower=0).sum() / 1e15,
        "Failures":    int(df["failure"].sum()),
    }
    row["Failure_Rate"] = row["Failures"] / max(row["Drive_Count"], 1)
    row["AFR_Pct"] = row["Failure_Rate"] * 365 * 100  # annualized failure rate
    for raw, name in SMART_COLS.items():
        if raw in df.columns:
            row[f"Mean_{name}"] = pd.to_numeric(df[raw], errors="coerce").mean()
    return row


def aggregate(csv_iter, out_daily: str, out_models: str) -> None:
    """Stream daily CSVs → fleet_daily.csv + model_stats.csv."""
    daily_rows = []
    model_acc = {}  # model -> [drive_days, failures, capacity_bytes]

    for i, (name, fh) in enumerate(csv_iter):
        usecols = lambda c: c in BASE_COLS or c in SMART_COLS
        df = pd.read_csv(fh, usecols=usecols, low_memory=False)
        if df.empty:
            continue
        daily_rows.append(_daily_aggregate(df))
        g = df.groupby("model").agg(
            drive_days=("serial_number", "size"),
            failures=("failure", "sum"),
            capacity=("capacity_bytes", "max"),
        )
        for m, r in g.iterrows():
            acc = model_acc.setdefault(m, [0, 0, 0])
            acc[0] += int(r["drive_days"])
            acc[1] += int(r["failures"])
            acc[2] = max(acc[2], r["capacity"])
        if (i + 1) % 10 == 0:
            print(f"  processed {i + 1} daily files…")

    daily = pd.DataFrame(daily_rows).sort_values("Timestamp")
    daily.to_csv(out_daily, index=False)

    models = pd.DataFrame(
        [
            {
                "Model": m,
                "Drive_Days": dd,
                "Failures": f,
                "Capacity_TB": cap / 1e12,
                "AFR_Pct": (f / dd) * 365 * 100 if dd else 0.0,
            }
            for m, (dd, f, cap) in model_acc.items()
        ]
    ).sort_values("Drive_Days", ascending=False)
    models.to_csv(out_models, index=False)

    print(f"✔ wrote {out_daily}  ({len(daily)} days)")
    print(f"✔ wrote {out_models} ({len(models)} drive models)")


def iter_zip(zpath):
    with zipfile.ZipFile(zpath) as z:
        for n in sorted(z.namelist()):
            if n.endswith(".csv") and not n.startswith("__MACOSX"):
                with z.open(n) as fh:
                    yield n, io.TextIOWrapper(fh, encoding="utf-8", errors="replace")


def iter_dir(raw_dir):
    for p in sorted(glob(os.path.join(raw_dir, "**", "*.csv"), recursive=True)):
        yield p, p


def iter_zips(zpaths):
    for zpath in zpaths:
        yield from iter_zip(zpath)


def main():
    ap = argparse.ArgumentParser(description="Backblaze Drive Stats loader")
    ap.add_argument("--quarter",
                    help="e.g. Q1_2025, or comma-separated list Q2_2024,Q3_2024,Q1_2025")
    ap.add_argument("--raw-dir", help="folder of already-extracted daily CSVs")
    ap.add_argument("--keep-raw", action="store_true", help="keep the downloaded ZIP")
    args = ap.parse_args()

    out_daily = os.path.join(DATA_DIR, "fleet_daily.csv")
    out_models = os.path.join(DATA_DIR, "model_stats.csv")

    if args.raw_dir:
        aggregate(iter_dir(args.raw_dir), out_daily, out_models)
    elif args.quarter:
        quarters = [q.strip() for q in args.quarter.split(",") if q.strip()]
        zpaths = [download(q, DATA_DIR) for q in quarters]
        aggregate(iter_zips(zpaths), out_daily, out_models)
        if not args.keep_raw:
            for zpath in zpaths:
                os.remove(zpath)
            print("✔ removed ZIPs (use --keep-raw to keep them)")
    else:
        ap.error("provide --quarter or --raw-dir")


if __name__ == "__main__":
    main()
