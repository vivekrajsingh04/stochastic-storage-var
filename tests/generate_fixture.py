"""
Generate a schema-accurate Backblaze Drive Stats mini-fixture for smoke tests.

Writes raw daily CSVs (same columns as the real dataset) to tests/raw/, so the
full loader → pipeline → dashboard chain can be exercised in CI without the
~1 GB download. NOT a substitute for real data — run data/download_backblaze.py
for actual analysis.

Usage:
    python tests/generate_fixture.py --days 150 --drives 800
    python data/download_backblaze.py --raw-dir tests/raw
    python StorageIQ_Pipeline.py
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("ST12000NM0008", 12e12), ("ST8000DM002", 8e12), ("HGST HMS5C4040BLE640", 4e12),
    ("TOSHIBA MG08ACA16TA", 16e12), ("WDC WUH721816ALE6L4", 16e12),
]


def main(days: int, drives: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    out = os.path.join(HERE, "raw")
    os.makedirs(out, exist_ok=True)

    serials = [f"ZX{i:06d}" for i in range(drives)]
    model_of = rng.choice(len(MODELS), drives, p=[0.3, 0.25, 0.2, 0.15, 0.1])
    # Hazard deliberately exaggerated (~8% AFR) so a small fixture fleet still
    # produces ~1 failure/day — enough signal to exercise every code path.
    base_fail = 2.2e-4
    dates = pd.date_range("2025-01-01", periods=days, freq="D")

    for d in dates:
        # mild seasonality + one thermal event window
        hazard = base_fail * (1 + 0.3 * np.sin(2 * np.pi * d.dayofyear / 90))
        thermal = 30 <= (d - dates[0]).days <= 36
        fail = rng.random(drives) < hazard * (3.0 if thermal else 1.0)
        rows = {
            "date": d.strftime("%Y-%m-%d"),
            "serial_number": serials,
            "model": [MODELS[m][0] for m in model_of],
            "capacity_bytes": [int(MODELS[m][1]) for m in model_of],
            "failure": fail.astype(int),
            "smart_5_raw": rng.poisson(2, drives),
            "smart_9_raw": rng.integers(1000, 40000, drives),
            "smart_187_raw": rng.poisson(0.2, drives),
            "smart_188_raw": rng.poisson(0.1, drives),
            "smart_194_raw": rng.normal(38 if thermal else 29, 3, drives).round(0),
            "smart_197_raw": rng.poisson(0.5, drives),
            "smart_198_raw": rng.poisson(0.3, drives),
        }
        pd.DataFrame(rows).to_csv(os.path.join(out, f"{d:%Y-%m-%d}.csv"), index=False)

    print(f"✔ wrote {days} daily CSVs ({drives} drives) to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--drives", type=int, default=5000)
    args = ap.parse_args()
    main(args.days, args.drives)
