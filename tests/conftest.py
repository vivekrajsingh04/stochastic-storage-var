"""Shared fixtures: schema-accurate raw CSVs → aggregated fleet_daily DataFrame."""
import os
import sys

import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))

import generate_fixture  # noqa: E402
from download_backblaze import aggregate, iter_dir  # noqa: E402


@pytest.fixture(scope="session")
def fleet_csvs(tmp_path_factory):
    """Generate raw daily CSVs once and aggregate them to fleet-level tables."""
    root = tmp_path_factory.mktemp("backblaze")
    raw = str(root / "raw")
    generate_fixture.main(days=120, drives=400, out=raw)
    out_daily = str(root / "fleet_daily.csv")
    out_models = str(root / "model_stats.csv")
    aggregate(iter_dir(raw), out_daily, out_models)
    return out_daily, out_models


@pytest.fixture(scope="session")
def fleet_df(fleet_csvs):
    daily, _ = fleet_csvs
    return pd.read_csv(daily, parse_dates=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)
