"""Smoke + correctness tests for the StorageIQ pipeline (no network, no 1 GB download)."""
import numpy as np
import pandas as pd
import pytest

import StorageIQ_Pipeline as pipe


# ── loader / aggregation ─────────────────────────────────────────────
def test_aggregate_schema(fleet_csvs):
    daily, models = fleet_csvs
    df = pd.read_csv(daily)
    for col in ["Timestamp", "Drive_Count", "Capacity_PB", "Failures",
                "Failure_Rate", "Mean_Temperature_C"]:
        assert col in df.columns
    assert len(df) == 120
    assert (df["Drive_Count"] > 0).all()

    m = pd.read_csv(models)
    assert {"Model", "Drive_Days", "Failures", "AFR_Pct"} <= set(m.columns)
    assert (m["Drive_Days"] > 0).all()


# ── features ─────────────────────────────────────────────────────────
def test_zscore_is_causal():
    """Changing future values must not change past z-scores (no look-ahead)."""
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(10, 2, 100))
    base = pipe.zscore(s)
    s2 = s.copy()
    s2.iloc[80:] += 100.0
    assert np.allclose(base.iloc[:80], pipe.zscore(s2).iloc[:80])


# ── anomaly detection ────────────────────────────────────────────────
def test_isolation_forest_actually_flags(fleet_df):
    df = pipe.add_features(fleet_df)
    df, iso = pipe.detect_anomalies(df)
    # contamination=0.03 on 120 days must flag >0 days (old bug: always 0 from IF)
    assert df["Is_Anomaly"].sum() > 0
    # score must be continuous (decision_function), not the ±1 label
    assert df["Anomaly_Score"].nunique() > 2
    # the injected thermal window (days 30-36) should surface as anomalous
    flagged = set(df.index[df["Is_Anomaly"]])
    assert flagged & set(range(28, 40))


# ── baselines & metrics ──────────────────────────────────────────────
def test_baseline_shapes_and_mase():
    s = np.arange(100, dtype=float)
    assert pipe.naive_forecast(s, 30).shape == (30,)
    assert (pipe.naive_forecast(s, 5) == 99).all()
    sn = pipe.seasonal_naive_forecast(s, 10, m=7)
    assert sn.shape == (10,) and sn[0] == 93 and sn[7] == 93
    # naive on a linear ramp: errors 1..10 (mean 5.5), seasonal-naive scale = 7
    train, test = s[:-10], s[-10:]
    m = pipe.mase(test, pipe.naive_forecast(train, 10), train, m=7)
    assert m == pytest.approx(5.5 / 7, rel=1e-6)


def test_walk_forward_fold_count():
    s = np.random.default_rng(1).normal(size=100)
    maes = pipe.walk_forward(s, pipe.naive_forecast, horizon=7, min_train=60, step=7)
    # cuts at 60, 67, 74, 81, 88 → 5 folds (95 > 100-7 stops)
    assert len(maes) == 5
    assert all(m >= 0 for m in maes)


# ── forecasting (leakage-safe) ───────────────────────────────────────
def test_forecast_end_to_end(fleet_df):
    df = pipe.add_features(fleet_df)
    fc, val = pipe.forecast(df)
    assert len(fc) == pipe.HORIZON
    assert (np.asarray(fc) >= 0).all()
    for name in ["naive", "seasonal_naive", "ets", "arima"]:
        assert f"{name}_mae" in val and val[f"{name}_mae"] >= 0
        assert f"{name}_mase" in val
    assert "walk_forward" in val and "naive" in val["walk_forward"]
    assert 0.0 <= val["lstm_weight"] <= 1.0


# ── Monte Carlo risk ─────────────────────────────────────────────────
def test_monte_carlo_quantiles(fleet_df):
    df = pipe.add_features(fleet_df)
    fc = np.full(pipe.HORIZON, df["Failure_Rate"].tail(30).mean())
    risk = pipe.monte_carlo_risk(df, fc)
    q = risk["cum_quantiles"]
    p05, p50, p95 = map(np.asarray, (q["p05"], q["p50"], q["p95"]))
    assert (p05 <= p50).all() and (p50 <= p95).all()
    # cumulative quantile curves must be non-decreasing over time
    assert (np.diff(p50) >= 0).all()
    assert risk["var95_failures"] <= risk["cvar95_failures"]
    assert risk["expected_failures"] > 0
