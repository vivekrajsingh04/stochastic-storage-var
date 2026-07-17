# Stochastic Storage VaR Engine ⚡

[![CI](https://github.com/vivekrajsingh04/stochastic-storage-var/actions/workflows/ci.yml/badge.svg)](https://github.com/vivekrajsingh04/stochastic-storage-var/actions/workflows/ci.yml)

**Quantifying storage-fleet failure risk with real-world telemetry.** An end-to-end pipeline that models hard-drive degradation as a stochastic process — built on **12 months of [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)** (Apr 2024 – Mar 2025: **107.9M drive-days** across ~312,000 production drives and 88 drive models) — and computes VaR₉₅ / CVaR₉₅ failure bounds via large-scale Monte Carlo simulation.

> **Why it matters:** capacity planners and SRE teams don't need a point forecast of failures — they need the *tail*. "How many drive replacements should we budget for next quarter with 95% confidence?" is a Value-at-Risk question, and this engine answers it with the same math used on trading desks.

## 🧮 Architecture & Math

| Component | Model | Implementation |
| :--- | :--- | :--- |
| **Forecasting** | LSTM + ARIMA (AIC-selected order) inverse-MAE weighted ensemble | PyTorch, `statsmodels` |
| **Baselines** | naive, seasonal-naive, damped ETS — every learned model must beat them | `statsmodels` |
| **Validation** | 30-day hold-out (MAE + MASE) **and** rolling-origin walk-forward CV | custom |
| **Anomaly detection** | Isolation Forest + 3σ z-score rules | `scikit-learn`, `scipy` |
| **Risk quantification** | Gamma-Poisson (over-dispersed) Monte Carlo, 10,000 paths | `numpy` |
| **Risk metrics** | VaR₉₅, CVaR₉₅ of cumulative failures over 90 days | custom |
| **Stationarity** | Augmented Dickey-Fuller | `statsmodels` |

The Monte Carlo engine models daily fleet failures as `Poisson(λ_t · N_t)` where the rate `λ_t` itself is Gamma-distributed (capturing over-dispersion measured in the data) and drifts along the ensemble forecast path.

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# 1. Fetch real telemetry (one Backblaze quarter ≈ 1 GB zipped)
python data/download_backblaze.py --quarter Q1_2025
#    …or several quarters as one continuous series:
# python data/download_backblaze.py --quarter Q2_2024,Q3_2024,Q4_2024,Q1_2025

# 2. Run the full pipeline: features → anomalies → forecast → Monte Carlo VaR
python StorageIQ_Pipeline.py

# 3. Launch the dashboard
cd StorageIQ_Dashboard && uvicorn main:app --reload
# → http://127.0.0.1:8000
```

PyTorch is optional — without it the pipeline falls back to ARIMA-only forecasting, which keeps free-tier deploys light.

## 📊 Dashboard

Dark-mode analytics dashboard (FastAPI + Chart.js): fleet AFR history with 30-day forecast overlay, daily failures vs. fleet size, the Monte Carlo fan chart with VaR₉₅/CVaR₉₅, flagged anomaly days, and a per-drive-model AFR league table.

The dashboard reads a single precomputed `dashboard_data.json` — no ML dependencies at serve time, so it deploys on Render's free tier as-is (`render.yaml` included).

## 📂 Repository Structure

```
stochastic-storage-var/
├── data/
│   └── download_backblaze.py     # Downloads + aggregates real Backblaze Drive Stats
├── StorageIQ_Pipeline.py         # Core engine: features → anomalies → forecast → MC VaR
├── StorageIQ_Dashboard/
│   ├── main.py                   # FastAPI backend (serves precomputed analytics)
│   └── index.html                # Chart.js dashboard
├── tests/
│   ├── generate_fixture.py       # Schema-accurate mini-fixture for CI / smoke tests
│   ├── conftest.py               # Session fixtures: raw CSVs → aggregated fleet tables
│   └── test_pipeline.py          # Loader, causality, anomaly, baseline, MC tests
├── .github/workflows/ci.yml      # pytest + full fixture→pipeline e2e smoke on every push
├── requirements.txt
└── render.yaml                   # One-click Render deploy config
```

## 🔌 Bring Your Own Fleet

**The engine is schema-generic — Backblaze is the validation corpus, not a dependency.** Any fleet that emits daily unit-counts and failures runs through the same pipeline unchanged: server fleets, IoT devices, vehicle fleets, battery packs.

Three ways in:

1. **Dashboard upload** — POST a CSV to `/api/analyze` (or drag-drop on the dashboard's *Analyze your fleet* section) and get an ARIMA forecast + Monte Carlo VaR₉₅/CVaR₉₅ back in seconds. Processed in memory, nothing stored.
2. **Loader** — point the aggregator at any folder of per-unit daily CSVs: `python data/download_backblaze.py --raw-dir /path/to/your/telemetry`
3. **Direct** — write `data/fleet_daily.csv` yourself and run the pipeline.

Minimal schema (one row per day):

| Column | Type | Required |
| :--- | :--- | :--- |
| `Timestamp` | date | ✅ |
| `Drive_Count` | int — units alive that day | ✅ |
| `Failures` | int — units that failed that day | ✅ |
| `Capacity_PB`, `Mean_<metric>` | float — any health telemetry | optional (powers anomaly typing) |

The models never reference anything storage-specific: the forecast layer sees a rate series, the risk layer sees a count process. Auto-refresh (`.github/workflows/refresh-data.yml`) keeps the deployed Backblaze instance current — when a new quarter publishes, it re-ingests, retrains, and redeploys itself monthly with zero manual steps.

## 🔬 Anomaly Taxonomy

The detector classifies flagged days into operational categories:

* 🔴 **FAILURE_SPIKE** — daily failures > 3σ above rolling baseline
* 🟠 **DEGRADATION_WAVE** — fleet-wide surge in reallocated/pending sectors (SMART 5/197)
* 🟡 **THERMAL_EVENT** — abnormal fleet temperature excursion (SMART 194)
* 🟣 **STATISTICAL_OUTLIER** — multivariate outlier per Isolation Forest

## 📈 Validation — leakage-free by construction

* **Baselines first.** Naive, seasonal-naive and damped-ETS forecasts are evaluated alongside ARIMA and the LSTM. MASE is reported for every model (MASE < 1 beats a seasonal-naive walk). When a baseline wins, the dashboard says so.
* **Two views of error.** A 30-day hold-out *and* rolling-origin walk-forward CV (expanding window, MAE mean ± std across folds) — a single split can flatter any model; the walk-forward can't.
* **No leakage.** The LSTM and its scaler are fit on the training window only; the test window is never seen before scoring. Rolling features are causal (verified by test). ARIMA order is selected by AIC on the training window and frozen before evaluation.
* **Tested in CI.** `pytest` covers loader schema, feature causality, anomaly flagging, baseline math, walk-forward fold logic and Monte Carlo quantile coherence; a full fixture→loader→pipeline e2e smoke runs on every push.

## ⚠️ Known Limitations

Fleet-level aggregation discards per-drive SMART signal (a per-drive survival model is the natural next step); failure clustering across days is not modeled by the independent-increment Monte Carlo; anomaly labels are unsupervised — there is no ground truth on real telemetry, so anomaly precision/recall is unknowable by design.

## 📜 Data License

Backblaze Drive Stats is free to use with attribution — data © Backblaze, published quarterly since 2013.
