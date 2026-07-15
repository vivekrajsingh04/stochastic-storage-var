# Stochastic Storage VaR Engine ⚡

**Quantifying storage-fleet failure risk with real-world telemetry.** An end-to-end pipeline that models hard-drive degradation as a stochastic process — trained on **[Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)**, the public dataset covering 250,000+ production drives — and computes VaR₉₅ / CVaR₉₅ failure bounds via large-scale Monte Carlo simulation.

> **Why it matters:** capacity planners and SRE teams don't need a point forecast of failures — they need the *tail*. "How many drive replacements should we budget for next quarter with 95% confidence?" is a Value-at-Risk question, and this engine answers it with the same math used on trading desks.

## 🧮 Architecture & Math

| Component | Model | Implementation |
| :--- | :--- | :--- |
| **Forecasting** | LSTM + ARIMA(2,1,2) inverse-MAE weighted ensemble | PyTorch, `statsmodels` |
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
│   └── generate_fixture.py       # Schema-accurate mini-fixture for CI / smoke tests
├── requirements.txt
└── render.yaml                   # One-click Render deploy config
```

## 🔬 Anomaly Taxonomy

The detector classifies flagged days into operational categories:

* 🔴 **FAILURE_SPIKE** — daily failures > 3σ above rolling baseline
* 🟠 **DEGRADATION_WAVE** — fleet-wide surge in reallocated/pending sectors (SMART 5/197)
* 🟡 **THERMAL_EVENT** — abnormal fleet temperature excursion (SMART 194)
* 🟣 **STATISTICAL_OUTLIER** — multivariate outlier per Isolation Forest

## 📈 Validation

Forecasts are validated on a 30-day hold-out (MAE/RMSE reported in `models/model_meta.json` and on the dashboard). Ensemble weights are set by inverse validation MAE, so the better model on *your* data window automatically dominates.

## 📜 Data License

Backblaze Drive Stats is free to use with attribution — data © Backblaze, published quarterly since 2013.
