"""
StorageIQ Dashboard — FastAPI backend
=====================================
Serves the dashboard UI, the precomputed analytics payload produced by
StorageIQ_Pipeline.py (outputs/dashboard_data.json), and a bring-your-own-
fleet analyzer (POST /api/analyze) that runs a light ARIMA + Gamma-Poisson
Monte Carlo on any uploaded fleet CSV.

Design note: the full pipeline (torch LSTM, walk-forward CV) runs offline;
the server needs only pandas/numpy/statsmodels — no torch — so it still
deploys on tiny free-tier instances.

Run locally:
    python StorageIQ_Pipeline.py            # generates outputs/dashboard_data.json
    cd StorageIQ_Dashboard
    uvicorn main:app --reload               # http://127.0.0.1:8000
"""

import io
import json
import os
import warnings

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    os.path.join(HERE, "dashboard_data.json"),                    # copied for deploy
    os.path.join(HERE, "..", "outputs", "dashboard_data.json"),   # local dev
]

app = FastAPI(title="StorageIQ API", version="2.0.0",
              description="Stochastic Storage VaR — real Backblaze telemetry")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def load_payload() -> dict:
    for p in CANDIDATES:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


PAYLOAD = load_payload()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    with open(os.path.join(HERE, "index.html")) as f:
        return f.read()


@app.get("/api/status")
def status():
    return {
        "status": "online",
        "data_loaded": bool(PAYLOAD),
        "meta": PAYLOAD.get("meta", {}),
        "project": "StorageIQ — Stochastic Storage VaR",
    }


@app.get("/api/data")
def data():
    """Full analytics payload (KPIs, series, forecast, anomalies, risk)."""
    if not PAYLOAD:
        raise HTTPException(503, "No analytics found. Run StorageIQ_Pipeline.py first.")
    return PAYLOAD


@app.post("/api/reload")
def reload_data():
    global PAYLOAD
    PAYLOAD = load_payload()
    return {"reloaded": bool(PAYLOAD)}


# ════════════════════════════════════════════════════════════════════
# Bring your own fleet — POST /api/analyze
# ════════════════════════════════════════════════════════════════════
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_ROWS = 4000
MIN_ROWS = 60
REQUIRED_COLS = {"Timestamp", "Drive_Count", "Failures"}
RISK_DAYS = 90
FC_DAYS = 30
N_PATHS = 4000


def _parse_fleet_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Could not parse file as CSV.")
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise HTTPException(
            422,
            f"Missing required columns: {sorted(missing)}. "
            "Schema: Timestamp (date), Drive_Count (int), Failures (int) — "
            "one row per day. See README → Bring Your Own Fleet.")
    try:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="raise")
    except Exception:
        raise HTTPException(422, "Timestamp column is not parseable as dates.")
    for c in ("Drive_Count", "Failures"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = (df.dropna(subset=["Timestamp", "Drive_Count", "Failures"])
            .sort_values("Timestamp").reset_index(drop=True))
    if (df["Drive_Count"] <= 0).any() or (df["Failures"] < 0).any():
        raise HTTPException(422, "Drive_Count must be > 0 and Failures >= 0.")
    if len(df) < MIN_ROWS:
        raise HTTPException(422, f"Need at least {MIN_ROWS} daily rows; got {len(df)}.")
    return df.tail(MAX_ROWS)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Light analyzer for any fleet: ARIMA forecast + Gamma-Poisson MC VaR.
    Nothing is stored — the upload is processed in memory and discarded."""
    from statsmodels.tsa.arima.model import ARIMA

    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // 1024 // 1024} MB limit.")
    df = _parse_fleet_csv(raw)

    rate = (df["Failures"] / df["Drive_Count"]).astype(float)
    rate7 = rate.rolling(7, min_periods=1).mean()

    # ARIMA on the smoothed rate — small AIC grid keeps latency low
    series = rate7.values
    best_fit, best_aic = None, np.inf
    for order in [(0, 1, 0), (0, 1, 1), (1, 1, 0), (1, 1, 1)]:
        try:
            fit = ARIMA(series, order=order).fit()
            if fit.aic < best_aic:
                best_fit, best_aic = fit, fit.aic
        except Exception:
            continue
    if best_fit is None:
        raise HTTPException(500, "Forecast failed on this series.")
    fc = np.clip(np.asarray(best_fit.forecast(FC_DAYS)), 1e-12, None)

    # Gamma-Poisson Monte Carlo (same model as the main pipeline)
    n_drives = float(df["Drive_Count"].iloc[-1])
    recent = rate.tail(90)
    mean_rate = max(recent.mean(), 1e-9)
    disp = max(recent.var() / mean_rate, 1e-12)
    shape = mean_rate / disp
    drift = np.concatenate([fc, np.full(RISK_DAYS - FC_DAYS, fc[-1])])
    rng = np.random.default_rng(42)
    lam = rng.gamma(shape, drift / shape, (N_PATHS, RISK_DAYS))
    total = rng.poisson(lam * n_drives).sum(axis=1)
    var95 = float(np.percentile(total, 95))

    last = df["Timestamp"].iloc[-1]
    return {
        "kpis": {
            "days": len(df),
            "drives": int(n_drives),
            "failures_total": int(df["Failures"].sum()),
            "afr_pct": float(rate7.iloc[-1] * 365 * 100),
        },
        "series": {
            "dates": df["Timestamp"].dt.strftime("%Y-%m-%d").tolist(),
            "afr_7d": (rate7 * 365 * 100).round(4).tolist(),
        },
        "forecast": {
            "dates": [(last + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
                      for i in range(FC_DAYS)],
            "afr_pct": (fc * 365 * 100).round(4).tolist(),
            "arima_order": list(best_fit.model.order),
        },
        "risk": {
            "horizon_days": RISK_DAYS,
            "expected_failures": float(total.mean()),
            "var95_failures": var95,
            "cvar95_failures": float(total[total >= var95].mean()),
        },
    }
