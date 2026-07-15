"""
StorageIQ Dashboard — FastAPI backend
=====================================
Serves the dashboard UI and the precomputed analytics payload produced by
StorageIQ_Pipeline.py (outputs/dashboard_data.json).

Design note: the pipeline does the heavy lifting (torch/statsmodels) offline;
the dashboard only reads JSON, so it deploys on tiny free-tier instances with
no ML dependencies.

Run locally:
    python StorageIQ_Pipeline.py            # generates outputs/dashboard_data.json
    cd StorageIQ_Dashboard
    uvicorn main:app --reload               # http://127.0.0.1:8000
"""

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
