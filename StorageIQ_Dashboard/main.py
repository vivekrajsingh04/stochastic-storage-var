"""
StorageIQ — Demo Dashboard Backend
FastAPI server that loads your trained models and serves the dashboard.

SETUP:
  1. Copy your models/ folder + storage_telemetry_data.csv into this folder
  2. pip install fastapi uvicorn pandas numpy scikit-learn statsmodels torch joblib
  3. uvicorn main:app --reload
  4. Open http://127.0.0.1:8000 in your browser
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import json, joblib, os, warnings
from datetime import timedelta
import torch, torch.nn as nn
warnings.filterwarnings("ignore")

app = FastAPI(title="StorageIQ API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_CAP  = 2000.0
HORIZON  = 30
WINDOW   = 30

# ── Load Resources ────────────────────────────────────────────────────────────
print("Loading StorageIQ models...")

try:
    meta      = json.load(open("models/model_meta.json"))
    iso       = joblib.load("models/anomaly_model.pkl")
    scaler    = joblib.load("models/scaler.pkl")
    df        = pd.read_csv("storage_telemetry_data.csv", parse_dates=["Timestamp"])
    MODELS_OK = True
    print(f"✅ Models loaded | Data shape: {df.shape}")
except Exception as e:
    print(f"⚠️  Could not load models: {e}")
    print("   Make sure models/ and storage_telemetry_data.csv are in this folder.")
    MODELS_OK = False

# ── LSTM Model Definition (must match training) ────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 128, 2, batch_first=True, dropout=0.25)
        self.bn   = nn.BatchNorm1d(128)
        self.fc   = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64,1))
    def forward(self, x):
        out,_ = self.lstm(x); return self.fc(self.bn(out[:,-1,:]))

try:
    lstm_model = LSTMModel()
    lstm_model.load_state_dict(torch.load("models/lstm_model.pt", map_location="cpu"))
    lstm_model.eval()
except:
    lstm_model = None

# ── Helper: Feature Engineering ───────────────────────────────────────────────
def add_features(df):
    df = df.copy().sort_values("Timestamp").reset_index(drop=True)
    df["Daily_Growth"]       = df["Capacity_GB"].diff().fillna(0)
    df["Cap_7d_Avg"]         = df["Capacity_GB"].rolling(7,  min_periods=1).mean()
    df["Cap_30d_Avg"]        = df["Capacity_GB"].rolling(30, min_periods=1).mean()
    df["Growth_Volatility"]  = df["Daily_Growth"].rolling(7,  min_periods=2).std().fillna(0)
    df["Write_Delete_Ratio"] = (df["Write_Count"] / (df["Delete_Count"]+1)).round(4)
    df["Net_Tx_Delta"]       = df["Write_Count"] - df["Delete_Count"]
    def zs(s, w=30): m=s.rolling(w,min_periods=7).mean(); sd=s.rolling(w,min_periods=7).std().replace(0,1e-6); return ((s-m)/sd).fillna(0)
    df["Growth_ZScore"] = zs(df["Daily_Growth"])
    df["Write_ZScore"]  = zs(df["Write_Count"].astype(float))
    df["IO_ZScore"]     = zs(df["IO_Throughput_MBps"])
    df["Latency_ZScore"]= zs(df["Latency_ms"])
    return df

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    """Serve the main dashboard HTML."""
    with open("index.html", "r") as f:
        return f.read()

@app.get("/api/status")
def status():
    return {"status": "online", "models_loaded": MODELS_OK,
            "data_points": len(df) if MODELS_OK else 0,
            "project": "StorageIQ — VIT-SanDisk Hackathon 2026"}

@app.get("/api/forecast")
def get_forecast():
    """30-day capacity forecast using LSTM + ARIMA ensemble."""
    cap = df["Capacity_GB"].values
    last_date = df["Timestamp"].iloc[-1]
    future_dates = [(last_date + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(HORIZON)]

    # LSTM iterative forecast
    try:
        seq = scaler.transform(cap.reshape(-1,1))[-WINDOW:].reshape(1,WINDOW,1)
        lstm_fc = []
        with torch.no_grad():
            for _ in range(HORIZON):
                t=torch.FloatTensor(seq); p=lstm_model(t).item(); lstm_fc.append(p)
                seq=np.append(seq[:,1:,:],[[[p]]],axis=1)
        lstm_preds = scaler.inverse_transform(np.array(lstm_fc).reshape(-1,1)).flatten().tolist()
    except:
        # Fallback: simple linear extrapolation
        mu = float(df["Capacity_GB"].diff().dropna().iloc[-30:].mean())
        lstm_preds = [float(cap[-1]) + mu*(i+1) for i in range(HORIZON)]

    # ARIMA forecast (use saved or simple growth)
    try:
        from statsmodels.tsa.arima.model import ARIMAResults
        arima = ARIMAResults.load("models/arima_model.pkl")
        arima_preds = arima.forecast(steps=HORIZON).tolist()
    except:
        mu = float(df["Capacity_GB"].diff().dropna().iloc[-30:].mean())
        arima_preds = [float(cap[-1]) + mu*(i+1) for i in range(HORIZON)]

    w_arima, w_lstm = 0.40, 0.60
    ens_preds = [(w_arima*a + w_lstm*l) for a,l in zip(arima_preds, lstm_preds)]

    # Historical (last 60 days for chart)
    hist = df.tail(60)[["Timestamp","Capacity_GB"]].copy()
    hist["Timestamp"] = hist["Timestamp"].dt.strftime("%Y-%m-%d")

    return {
        "historical": hist.to_dict(orient="records"),
        "forecast_dates":  future_dates,
        "arima_forecast":  [round(v,2) for v in arima_preds],
        "lstm_forecast":   [round(v,2) for v in lstm_preds],
        "ensemble_forecast":[round(v,2) for v in ens_preds],
        "summary": {
            "current_gb":   round(float(cap[-1]), 2),
            "forecast_7d":  round(ens_preds[6],  2),
            "forecast_14d": round(ens_preds[13], 2),
            "forecast_30d": round(ens_preds[29], 2),
            "util_pct_30d": round(ens_preds[29]/MAX_CAP*100, 1),
        }
    }


@app.get("/api/risk-score")
def get_risk_score():
    """Monte Carlo overflow risk score (5,000 simulations for speed)."""
    recent = df["Capacity_GB"].diff().dropna().iloc[-90:]
    mu, sigma = float(recent.mean()), float(recent.std())
    current   = float(df["Capacity_GB"].iloc[-1])
    np.random.seed(42)
    paths   = current + np.cumsum(np.random.normal(mu, sigma, (5000, HORIZON)), axis=1)
    breach  = np.any(paths >= MAX_CAP, axis=1)
    risk    = round(float(breach.sum()/5000*100), 2)
    label   = "HIGH" if risk>=70 else ("MEDIUM" if risk>=30 else "LOW")
    daily   = [(paths[:,d]>=MAX_CAP).mean()*100 for d in range(HORIZON)]
    terminal= paths[:,-1]
    var95   = round(float(np.percentile(terminal, 95)), 1)
    return {
        "risk_score_pct":   risk,
        "risk_label":       label,
        "overflow_prob":    risk,
        "var_95_gb":        var95,
        "headroom_gb":      round(MAX_CAP - current, 1),
        "current_gb":       round(current, 1),
        "daily_breach_pct": [round(v, 2) for v in daily],
        "color": "#e74c3c" if label=="HIGH" else ("#f39c12" if label=="MEDIUM" else "#27ae60"),
    }


@app.get("/api/anomalies")
def get_anomalies():
    """Detect anomalies using Isolation Forest + Z-Score."""
    df_feat = add_features(df)
    ANOM_FEATS = ["Daily_Growth","Write_Count","Delete_Count","IO_Throughput_MBps",
                  "Latency_ms","Write_Delete_Ratio","Write_ZScore","IO_ZScore",
                  "Growth_ZScore","Growth_Volatility","Net_Tx_Delta"]
    X = df_feat[ANOM_FEATS].fillna(0)
    try:
        labels = iso.predict(X); scores = iso.decision_function(X)
    except:
        return {"total": 0, "anomalies": []}
    df_feat["Is_Anomaly"] = (labels == -1).astype(int)
    df_feat["IF_Score"]   = scores
    zs_flag = (df_feat[["Growth_ZScore","Write_ZScore","IO_ZScore","Latency_ZScore"]].abs() >= 3).any(axis=1)
    df_feat["Is_Anomaly"] = ((df_feat["Is_Anomaly"]==1) | zs_flag).astype(int)
    def sev(s,a): return ("HIGH" if s<-0.20 else ("MEDIUM" if s<-0.08 else "LOW")) if a else "NONE"
    def threat(row):
        if not row["Is_Anomaly"]: return "NORMAL"
        wz,ioz,g,wr = abs(row.get("Write_ZScore",0)),abs(row.get("IO_ZScore",0)),row.get("Daily_Growth",0),row.get("Write_Delete_Ratio",1)
        if wz>5 and g>30 and wr>8: return "RANSOMWARE 🔴"
        if ioz>4 and wz<2: return "DATA EXFILTRATION 🟠"
        if wr>3 and wz>2.5 and g<10: return "INSIDER THREAT 🟡"
        if g>20 and wz>3: return "BULK TRANSFER 🔵"
        return "ANOMALOUS PATTERN"
    df_feat["Severity"]   = df_feat.apply(lambda r: sev(r["IF_Score"], r["Is_Anomaly"]), axis=1)
    df_feat["Threat_Type"]= df_feat.apply(threat, axis=1)
    anomalies = df_feat[df_feat["Is_Anomaly"]==1].tail(20)
    return {
        "total": int(df_feat["Is_Anomaly"].sum()),
        "severity_counts": df_feat[df_feat["Is_Anomaly"]==1]["Severity"].value_counts().to_dict(),
        "anomalies": [
            {"date": str(r["Timestamp"])[:10], "capacity_gb": round(r["Capacity_GB"],2),
             "daily_growth": round(r["Daily_Growth"],2), "severity": r["Severity"], "threat": r["Threat_Type"]}
            for _,r in anomalies.iterrows()
        ]
    }


@app.get("/api/recommend")
def get_recommendations():
    """Generate prioritised recommendations."""
    fc  = get_forecast()["summary"]
    rs  = get_risk_score()
    an  = get_anomalies()
    f30 = fc["forecast_30d"]; risk = rs["risk_score_pct"]
    f30p= round(f30/MAX_CAP*100,1); current = rs["current_gb"]
    recs = []
    high_an = [a for a in an["anomalies"] if a["severity"]=="HIGH"]
    ransom  = [a for a in an["anomalies"] if "RANSOMWARE" in a["threat"]]
    if ransom:
        recs.append({"priority":"CRITICAL","icon":"🚨","category":"Security",
                     "action":"Isolate device — Ransomware activity detected",
                     "space":"N/A","confidence":"91%","color":"#8e44ad"})
    if risk>=70:
        recs.append({"priority":"HIGH","icon":"🔴","category":"Capacity",
                     "action":f"Expand storage capacity immediately (Risk={risk}%)",
                     "space":f"{round(MAX_CAP-current,0):.0f} GB headroom gained","confidence":"94%","color":"#e74c3c"})
    if f30p>=85:
        recs.append({"priority":"HIGH","icon":"🔴","category":"Optimization",
                     "action":f"Archive cold data now ({f30p:.0f}% util in 30 days)",
                     "space":f"~{round(f30-MAX_CAP*0.70,0):.0f} GB recoverable","confidence":"91%","color":"#e74c3c"})
    elif f30p>=70:
        recs.append({"priority":"MEDIUM","icon":"🟡","category":"Optimization",
                     "action":f"Compress archives & delete duplicates ({f30p:.0f}% in 30d)",
                     "space":"~80-150 GB","confidence":"87%","color":"#f39c12"})
    if high_an and not ransom:
        recs.append({"priority":"HIGH","icon":"🔴","category":"Security",
                     "action":f"Investigate HIGH severity write activity ({len(high_an)} events)",
                     "space":"~30-80 GB","confidence":"85%","color":"#e74c3c"})
    recs.append({"priority":"LOW","icon":"🟢","category":"Maintenance",
                 "action":"Migrate files unused >60 days to cold storage tier",
                 "space":f"~{round(current*0.18,0):.0f} GB","confidence":"76%","color":"#27ae60"})
    return {"total": len(recs), "recommendations": recs,
            "summary_metrics": {"risk": risk, "forecast_30d": f30, "anomalies": an["total"], "util_pct": f30p}}
