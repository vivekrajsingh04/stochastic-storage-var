"""
StorageIQ — Stochastic Storage VaR Engine (real-data edition)
=============================================================
End-to-end pipeline over REAL drive telemetry (Backblaze Drive Stats):

  1. Load fleet-level daily aggregates   (data/fleet_daily.csv)
  2. Feature engineering                 (rolling stats, volatility, z-scores)
  3. Anomaly detection                   (Isolation Forest + 3σ rules)
  4. Stationarity testing                (Augmented Dickey-Fuller)
  5. Forecasting                         (LSTM + ARIMA ensemble; ARIMA-only
                                          fallback when PyTorch is absent)
  6. Monte Carlo risk                    (Gamma-Poisson failure paths →
                                          VaR95 / CVaR95 of cumulative failures)
  7. Export                              (models/ + outputs/dashboard_data.json)

First fetch real data:
  python data/download_backblaze.py --quarter Q1_2025
Then run:
  python StorageIQ_Pipeline.py
"""

import json
import os
import warnings
from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

warnings.filterwarnings("ignore")
np.random.seed(42)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "data", "fleet_daily.csv")
MODEL_CSV = os.path.join(HERE, "data", "model_stats.csv")
MODELS_DIR = os.path.join(HERE, "models")
OUT_DIR = os.path.join(HERE, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON = 30        # forecast horizon (days)
RISK_HORIZON = 90   # Monte Carlo projection (days)
N_PATHS = 10_000    # Monte Carlo paths
WINDOW = 30         # LSTM lookback
TARGET = "Failure_Rate_7d"  # smoothed target series


# ════════════════════════════════════════════════════════════════════
# 1. LOAD REAL DATA
# ════════════════════════════════════════════════════════════════════
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_CSV):
        raise SystemExit(
            f"Missing {DATA_CSV}.\n"
            "Fetch real telemetry first:\n"
            "  python data/download_backblaze.py --quarter Q1_2025"
        )
    df = pd.read_csv(DATA_CSV, parse_dates=["Timestamp"]).sort_values("Timestamp")
    df = df.reset_index(drop=True)
    print(f"✔ Loaded {len(df)} days of fleet telemetry "
          f"({df['Timestamp'].min():%Y-%m-%d} → {df['Timestamp'].max():%Y-%m-%d}, "
          f"~{df['Drive_Count'].iloc[-1]:,.0f} drives)")
    return df


# ════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ════════════════════════════════════════════════════════════════════
def zscore(s: pd.Series, w: int = 30) -> pd.Series:
    m = s.rolling(w, min_periods=7).mean()
    sd = s.rolling(w, min_periods=7).std().replace(0, 1e-9)
    return ((s - m) / sd).fillna(0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Failure_Rate_7d"] = df["Failure_Rate"].rolling(7, min_periods=1).mean()
    df["AFR_7d"] = df["Failure_Rate_7d"] * 365 * 100
    df["Fail_Volatility"] = df["Failures"].rolling(7, min_periods=2).std().fillna(0)
    df["Drive_Growth"] = df["Drive_Count"].diff().fillna(0)

    zcols = ["Failures", "Fail_Volatility"]
    for c in df.columns:
        if c.startswith("Mean_"):
            zcols.append(c)
    for c in zcols:
        df[f"Z_{c}"] = zscore(df[c].astype(float))
    return df


# ════════════════════════════════════════════════════════════════════
# 3. ANOMALY DETECTION
# ════════════════════════════════════════════════════════════════════
def detect_anomalies(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c.startswith("Z_")]
    X = df[feat_cols].values
    iso = IsolationForest(n_estimators=200, contamination=0.03, random_state=42)
    df["Anomaly_Score"] = -iso.fit_predict(X)  # 1 anomalous ↔ -1 normal → 2/0
    df["Is_Anomaly"] = (df["Anomaly_Score"] > 1) | (df["Z_Failures"].abs() > 3)

    def classify(r):
        if not r["Is_Anomaly"]:
            return "normal"
        if r["Z_Failures"] > 3:
            return "FAILURE_SPIKE"
        if r.get("Z_Mean_Pending_Sectors", 0) > 3 or r.get("Z_Mean_Realloc_Sectors", 0) > 3:
            return "DEGRADATION_WAVE"
        if r.get("Z_Mean_Temperature_C", 0) > 3:
            return "THERMAL_EVENT"
        return "STATISTICAL_OUTLIER"

    df["Anomaly_Type"] = df.apply(classify, axis=1)
    n = int(df["Is_Anomaly"].sum())
    print(f"✔ Anomaly detection: {n} anomalous days "
          f"({df.loc[df['Is_Anomaly'], 'Anomaly_Type'].value_counts().to_dict()})")
    return df, iso


# ════════════════════════════════════════════════════════════════════
# 4. STATIONARITY (ADF)
# ════════════════════════════════════════════════════════════════════
def adf_report(series: pd.Series) -> dict:
    r = adfuller(series.dropna())
    out = {"adf_stat": float(r[0]), "p_value": float(r[1]),
           "stationary": bool(r[1] < 0.05)}
    print(f"✔ ADF test on {series.name}: p={out['p_value']:.4f} "
          f"({'stationary' if out['stationary'] else 'non-stationary'})")
    return out


# ════════════════════════════════════════════════════════════════════
# 5. FORECASTING — LSTM + ARIMA ensemble
# ════════════════════════════════════════════════════════════════════
if HAS_TORCH:
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])


def train_lstm(series: np.ndarray, scaler: MinMaxScaler, epochs: int = 60):
    if not HAS_TORCH or len(series) < WINDOW + 10:
        return None
    scaled = scaler.transform(series.reshape(-1, 1)).flatten()
    X = np.stack([scaled[i:i + WINDOW] for i in range(len(scaled) - WINDOW)])
    y = scaled[WINDOW:]
    ds = TensorDataset(torch.FloatTensor(X).unsqueeze(-1), torch.FloatTensor(y).unsqueeze(-1))
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = LSTMModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    return model


def lstm_forecast(model, series, scaler, horizon=HORIZON):
    seq = scaler.transform(series.reshape(-1, 1))[-WINDOW:].reshape(1, WINDOW, 1)
    preds = []
    with torch.no_grad():
        for _ in range(horizon):
            p = model(torch.FloatTensor(seq)).item()
            preds.append(p)
            seq = np.append(seq[:, 1:, :], [[[p]]], axis=1)
    return scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()


def arima_forecast(series: np.ndarray, horizon=HORIZON):
    model = ARIMA(series, order=(2, 1, 2)).fit()
    return model, np.asarray(model.forecast(horizon))


def forecast(df: pd.DataFrame):
    series = df[TARGET].values.astype(float)
    scaler = MinMaxScaler().fit(series.reshape(-1, 1))

    # Hold-out validation on last HORIZON days
    train, test = series[:-HORIZON], series[-HORIZON:]
    _, arima_val = arima_forecast(train)
    val = {"arima_mae": float(mean_absolute_error(test, arima_val)),
           "arima_rmse": float(np.sqrt(mean_squared_error(test, arima_val)))}

    lstm_model = train_lstm(series, scaler)
    if lstm_model is not None:
        lstm_val = lstm_forecast(lstm_model, train, scaler)
        val["lstm_mae"] = float(mean_absolute_error(test, lstm_val))
        # inverse-MAE weighting
        w_l = 1 / max(val["lstm_mae"], 1e-12)
        w_a = 1 / max(val["arima_mae"], 1e-12)
        w_lstm = w_l / (w_l + w_a)
    else:
        w_lstm = 0.0
    val["lstm_weight"] = round(w_lstm, 3)

    arima_full, arima_fc = arima_forecast(series)
    if lstm_model is not None:
        lstm_fc = lstm_forecast(lstm_model, series, scaler)
        fc = w_lstm * lstm_fc + (1 - w_lstm) * arima_fc
    else:
        fc = arima_fc
    fc = np.clip(fc, 0, None)

    mode = f"LSTM+ARIMA (w_lstm={w_lstm:.2f})" if lstm_model is not None else "ARIMA only"
    print(f"✔ Forecast ({mode}): ARIMA MAE={val['arima_mae']:.3e}"
          + (f", LSTM MAE={val['lstm_mae']:.3e}" if "lstm_mae" in val else ""))

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(arima_full, os.path.join(MODELS_DIR, "arima_model.pkl"))
    if lstm_model is not None:
        torch.save(lstm_model.state_dict(), os.path.join(MODELS_DIR, "lstm_model.pt"))
    return fc, val


# ════════════════════════════════════════════════════════════════════
# 6. MONTE CARLO RISK — Gamma-Poisson failure paths
# ════════════════════════════════════════════════════════════════════
def monte_carlo_risk(df: pd.DataFrame, fc_rate: np.ndarray):
    """
    Daily failures ~ Poisson(λ_t · N_t) with Gamma-distributed λ uncertainty
    (over-dispersion), λ_t drifting along the forecast path.
    Returns VaR95/CVaR95 of cumulative failures over RISK_HORIZON days.
    """
    n_drives = df["Drive_Count"].iloc[-1]
    recent = df["Failure_Rate"].tail(90)
    mean_rate = max(recent.mean(), 1e-9)
    disp = max(recent.var() / mean_rate, 1e-12)  # over-dispersion index

    # extend forecast rate path to RISK_HORIZON
    drift = np.concatenate([fc_rate, np.full(RISK_HORIZON - len(fc_rate), fc_rate[-1])]) \
        if len(fc_rate) < RISK_HORIZON else fc_rate[:RISK_HORIZON]
    drift = np.clip(drift, 1e-9, None)

    shape = mean_rate / max(disp, 1e-12)
    paths = np.empty((N_PATHS, RISK_HORIZON))
    for t in range(RISK_HORIZON):
        lam = np.random.gamma(shape, drift[t] / shape, N_PATHS)  # rate uncertainty
        paths[:, t] = np.random.poisson(lam * n_drives)
    cum = paths.cumsum(axis=1)
    total = cum[:, -1]

    var95 = float(np.percentile(total, 95))
    cvar95 = float(total[total >= var95].mean())
    out = {
        "horizon_days": RISK_HORIZON,
        "n_paths": N_PATHS,
        "expected_failures": float(total.mean()),
        "var95_failures": var95,
        "cvar95_failures": cvar95,
        "afr_now_pct": float(df["AFR_7d"].iloc[-1]),
        "afr_forecast_pct": float(drift[-1] * 365 * 100),
        "cum_quantiles": {
            "p05": np.percentile(cum, 5, axis=0).tolist(),
            "p50": np.percentile(cum, 50, axis=0).tolist(),
            "p95": np.percentile(cum, 95, axis=0).tolist(),
        },
    }
    print(f"✔ Monte Carlo ({N_PATHS:,} paths, {RISK_HORIZON}d): "
          f"E[failures]={out['expected_failures']:.0f}, "
          f"VaR95={var95:.0f}, CVaR95={cvar95:.0f}")
    return out


# ════════════════════════════════════════════════════════════════════
# 7. EXPORT — everything the dashboard needs, in one JSON
# ════════════════════════════════════════════════════════════════════
def export(df, fc, val, risk, adf, iso):
    last_date = df["Timestamp"].iloc[-1]
    fc_dates = [(last_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                for i in range(len(fc))]
    anomalies = df[df["Is_Anomaly"]]
    payload = {
        "meta": {
            "generated": pd.Timestamp.now().isoformat(timespec="seconds"),
            "source": "Backblaze Drive Stats (real telemetry)",
            "days": len(df),
            "start": f"{df['Timestamp'].iloc[0]:%Y-%m-%d}",
            "end": f"{last_date:%Y-%m-%d}",
        },
        "kpis": {
            "drives": int(df["Drive_Count"].iloc[-1]),
            "capacity_pb": float(df["Capacity_PB"].iloc[-1]),
            "failures_total": int(df["Failures"].sum()),
            "afr_pct": float(df["AFR_7d"].iloc[-1]),
            "anomaly_days": int(df["Is_Anomaly"].sum()),
        },
        "series": {
            "dates": df["Timestamp"].dt.strftime("%Y-%m-%d").tolist(),
            "failures": df["Failures"].astype(int).tolist(),
            "afr_7d": df["AFR_7d"].round(4).tolist(),
            "drive_count": df["Drive_Count"].astype(int).tolist(),
            "temperature": df.get("Mean_Temperature_C", pd.Series(dtype=float)).round(2).tolist(),
        },
        "forecast": {
            "dates": fc_dates,
            "afr_pct": (np.asarray(fc) * 365 * 100).round(4).tolist(),
            "validation": val,
        },
        "anomalies": [
            {"date": f"{r.Timestamp:%Y-%m-%d}", "type": r.Anomaly_Type,
             "failures": int(r.Failures), "z": round(float(r.Z_Failures), 2)}
            for r in anomalies.itertuples()
        ],
        "risk": risk,
        "adf": adf,
    }
    if os.path.exists(MODEL_CSV):
        top = pd.read_csv(MODEL_CSV).head(15)
        payload["drive_models"] = top.round(3).to_dict(orient="records")

    with open(os.path.join(OUT_DIR, "dashboard_data.json"), "w") as f:
        json.dump(payload, f)
    joblib.dump(iso, os.path.join(MODELS_DIR, "anomaly_model.pkl"))
    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump({"target": TARGET, "window": WINDOW, "horizon": HORIZON,
                   "torch": HAS_TORCH, "validation": val}, f, indent=2)
    print(f"✔ Exported outputs/dashboard_data.json + models/")


def main():
    print("█" * 60 + "\n  StorageIQ — Stochastic Storage VaR (real data)\n" + "█" * 60)
    df = load_data()
    df = add_features(df)
    df, iso = detect_anomalies(df)
    adf = adf_report(df[TARGET])
    fc, val = forecast(df)
    risk = monte_carlo_risk(df, np.asarray(fc))
    export(df, fc, val, risk, adf, iso)
    print("\n✅ Pipeline complete. Launch the dashboard:\n"
          "   cd StorageIQ_Dashboard && uvicorn main:app --reload")


if __name__ == "__main__":
    main()
