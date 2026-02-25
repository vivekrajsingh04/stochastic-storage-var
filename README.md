# Stochastic Storage VaR (Value at Risk) Engine

A highly optimized ML orchestration framework for predictive storage telemetry, anomaly detection, and risk quantification using stochastic modeling. This engine calculates hardware failure probabilities and operational risks (VaR, CVaR) using large-scale Monte Carlo simulations.

## ⚙️ Core Architecture & Math

This repository implements a complete end-to-end pipeline modeling storage system degradation as a stochastic process.

| Component                   | Architecture / Mathematical Model                    | Implementation                |
| :-------------------------- | :--------------------------------------------------- | :---------------------------- |
| **Time Series Forecasting** | LSTM (Long Short-Term Memory) + ARIMA                | PyTorch, `statsmodels`        |
| **Anomaly Detection**       | Isolation Forests + Z-Score Thresholding ($3\sigma$) | `scikit-learn`, `scipy`       |
| **Risk Quantification**     | Monte Carlo Simulations (10k iterations)             | `numpy`                       |
| **Risk Metrics**            | Conditional Value at Risk ($CVaR_{95}$), $VaR_{95}$  | Custom Quant-Finance approach |
| **Stationarity Testing**    | Augmented Dickey-Fuller (ADF)                        | `statsmodels`                 |

## 🚀 System Pipeline

The core logic executes in a unified pipeline capable of generating, processing, and evaluating telemetry data at scale.

1.  **Stochastic Data Generation:** Generates synthetic, statistically rigorous storage telemetry data (Daily Writes, Read Latency, Error Rates) mimicking real-world NAND degradation.
2.  **Feature Engineering:** Extracts advanced rolling statistics, volatility measures, lag features, and domain-specific ratios (e.g., Write-to-Read Ratio).
3.  **Threat & Anomaly Classification:** Identifies hardware degradation and classifies potential security anomalies (Ransomware patterns, Data Exfiltration).
4.  **Ensemble Forecasting:** Utilizes an optimized, weighted ensemble of Deep Learning (LSTM with Cosine Annealing, AdamW) and Statistical (ARIMA) models.
5.  **Monte Carlo Risk Evaluation:** Runs 10,000 simulations projecting future capacity and degradation trajectories to calculate specific risk bounds.

## 📂 Repository Structure

The intelligence pipeline is structured for modular deployment:

```text
stochastic-storage-var/
├── StorageIQ_Pipeline.py         # The core engine orchestrating data ops, ML inference, and risk math
├── StorageIQ_Colab.ipynb         # Interactive training environment with visualizations
├── models/                       # Serialized state dictionaries and model artifacts
│   ├── anomaly_model.pkl         # Trained Isolation Forest model
│   ├── arima_model.pkl           # Trained ARIMA statistical model
│   ├── lstm_model.pt             # Trained PyTorch LSTM weights
│   ├── scaler.pkl                # Data normalization scaler
│   └── model_meta.json           # Model configuration and metadata
├── StorageIQ_Dashboard/          # Real-time inference monitoring UI
│   ├── main.py                   # FastAPI backend server
│   └── index.html                # Frontend visualization client
└── README.md                     # Architecture documentation
```

## 💻 API Endpoints (Inference Orchestrator)

The system exposes a highly efficient, thread-safe inference wrapper designed for easy integration with external APIs.

```python
from model_export_and_api import ModelBundle, get_forecast, get_risk_score

# Initialize the inference registry (loads tensor representations into memory)
bundle = ModelBundle()

# Run fast inference across the orchestrator
forecast_results = get_forecast(df, bundle)
risk_assessment = get_risk_score(df, bundle)
```

## 🔒 Security Threat Vector Analysis

Beyond pure hardware lifecycle modeling, the anomaly detection module acts as a secondary security surface:

*   🔴 **RANSOMWARE_INDICATOR:** High Write Volatility + Elevated Latency
*   🟠 **DATA_EXFILTRATION:** Severe Read Spikes during historical low-traffic periods
*   🟡 **INSIDER_THREAT:** Erratic Write/Delete modifications

## 🛠 Prerequisites & Installation

To run the pipeline and local dashboard:

```bash
# Core mathematical and ML operations
pip install pandas numpy scipy scikit-learn statsmodels torch matplotlib joblib

# API / Dashboard components
pip install fastapi uvicorn
```
