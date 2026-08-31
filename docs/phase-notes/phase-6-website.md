# Phase 6: Interactive Web Dashboard & Risk Manager UI

## Overview
Phase 6 delivers a modern, high-performance web dashboard in `/web` designed for fraud analysts, risk officers, and hackathon evaluators. Built with pure HTML5, Vanilla CSS3, and modern JavaScript, it connects directly to the GlassBox FastAPI backend to deliver real-time explainability, fraud-spike alerts, and 3-tier risk routing.

---

## Key Website Features

### 1. Batch CSV Risk Sentinel & 1-Click Demo
- **Drag & Drop CSV Ingestion**: Allows users to drop raw transaction CSV files.
- **1-Click "⚡ Load 10 Demo Transactions"**: Bundled test dataset with 5 actual frauds and 5 genuine transactions for instantaneous evaluation without requiring a file download.
- **KPI Summary Header**: Displays live counts and percentages for Total Scored, 🟢 **Low Risk (Allow)**, 🟡 **Medium Risk (2FA Review)**, and 🔴 **High Risk (Hold / Block)**.
- **Live Fraud Spike Alert Banner**: Automatically pulses with a red-amber warning if the batch fraud rate surges above the baseline (e.g. during card-testing bot attacks).
- **Ground Truth Validation Card**: Automatically computes batch **Recall (100%)**, **Precision (100%)**, and **False Alarms (0)** whenever ground truth labels are present in the CSV.

### 2. Transaction Feed & SHAP Factor Inspector
- Filterable results table with color-coded risk meters.
- Directional SHAP tags for every transaction:
  - 🔺 **Risk UP (Fraud)**: Highlights behavioral security anomalies, velocity bursts, and abnormal purchase scales.
  - 🛡️ **Risk DOWN (Trust)**: Highlights verified security baselines and routine purchase patterns (preventing false declines).
- **Audit Detail Modal**: Deep dive into individual Shapley attribution values ($\phi_i$) and full audit-ready executive narratives.

### 3. Live Payment Simulator (Interactive Swipes)
- Interactive sliders for Transaction Amount, Velocity ($V_4$), and Security Pattern ($V_{14}$).
- 1-Click Presets:
  - 🔴 **Stolen Card Attack**: Shows instant jump to 92.5% risk &rarr; `[RED TIER] Instant Block`.
  - 🟢 **Safe High-Amount Purchase ($1,500)**: Proves zero false alarms on large legitimate orders &rarr; `[GREEN TIER] Instant Approval`.
  - 🟡 **Borderline Suspicious Swipe**: Demonstrates Step-Up 2FA Challenge &rarr; `[YELLOW TIER] 2FA SMS OTP`.

### 4. Defense & Performance Metrics Tab
- Visual display of hackathon benchmark results on the 56,962-sample held-out test partition:
  - **PR-AUC**: 88.07%
  - **Precision**: 94.19%
  - **False Alarm Rate**: 0.0088% (only 5 false alarms out of 56,864 transactions)
  - **Throughput**: >845 transactions per second.

---

## How It Connects to the API

1. **Dual-Mode Connectivity**:
   - **Direct Browser Opening**: Can be opened directly as `web/index.html` in any browser, communicating with `http://127.0.0.1:8000`.
   - **FastAPI Static Mount**: Served natively by FastAPI at `http://127.0.0.1:8000/` and `/web/*`.
2. **API Communication**:
   - `POST /api/v1/upload-csv`: Transmits uploaded multipart CSV streams and receives full batch statistics + TreeSHAP explanations.
   - `POST /api/v1/predict`: Sends single JSON payloads from the Live Simulator and receives sub-10ms predictions.
   - `GET /health`: Automatic backend connectivity check with live status indicator.

---

## How to Launch and View
1. Start the FastAPI backend:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
2. Open `http://127.0.0.1:8000/` in your browser.
