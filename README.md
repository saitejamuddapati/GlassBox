# GlassBox: AI Risk Manager & Fraud Spike Sentinel
> **Track 02**: *Stop the merchant losing money to fraud, returns, and chargebacks.*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-FF6600)](https://xgboost.ai)
[![SHAP](https://img.shields.io/badge/TreeSHAP-C++%20Engine-000000)](https://github.com/slundberg/shap)
[![Defense](https://img.shields.io/badge/Defense--Only-Compliant-10B981)](#-why-glassbox-is-strictly-defensive)

---

## 🚀 Overview

In credit card payments, fraud detection systems face a costly dilemma:
* **Lenient Models** leak merchant capital through chargebacks, refund abuse, and stolen card transactions.
* **Aggressive Models** trigger false alarms, auto-declining genuine shoppers. Research reveals that **~33% of falsely declined customers permanently abandon that merchant**.

**GlassBox** is an open, explainable AI Risk Management platform designed to solve both problems simultaneously:
1. **Captures >85.7% of Fraud Attacks** and detects live multi-card fraud spikes in real time.
2. **Protects Legitimate Shoppers (Zero-Customer-Loss Policy)**: Reduces hard false alarms down to **0.012% (only 7 false blocks out of 56,864 genuine shoppers)**.
3. **Explains Every Decision in Plain English**: Powered by native C++ **TreeSHAP**, providing exact mathematical risk drivers and trust drivers without black-box opacity.

---

## 📊 Held-Out Test Set Benchmark Results
*Evaluated strictly on the isolated held-out test partition of **56,962 transactions** (56,864 legitimate, 98 actual frauds).*

| Metric | Result | Benchmark Significance |
| :--- | :--- | :--- |
| **PR-AUC (Average Precision)** | **88.07%** | Gold standard metric on heavy 0.17% class imbalance |
| **Hard Block Precision** | **92.05%** | 92%+ of instant hard-blocked cards ($\ge 0.70$) are confirmed fraud |
| **Hard False Alarm Rate** | **0.012%** | **Only 7 false blocks out of 56,864 genuine cardholders** |
| **Brier Calibration Score** | **0.000406** | Mathematical proof of probabilistic alignment ($0.0$ is perfect) |
| **Inference Throughput** | **>845 tx/s** | Sub-millisecond execution for live card-swipe authorization |

---

## 🛡️ The 3-Tier Zero-Customer-Loss Action Engine

Rather than relying on a crude binary "Block / Allow" switch, GlassBox routes transactions into 3 operational risk tiers:

```
                                [ Incoming Payment Stream ]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
             [ Supervised XGBoost ]                     [ Isolation Forest ]
             (5-Fold Platt Scaling)                     (Unsupervised Anomaly)
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │    Calibrated P(Fraud)    │
                               └─────────────┬─────────────┘
                                             │
         ┌───────────────────────────────────┼───────────────────────────────────┐
         ▼                                   ▼                                   ▼
 🟢 [GREEN TIER] (<0.08)            🟡 [YELLOW TIER] (0.08 - 0.70)      🔴 [RED TIER] (>=0.70)
   • Action: ALLOW                     • Action: REVIEW / 2FA              • Action: HOLD
   • 99.97% of normal users            • 3-Second SMS OTP Challenge        • Instant Hard Block
   • Frictionless checkout             • 0 Customer Loss, 0 Fraud Loss     • 92.05% Precision
```

---

## 🔍 Explainable AI (XAI) & SHAP Attributions

GlassBox translates complex mathematical feature attributions into plain-English analyst summaries:
* 🔺 **Top 3 Risk Drivers (Score UP)**: Highlights behavioral security divergences ($V_{14}$), velocity burst spikes ($V_4$), and non-standard purchase amounts.
* 🛡️ **Top 3 Trust Drivers (Score DOWN)**: Identifies verified device signatures and routine transaction baselines (preventing false declines on high-dollar \$1,500+ legitimate orders).

---

## ⚡ Quickstart: Running the Application

### 1. Installation
```powershell
git clone https://github.com/saitejamuddapati/GlassBox.git
cd GlassBox
pip install -r requirements.txt
```

### 2. Start the Backend & Web Dashboard
```powershell
python -m uvicorn src.api.main:app --port 8000 --reload
```

### 3. Open in Browser
Navigate to **`http://127.0.0.1:8000/`**:
* Click **"⚡ Load 10 Demo Transactions"** for instant 1-click evaluation.
* Drag and drop any transaction CSV (e.g. [`data/sample_eval_transactions.csv`](file:///c:/Users/tejbo/OneDrive/Documents/GlassBox/data/sample_eval_transactions.csv) or full datasets).
* Switch to the **⚡ Live Payment Simulator** tab to test card swipes with live parameter sliders.

---

## 📁 Repository Structure

```
GlassBox/
├── data/                       # Processed stratified datasets & sample evaluation CSVs
│   ├── processed/              # train.csv (227k) and test.csv (56k held-out)
│   └── sample_eval_transactions.csv
├── docs/                       # Comprehensive documentation & phase records
│   ├── architecture.md         # End-to-end system design & mathematical proofs
│   └── phase-notes/            # Step-by-step logs (Phases 0 through 6)
├── logs/                       # Immutable regulatory audit log (logs/audit.log)
├── src/
│   ├── data/                   # Data ingestion, cleaning, and stratified splitting
│   ├── models/                 # Model training, Platt calibration, Isolation Forest
│   │   └── saved/              # Serialized model binaries and metadata
│   ├── explain/                # Native TreeSHAP engine & plain-English translator
│   └── api/                    # FastAPI service, schemas, fraud spike detector, audit logger
├── web/                        # Modern Fintech Web Dashboard (HTML5/CSS3/Vanilla JS)
├── requirements.txt            # Project dependencies
└── README.md
```

---

## 🛡️ Hackathon Defense-Only Compliance
* **Pure Defensive Architecture**: Built strictly as an **AI Risk Manager & Fraud Spike Sentinel** to defend merchants and genuine cardholders from financial loss.
* **PCI-DSS Privacy by Design**: Processes only PCA-transformed behavioral vectors and tokenized amounts with zero storage of raw credit card numbers or sensitive PII.
* **Auditability**: Every transaction and batch execution is automatically logged to `logs/audit.log`.
