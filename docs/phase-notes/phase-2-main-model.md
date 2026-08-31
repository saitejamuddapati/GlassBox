# Phase 2: Main Fraud Detection Model (XGBoost)

## Overview
Phase 2 implements the primary machine learning classification model for **GlassBox**. We trained an Extreme Gradient Boosting (**XGBoost**) classifier with cost-sensitive class weighting and evaluated it strictly on the held-out test partition (56,962 transactions).

---

## Actual Test Set Results

Evaluation on held-out test set (`data/processed/test.csv`):

| Metric | Score / Value | Why It Matters for the Panel |
| :--- | :--- | :--- |
| **PR-AUC (Precision-Recall AUC)** | **0.8584 (85.84%)** | Gold standard metric for rare events; reflects high precision across varying recall thresholds. |
| **Recall (Sensitivity)** | **0.8571 (85.71%)** | Caught **84 out of 98** actual fraud transactions. |
| **Precision** | **0.5957 (59.57%)** | When the model triggers an alert, ~60% are confirmed frauds. |
| **F1 Score** | **0.7029** | Strong harmonic balance between fraud detection and alert volume. |

### Confusion Matrix Breakdown (56,962 Total Test Transactions)
- **True Positives (Frauds Caught)**: **84**
- **False Negatives (Missed Frauds)**: **14**
- **False Positives (False Alarms)**: **57** (only ~0.10% of legitimate transactions flagged)
- **True Negatives (Legitimate Cleared)**: **56,807** / 56,864

---

## Panel Talking Points: Why These Technical Choices?

### 1. Why Accuracy is a Fatal Metric in Fraud Detection
- In our dataset, **99.83%** of transactions are legitimate and only **0.17%** are fraudulent.
- A useless "dummy" model that predicts *every transaction is legitimate* achieves **99.83% accuracy**, yet catches **0% of frauds**.
- In production, missing fraud causes direct financial loss, so we optimize for **Recall** and **Precision-Recall AUC (PR-AUC)** rather than raw accuracy.

### 2. Why XGBoost with Class Weighting (`scale_pos_weight`)?
- **Extreme Class Imbalance**: With ~577 legitimate transactions for every 1 fraud, standard gradient descent treats fraud as negligible noise.
- **Cost-Sensitive Learning**: Setting `scale_pos_weight = 577.29` tells XGBoost to penalize missing a fraud 577 times more heavily than generating a false alarm.
- **Superior Tabular Performance**: Gradient-boosted decision trees naturally capture non-linear relationships and interactions between PCA features (`V1`–`V28`) without requiring artificial data distortion.

### 3. Why PR-AUC Over ROC-AUC?
- ROC-AUC incorporates the True Negative Rate ($TN / (TN + FP)$). Because legitimate transactions ($TN$) are overwhelmingly large (~56,800), the False Positive Rate stays tiny even with hundreds of false alarms, creating an artificially optimistic ROC-AUC curve (often > 0.98).
- **PR-AUC focuses strictly on the minority positive class** (Precision vs. Recall). Achieving **0.8584 PR-AUC** proves real predictive power on needle-in-a-haystack fraud cases.

---

## Model Artifacts
- **Model format**: Saved to `src/models/saved/xgb_fraud_model.json` (~598 KB) and `src/models/saved/xgb_fraud_model.joblib` (~434 KB).
- **Metadata**: Saved to `src/models/saved/model_metadata.json`.
- Lightweight and ready for real-time inference and SHAP explainability.
