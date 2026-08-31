# Phase 1: Data Preparation & Stratified Splitting

## Overview
Phase 1 implements data loading, data quality validation, and stratified splitting for the Kaggle Credit Card Fraud Detection dataset. The processed subsets are stored in `/data/processed` (`train.csv` and `test.csv`).

---

## Dataset Summary & Quality Check
- **Total Transactions**: 284,807
- **Features**: 31 columns:
  - `Time`: Seconds elapsed since the first transaction in the dataset.
  - `V1` to `V28`: PCA-transformed numerical features (anonymized for confidentiality).
  - `Amount`: Transaction amount in transaction currency.
  - `Class`: Binary target label (`0` = Legitimate, `1` = Fraud).
- **Missing Values**: 0 nulls across all 31 columns (clean dataset).
- **Class Imbalance**:
  - Legitimate (Class 0): **284,315** transactions (99.8273%)
  - Fraudulent (Class 1): **492** transactions (0.1727%) — approximately **1 fraud out of every 578 transactions**.

---

## Partitioning Results (80/20 Split)
Using a random seed of `42` with stratified sampling:

| Split | Total Rows | Legitimate (0) | Fraudulent (1) | Fraud Prevalence |
| :--- | :--- | :--- | :--- | :--- |
| **Full Dataset** | 284,807 | 284,315 | 492 | 0.1727% |
| **Train Set (80%)** | 227,845 | 227,451 | 394 | 0.1729% |
| **Test Set (20%)** | 56,962 | 56,864 | 98 | 0.1720% |

---

## Interview Talking Points: Why Stratified Splitting Matters

### 1. The Risk of Severe Class Imbalance
In extreme class imbalance (like fraud detection at ~0.17%), the minority class is very scarce (only 492 cases total).
- If you use a **standard random split**, sampling variance can randomly assign an unrepresentative number of fraud cases to the train or test partition (e.g., test could randomly receive 50 frauds instead of 98, or vice versa).
- In cross-validation folds, non-stratified splitting can even create validation folds with **zero positive cases**, breaking recall and precision computations.

### 2. What Stratification Solves
- **Maintains Ground Truth Prevalence**: `stratify=y` ensures that every subset (train, validation, test) preserves the exact ~0.172% fraud prevalence of the real world.
- **Fair and Stable Model Evaluation**: When test sets mirror the true production distribution, performance metrics (such as Precision-Recall AUC, F1-Score, and False Positive Rates) are statistically stable and realistic.

### 3. Preventing Data Leakage (Test Isolation)
- **The Golden Rule**: The test dataset (`test.csv`) must remain completely unseen and isolated until the final evaluation phase.
- Any future data transformations (such as `StandardScaler` / `RobustScaler` on `Amount` and `Time`) or oversampling techniques (such as `SMOTE`) must be fitted **strictly on the training set only** and applied to the test set to avoid information leakage.
