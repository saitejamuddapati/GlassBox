# GlassBox Empirical Benchmark Assets

This directory contains publication-grade, high-resolution visual proof figures generated directly from true empirical inference on the **56,962 held-out test transactions** (`data/processed/test.csv`) and saved model weights (`src/models/saved/`).

---

## Generated Figures & Descriptions

### 1. `pr_roc_benchmark.png`
- **Panel A (Left)**: **Precision-Recall Curve** demonstrating an empirical **PR-AUC of 86.66%** against the no-skill baseline (0.172% fraud prevalence). Shows the optimal operating point at $\tau = 0.79$ (Precision: 92.0%, Recall: 81.6%).
- **Panel B (Right)**: **ROC Curve** demonstrating an empirical **ROC-AUC of 98.24%** with near-zero False Positive Rate (0.012%) at operational sensitivity.

### 2. `precision_recall_threshold_tradeoff.png`
- Plots **Precision** (rising curve, blue) and **Recall** (falling curve, red) across all decision thresholds $\tau \in [0, 1]$.
- Mathematically validates the **3-Tier Zero-Customer-Loss Operating Points**:
  - **Yellow Tier ($\tau = 0.08$)**: Maximizes recall (~92%) for non-intrusive 2FA challenges with zero customer abandonment loss.
  - **Red Tier ($\tau = 0.70$)**: Guarantees high-confidence automated blocking with **85.3% precision**.
  - **Optimal F1 Point ($\tau = 0.79$)**: Peak balanced F1 score of **86.5%**.

### 3. `calibration_curve.png`
- **Reliability Diagram**: Compares uncalibrated XGBoost vs. **5-Fold Cross-Validated Platt Scaling** against the ideal $y = x$ diagonal.
- Confirms posterior probability $P(\text{Fraud} \mid X)$ calibration with an ultra-low **Brier Score of 0.000483**.
- Includes probability density distribution across legitimate shoppers vs. confirmed fraud attacks.

### 4. `vae_reconstruction_divergence.png`
- Log-scale distribution comparison of reconstruction Mean Squared Error (MSE) from the **Deep Variational Autoencoder (VAE)** Sentinel.
- Demonstrates a **153.9x Anomaly Divergence Ratio**:
  - Legitimate cardholders: Mean MSE = **0.06875** (95th percentile = 0.2089)
  - Confirmed fraud attacks: Mean MSE = **10.58225** (95th percentile = 48.7414)
- Highlights the 99.5th percentile sentinel threshold at **0.7256**.

### 5. `zero_customer_loss_matrix.png`
- Infographic and routing matrix of all 56,962 test transactions across the 3-Tier engine:
  - **Green Tier (Auto-Approve, score < 0.08)**: 56,856 shoppers (99.81% volume) experience zero checkout friction.
  - **Yellow Tier (2FA Challenge, 0.08 ≤ score < 0.70)**: 11 transactions challenged; recovers fraud while allowing legitimate cardholders to complete purchases via SMS/app push without false decline loss.
  - **Red Tier (Direct Block, score ≥ 0.70)**: 95 high-confidence blocks with **85.3% precision**.
