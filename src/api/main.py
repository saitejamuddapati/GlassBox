"""
FastAPI Service for GlassBox Explainable Fraud Detection System.

Provides endpoints for real-time transaction scoring, batch CSV uploads,
SHAP explainability, 3-tier risk routing (Allow/Review/Hold), fraud-spike detection,
compliance audit logging, and serves the web frontend dashboard.
"""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.explain.explainer import FraudExplainer
from src.api.schemas import (
    TransactionInput,
    TransactionAnalysisResult,
    ExplanationReason,
    BatchSummary,
    FraudSpikeStatus,
    GroundTruthEvaluation,
    BatchAnalysisResponse,
    HealthCheckResponse,
)
from src.api.audit import audit_logger


app = FastAPI(
    title="GlassBox: AI Risk Manager & Fraud Spike Sentinel",
    description="Explainable, real-time credit card fraud detection and risk management service.",
    version="1.0.0",
)

# Enable CORS for browser frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
root_dir = Path(__file__).resolve().parents[2]
models_dir = root_dir / "src" / "models" / "saved"
web_dir = root_dir / "web"

# Mount static files under /web and /static for assets
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

# Load global model & explainer instance
try:
    explainer = FraudExplainer(models_dir=models_dir)
except Exception as e:
    print(f"[WARNING] Could not initialize FraudExplainer on startup: {e}")
    explainer = None


def map_score_to_band_and_action(prob_fraud: float) -> Tuple[str, str, str]:
    """
    Map a calibrated probability score to Risk Band and Recommended Action.
    
    Bands:
    - Low (< 0.08): Allow (Frictionless Approval)
    - Medium (0.08 to 0.70): Review (Step-Up 2FA Challenge / SMS OTP)
    - High (>= 0.70): Hold (Instant Hard Block / Freeze)
    """
    if prob_fraud is None or pd.isna(prob_fraud):
        prob_fraud = 0.0

    if prob_fraud >= 0.70:
        return (
            "High",
            "Hold",
            "Decline transaction immediately and flag card for investigation.",
        )
    elif prob_fraud >= 0.08:
        return (
            "Medium",
            "Review",
            "Prompt cardholder with SMS OTP or biometric verification (Do not auto-decline).",
        )
    else:
        return (
            "Low",
            "Allow",
            "Fast-track transaction without customer friction.",
        )


def sanitize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and normalize column casing to match expected dataset schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == "amount":
            rename_map[col] = "Amount"
        elif col_lower == "time":
            rename_map[col] = "Time"
        elif col_lower == "class":
            rename_map[col] = "Class"
        elif col_lower.startswith("v") and col_lower[1:].isdigit():
            rename_map[col] = f"V{col_lower[1:]}"

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


# -------------------------------------------------------------
# Web Frontend Serving Routes
# -------------------------------------------------------------
@app.get("/")
def serve_dashboard():
    """Serve the GlassBox interactive web dashboard at root."""
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file, media_type="text/html")
    return {
        "message": "GlassBox API is running. Open /web/index.html or /health for details."
    }


@app.get("/style.css")
def serve_css():
    """Serve CSS stylesheet for root dashboard."""
    css_file = web_dir / "style.css"
    if css_file.exists():
        return FileResponse(css_file, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


@app.get("/app.js")
def serve_js():
    """Serve JavaScript logic for root dashboard."""
    js_file = web_dir / "app.js"
    if js_file.exists():
        return FileResponse(js_file, media_type="application/javascript")
    raise HTTPException(status_code=404, detail="app.js not found")


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
@app.get("/health", response_model=HealthCheckResponse)
def health_check() -> HealthCheckResponse:
    """Service health and readiness check."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=explainer is not None,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/api/v1/predict", response_model=TransactionAnalysisResult)
def predict_single_transaction(transaction: TransactionInput) -> TransactionAnalysisResult:
    """
    Score and explain a single incoming transaction in real time.
    """
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure model artifacts exist in src/models/saved/.",
        )

    data_dict = transaction.model_dump()
    explanation = explainer.explain_transaction(data_dict)

    prob = float(explanation["fraud_probability"])
    risk_band, action, action_desc = map_score_to_band_and_action(prob)

    reasons_up = [
        ExplanationReason(
            feature=f["feature"],
            raw_value=f["raw_value"],
            shap_impact=f["shap_value"],
            direction=f["direction"],
            description=f["explanation"],
        )
        for f in explanation["top_risk_drivers_up"]
    ]

    reasons_down = [
        ExplanationReason(
            feature=f["feature"],
            raw_value=f["raw_value"],
            shap_impact=f["shap_value"],
            direction=f["direction"],
            description=f["explanation"],
        )
        for f in explanation["top_trust_drivers_down"]
    ]

    # Audit log entry
    audit_logger.log_transaction(
        transaction_id=1,
        amount=transaction.Amount,
        risk_score=prob,
        risk_band=risk_band,
        recommended_action=action,
        metadata={"decision_tier": explanation["decision_tier"]},
    )

    return TransactionAnalysisResult(
        transaction_id=1,
        original_data=data_dict,
        risk_score=round(prob, 4),
        risk_score_pct=f"{prob * 100:.2f}%",
        risk_band=risk_band,
        recommended_action=action,
        action_description=action_desc,
        top_3_reasons_risk_up=reasons_up,
        top_3_reasons_risk_down=reasons_down,
        executive_narrative=explanation["executive_narrative"],
    )


@app.post("/api/v1/upload-csv", response_model=BatchAnalysisResponse)
async def upload_csv_batch(file: UploadFile = File(...)) -> BatchAnalysisResponse:
    """
    Ingest a CSV file of transactions, run vectorized batch inference and SHAP explainability,
    compute risk band statistics, detect fraud spikes, and evaluate precision/recall if labeled.
    """
    if explainer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please ensure model artifacts exist in src/models/saved/.",
        )

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid .csv file.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse uploaded CSV file: {str(e)}",
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty.",
        )

    # Normalize column names
    df = sanitize_dataframe_columns(df)

    # Check required columns (Amount and V1..V28)
    required_cols = ["Amount"] + [f"V{i}" for i in range(1, 29)]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        missing_preview = missing[:5]
        extra_count = len(missing) - 5
        detail_msg = f"Missing columns: {missing_preview} (+{extra_count} more)" if extra_count > 0 else f"Missing columns: {missing}"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Uploaded CSV is missing required columns. {detail_msg}",
        )

    total_records = len(df)
    has_labels = "Class" in df.columns

    # 1. Vectorized Preparation & Scoring across ALL rows
    X_all = explainer._prepare_input_vector(df)
    probs = explainer.calibrated_model.predict_proba(X_all)[:, 1]

    red_mask = probs >= 0.70
    yellow_mask = (probs >= 0.08) & (probs < 0.70)
    green_mask = probs < 0.08

    risk_counts = {
        "High": int(red_mask.sum()),
        "Medium": int(yellow_mask.sum()),
        "Low": int(green_mask.sum()),
    }

    action_counts = {
        "Hold": int(red_mask.sum()),
        "Review": int(yellow_mask.sum()),
        "Allow": int(green_mask.sum()),
    }

    risk_percentages = {
        k: f"{(v / total_records) * 100:.2f}%" for k, v in risk_counts.items()
    }

    # 2. Fraud Spike Sentinel Logic
    flagged_total = risk_counts["High"] + risk_counts["Medium"]
    batch_fraud_rate = (flagged_total / total_records) * 100
    baseline_rate = 0.17  # Standard 0.17% population baseline

    spike_detected = batch_fraud_rate >= 1.0 and flagged_total >= 2
    if spike_detected:
        spike_msg = (
            f"CRITICAL FRAUD SPIKE DETECTED: Batch suspicious rate ({batch_fraud_rate:.2f}%) "
            f"is {batch_fraud_rate / baseline_rate:.1f}x higher than standard baseline ({baseline_rate}%). "
            f"Automated bot or card-testing attack pattern suspected."
        )
    else:
        spike_msg = f"Normal volume: Batch risk rate ({batch_fraud_rate:.2f}%) within baseline expectations."

    spike_status = FraudSpikeStatus(
        spike_detected=spike_detected,
        batch_fraud_rate_pct=round(batch_fraud_rate, 2),
        baseline_expected_rate_pct=baseline_rate,
        spike_alert_message=spike_msg,
    )

    # 3. Ground Truth Precision/Recall Evaluation (Strict Breakdown)
    gt_eval: Optional[GroundTruthEvaluation] = None
    if has_labels:
        y = df["Class"].values.astype(int)
        total_frauds = int((y == 1).sum())
        total_legit = int((y == 0).sum())

        red_tp = int(((red_mask) & (y == 1)).sum())
        red_fp = int(((red_mask) & (y == 0)).sum())  # True Hard False Alarms

        yellow_tp = int(((yellow_mask) & (y == 1)).sum())
        yellow_fp = int(((yellow_mask) & (y == 0)).sum())  # 2FA Challenges (NOT blocked)

        green_tn = int(((green_mask) & (y == 0)).sum())
        green_fn = int(((green_mask) & (y == 1)).sum())

        total_frauds_intercepted = red_tp + yellow_tp
        hard_prec_val = (red_tp / (red_tp + red_fp) * 100) if (red_tp + red_fp) > 0 else 0.0
        comb_prec_val = (total_frauds_intercepted / (total_frauds_intercepted + red_fp + yellow_fp) * 100) if (total_frauds_intercepted + red_fp + yellow_fp) > 0 else 0.0
        recall_val = (total_frauds_intercepted / total_frauds * 100) if total_frauds > 0 else 0.0
        false_alarm_rate_val = (red_fp / total_legit * 100) if total_legit > 0 else 0.0

        gt_eval = GroundTruthEvaluation(
            labels_detected=True,
            total_frauds_in_file=total_frauds,
            total_legitimate_in_file=total_legit,
            frauds_blocked_instantly=red_tp,
            frauds_intercepted_2fa=yellow_tp,
            total_frauds_intercepted=total_frauds_intercepted,
            fraud_capture_rate=f"{recall_val:.2f}%",
            hard_false_blocks=red_fp,
            hard_false_alarm_rate=f"{false_alarm_rate_val:.4f}%",
            step_up_2fa_challenges=yellow_fp,
            legitimate_auto_approved=green_tn,
            hard_block_precision=f"{hard_prec_val:.2f}%",
            combined_precision=f"{comb_prec_val:.2f}%",
        )

    # 4. Detailed Sample Selection for Browser Rendering (Zero-Lag Display)
    # If file is large (> 250 rows), prioritize all flagged records + sample of normal records
    MAX_DISPLAY = 250
    if total_records > MAX_DISPLAY:
        flagged_indices = np.where(red_mask | yellow_mask)[0]
        unflagged_indices = np.where(green_mask)[0]
        
        remaining_slots = max(0, MAX_DISPLAY - len(flagged_indices))
        selected_unflagged = unflagged_indices[:remaining_slots]
        
        chosen_indices = np.sort(np.concatenate([flagged_indices, selected_unflagged]))
        display_df = df.iloc[chosen_indices].copy()
        display_indices_map = chosen_indices
    else:
        display_df = df.copy()
        display_indices_map = np.arange(total_records)

    # Compute SHAP on displayed sample
    batch_explanations = explainer.explain_batch(display_df)

    results: List[TransactionAnalysisResult] = []
    for idx, (explanation, (_, row)) in enumerate(zip(batch_explanations, display_df.iterrows())):
        orig_row_idx = int(display_indices_map[idx])
        tx_id = orig_row_idx + 1
        prob = float(explanation["fraud_probability"])
        risk_band, action, action_desc = map_score_to_band_and_action(prob)

        reasons_up = [
            ExplanationReason(
                feature=f["feature"],
                raw_value=f["raw_value"],
                shap_impact=f["shap_value"],
                direction=f["direction"],
                description=f["explanation"],
            )
            for f in explanation["top_risk_drivers_up"]
        ]

        reasons_down = [
            ExplanationReason(
                feature=f["feature"],
                raw_value=f["raw_value"],
                shap_impact=f["shap_value"],
                direction=f["direction"],
                description=f["explanation"],
            )
            for f in explanation["top_trust_drivers_down"]
        ]

        results.append(
            TransactionAnalysisResult(
                transaction_id=tx_id,
                original_data=row.to_dict(),
                risk_score=round(prob, 4),
                risk_score_pct=f"{prob * 100:.2f}%",
                risk_band=risk_band,
                recommended_action=action,
                action_description=action_desc,
                top_3_reasons_risk_up=reasons_up,
                top_3_reasons_risk_down=reasons_down,
                executive_narrative=explanation["executive_narrative"],
            )
        )

    # Summary
    summary = BatchSummary(
        total_transactions=total_records,
        displayed_transactions=len(results),
        risk_band_counts=risk_counts,
        risk_band_percentages=risk_percentages,
        actions_breakdown=action_counts,
        fraud_spike_detector=spike_status,
        ground_truth_evaluation=gt_eval,
    )

    # Log summary event to audit
    audit_logger.log_batch_summary(
        batch_size=total_records,
        risk_counts=risk_counts,
        action_counts=action_counts,
        spike_detected=spike_detected,
    )

    return BatchAnalysisResponse(
        summary=summary,
        transactions=results,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
