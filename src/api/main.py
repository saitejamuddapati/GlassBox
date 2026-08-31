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

# Mount static files for web dashboard
if web_dir.exists():
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")

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
    - Low (< 0.08 or < 0.30): Allow (Frictionless Approval)
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


@app.get("/")
def serve_dashboard():
    """Serve the GlassBox interactive web dashboard."""
    index_file = web_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "message": "GlassBox API is running. Open /web/index.html or /health for details."
    }


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

    has_labels = "Class" in df.columns
    total_records = len(df)

    # Run accelerated vectorized batch explanation
    batch_explanations = explainer.explain_batch(df)

    results: List[TransactionAnalysisResult] = []
    audit_records: List[Dict[str, Any]] = []
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    action_counts = {"Allow": 0, "Review": 0, "Hold": 0}

    # Ground truth counters
    gt_total_fraud = int((df["Class"] == 1).sum()) if has_labels else 0
    gt_tp = 0
    gt_fp = 0
    gt_tn = 0
    gt_fn = 0

    for idx, (explanation, (_, row)) in enumerate(zip(batch_explanations, df.iterrows())):
        tx_id = idx + 1
        prob = float(explanation["fraud_probability"])
        risk_band, action, action_desc = map_score_to_band_and_action(prob)

        risk_counts[risk_band] += 1
        action_counts[action] += 1

        # Ground truth tracking
        if has_labels:
            actual_is_fraud = int(row["Class"]) == 1
            predicted_is_flagged = action in ["Hold", "Review"]

            if actual_is_fraud and predicted_is_flagged:
                gt_tp += 1
            elif not actual_is_fraud and predicted_is_flagged:
                gt_fp += 1
            elif not actual_is_fraud and not predicted_is_flagged:
                gt_tn += 1
            elif actual_is_fraud and not predicted_is_flagged:
                gt_fn += 1

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

        orig_data = row.to_dict()

        # Audit record
        audit_records.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transaction_id": str(tx_id),
                "amount": float(row.get("Amount", 0.0)),
                "risk_score": round(prob, 6),
                "risk_band": risk_band,
                "recommended_action": action,
            }
        )

        results.append(
            TransactionAnalysisResult(
                transaction_id=tx_id,
                original_data=orig_data,
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

    # Fast batch write to audit log
    audit_logger.log_batch_transactions(audit_records)

    # Risk band percentages
    risk_percentages = {
        k: f"{(v / total_records) * 100:.2f}%" for k, v in risk_counts.items()
    }

    # Fraud Spike Detection Logic
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

    # Ground Truth Evaluation
    gt_eval: Optional[GroundTruthEvaluation] = None
    if has_labels:
        recall_val = (gt_tp / gt_total_fraud * 100) if gt_total_fraud > 0 else 0.0
        prec_val = (gt_tp / (gt_tp + gt_fp) * 100) if (gt_tp + gt_fp) > 0 else 0.0

        gt_eval = GroundTruthEvaluation(
            labels_detected=True,
            total_frauds_in_file=gt_total_fraud,
            frauds_intercepted=gt_tp,
            recall=f"{recall_val:.2f}%",
            precision=f"{prec_val:.2f}%",
            false_alarms=gt_fp,
            true_negatives=gt_tn,
        )

    # Batch summary
    summary = BatchSummary(
        total_transactions=total_records,
        risk_band_counts=risk_counts,
        risk_band_percentages=risk_percentages,
        actions_breakdown=action_counts,
        fraud_spike_detector=spike_status,
        ground_truth_evaluation=gt_eval,
    )

    # Log batch event
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
