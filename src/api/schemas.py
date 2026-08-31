"""
Pydantic Data Schemas for GlassBox FastAPI Service.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Single transaction input payload."""
    Time: Optional[float] = Field(0.0, description="Seconds elapsed since start of dataset")
    Amount: float = Field(..., description="Transaction monetary amount")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Class: Optional[int] = Field(None, description="Optional ground truth class (0 or 1)")


class ExplanationReason(BaseModel):
    """Individual SHAP feature explanation reason."""
    feature: str
    raw_value: float
    shap_impact: float
    direction: str
    description: str


class TransactionAnalysisResult(BaseModel):
    """Analysis and explainability result for a single transaction."""
    transaction_id: int | str
    original_data: Dict[str, Any]
    risk_score: float
    risk_score_pct: str
    risk_band: str
    recommended_action: str
    action_description: str
    top_3_reasons_risk_up: List[ExplanationReason]
    top_3_reasons_risk_down: List[ExplanationReason]
    executive_narrative: str


class FraudSpikeStatus(BaseModel):
    """Fraud spike anomaly detector status."""
    spike_detected: bool
    batch_fraud_rate_pct: float
    baseline_expected_rate_pct: float
    spike_alert_message: str


class GroundTruthEvaluation(BaseModel):
    """Detailed ground truth evaluation metrics."""
    labels_detected: bool
    total_frauds_in_file: int
    total_legitimate_in_file: int
    frauds_blocked_instantly: int
    frauds_intercepted_2fa: int
    total_frauds_intercepted: int
    fraud_capture_rate: str
    hard_false_blocks: int
    hard_false_alarm_rate: str
    step_up_2fa_challenges: int
    legitimate_auto_approved: int
    hard_block_precision: str
    combined_precision: str


class BatchSummary(BaseModel):
    """Aggregated batch statistics."""
    total_transactions: int
    displayed_transactions: int
    risk_band_counts: Dict[str, int]
    risk_band_percentages: Dict[str, str]
    actions_breakdown: Dict[str, int]
    fraud_spike_detector: FraudSpikeStatus
    ground_truth_evaluation: Optional[GroundTruthEvaluation] = None


class BatchAnalysisResponse(BaseModel):
    """Complete response payload for CSV batch analysis."""
    summary: BatchSummary
    transactions: List[TransactionAnalysisResult]


class HealthCheckResponse(BaseModel):
    """Health check response."""
    model_config = {"protected_namespaces": ()}
    status: str
    version: str
    model_loaded: bool
    timestamp: str
