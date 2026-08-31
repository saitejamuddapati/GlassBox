"""
Audit Logging Module for GlassBox Fraud Detection Service.

Records all scored transactions, assigned risk bands, actions taken, and timestamps
to an immutable local audit log for regulatory compliance and analyst forensics.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import threading
from typing import Dict, Any


class AuditLogger:
    """Thread-safe audit logger for scoring transactions."""

    def __init__(self, log_path: Path | str = "logs/audit.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log_transaction(
        self,
        transaction_id: str | int,
        amount: float,
        risk_score: float,
        risk_band: str,
        recommended_action: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        """Append a single scored transaction record to the audit log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "transaction_id": str(transaction_id),
            "amount": float(amount),
            "risk_score": round(float(risk_score), 6),
            "risk_band": risk_band,
            "recommended_action": recommended_action,
            "metadata": metadata or {},
        }

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    def log_batch_summary(
        self,
        batch_size: int,
        risk_counts: Dict[str, int],
        action_counts: Dict[str, int],
        spike_detected: bool,
    ) -> None:
        """Log a batch execution event to the audit log."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": "BATCH_ANALYSIS",
            "batch_size": batch_size,
            "risk_counts": risk_counts,
            "action_counts": action_counts,
            "spike_detected": spike_detected,
        }

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


# Global audit logger instance
audit_logger = AuditLogger()
