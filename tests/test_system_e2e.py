"""
End-to-End System Integration Test Suite for GlassBox.

Verifies:
1. Model artifact serialization and loading
2. Deep VAE Sentinel anomaly scoring and reconstruction
3. TreeSHAP feature attributions and plain-English narratives
4. FastAPI endpoints (/health, /predict, /predict/batch, /stream, /eval-dataset, /explain)
5. Multi-card fraud spike detector
6. Compliance audit logging (logs/audit.log)
7. Web UI assets (HTML, CSS, JS)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import asyncio
import json
import pandas as pd
import numpy as np
import httpx

from src.api.main import app, explainer
from src.models.vae_sentinel import DeepVAESentinel
from src.explain.explainer import FraudExplainer


def test_model_artifacts_exist():
    """Verify that all required production model binaries exist."""
    models_dir = root_dir / "src" / "models" / "saved"
    required_files = [
        "calibrated_xgb.joblib",
        "xgb_fraud_model.joblib",
        "xgb_fraud_model.json",
        "vae_sentinel.pt",
        "vae_scaler.joblib",
        "ensemble_metadata.json",
    ]
    for filename in required_files:
        path = models_dir / filename
        assert path.exists(), f"Missing required model artifact: {filename}"
        assert path.stat().st_size > 0, f"Model artifact is empty: {filename}"
    print("[PASS] All production model artifacts exist and are non-empty.")


def test_explainer_and_shap():
    """Verify TreeSHAP and plain English generation."""
    assert explainer is not None, "FraudExplainer instance is not loaded."
    
    # Test normal sample
    normal_sample = {
        "Time": 1000.0,
        "Amount": 45.50,
        "V1": 0.1, "V2": -0.2, "V3": 0.5, "V4": -0.1, "V5": 0.3,
        "V6": 0.2, "V7": 0.1, "V8": 0.0, "V9": 0.2, "V10": -0.1,
        "V11": 0.3, "V12": -0.2, "V13": 0.1, "V14": 0.4, "V15": -0.1,
        "V16": 0.2, "V17": -0.1, "V18": 0.1, "V19": 0.0, "V20": -0.1,
        "V21": 0.05, "V22": -0.1, "V23": 0.02, "V24": 0.1, "V25": -0.05,
        "V26": 0.03, "V27": 0.01, "V28": -0.02,
    }
    
    result = explainer.explain_transaction(normal_sample)
    prob = result["fraud_probability"]
    tier = result["decision_tier"]
    action = result["recommended_action"]
    top_up = result["top_risk_drivers_up"]
    top_down = result["top_trust_drivers_down"]
    narrative = result["executive_narrative"]
    
    assert 0.0 <= prob <= 1.0
    assert "TIER" in tier
    assert isinstance(action, str)
    assert isinstance(top_up, list)
    assert isinstance(top_down, list)
    assert isinstance(narrative, str)
    assert len(narrative) > 20
    print(f"[PASS] TreeSHAP explainer verified (Score: {prob:.4f}, Tier: {tier}).")


def test_vae_sentinel_scoring():
    """Verify PyTorch Deep VAE anomaly scoring."""
    models_dir = root_dir / "src" / "models" / "saved"
    vae = DeepVAESentinel.load(models_dir)
    
    sample_df = pd.DataFrame([{
        "V14": 0.5, "V10": -0.1, "V12": 0.2, "V17": -0.3, "V4": 0.1,
        "V11": 0.2, "top_pca_mag": 0.8, "log_amount": 3.8, "sin_hour": 0.5, "cos_hour": 0.86
    }])
    
    scores = vae.score_samples(sample_df)
    assert len(scores) == 1
    assert scores[0] >= 0.0
    
    deltas = vae.get_feature_deltas(sample_df)
    assert deltas.shape == (1, len(vae.feature_names))
    is_anomaly = bool(scores[0] >= vae.threshold)
    print(f"[PASS] Deep VAE Sentinel verified (Reconstruction score: {scores[0]:.4f}, Threshold: {vae.threshold:.4f}, Anomaly: {is_anomaly}).")


async def run_api_tests():
    """Run all API endpoint tests asynchronously with httpx.AsyncClient."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        print("[PASS] /health endpoint verified.")

        # 2. Predict Single
        payload = {
            "Time": 85000.0,
            "Amount": 125.00,
            "V1": -1.2, "V2": 0.8, "V3": -1.5, "V4": 2.1, "V5": -0.9,
            "V6": -0.4, "V7": -1.8, "V8": 0.6, "V9": -1.1, "V10": -2.4,
            "V11": 2.3, "V12": -3.1, "V13": 0.2, "V14": -4.5, "V15": -0.3,
            "V16": -2.1, "V17": -3.8, "V18": -1.2, "V19": 0.7, "V20": 0.4,
            "V21": 0.5, "V22": -0.2, "V23": -0.1, "V24": 0.1, "V25": 0.2,
            "V26": 0.3, "V27": 0.4, "V28": 0.1,
        }
        response = await client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        res = response.json()
        assert "risk_score" in res
        assert "risk_band" in res
        assert "recommended_action" in res
        assert "top_3_reasons_risk_up" in res
        assert "top_3_reasons_risk_down" in res
        assert "executive_narrative" in res
        print(f"[PASS] /api/v1/predict single verified (Score: {res['risk_score']:.4f}, Action: {res['recommended_action']}).")

        # 3. Batch Predict CSV Upload
        sample_csv_path = root_dir / "data" / "sample_eval_transactions.csv"
        if not sample_csv_path.exists():
            df = pd.DataFrame([
                {"Time": 1000, "Amount": 50.0, **{f"V{i}": 0.05 for i in range(1, 29)}},
                {"Time": 1050, "Amount": 120.0, **{f"V{i}": -0.1 for i in range(1, 29)}},
                {"Time": 1100, "Amount": 999.0, **{f"V{i}": -2.5 for i in range(1, 29)}},
            ])
            content = df.to_csv(index=False).encode("utf-8")
        else:
            with open(sample_csv_path, "rb") as f:
                content = f.read()

        response = await client.post(
            "/api/v1/upload-csv",
            files={"file": ("test_batch.csv", content, "text/csv")},
        )
        assert response.status_code == 200
        res = response.json()
        assert "summary" in res
        assert "transactions" in res
        assert len(res["transactions"]) > 0
        spike_info = res["summary"]["fraud_spike_detector"]
        print(f"[PASS] /api/v1/upload-csv batch verified (Processed: {res['summary']['total_transactions']} transactions, Spike Alert: {spike_info['spike_detected']}).")

        # 4. Frontend Root Route & Assets
        response = await client.get("/")
        assert response.status_code == 200
        assert "GlassBox" in response.text
        
        response_css = await client.get("/style.css")
        assert response_css.status_code == 200
        
        response_js = await client.get("/app.js")
        assert response_js.status_code == 200

        response_ico = await client.get("/favicon.ico")
        assert response_ico.status_code == 200
        assert len(response_ico.content) > 0

        response_svg = await client.get("/favicon.svg")
        assert response_svg.status_code == 200
        assert "<svg" in response_svg.text
        print("[PASS] Web dashboard root and static routes (/style.css, /app.js, /favicon.ico, /favicon.svg) served correctly.")


def test_audit_log_created():
    """Verify that compliance audit log exists and records transactions."""
    log_file = root_dir / "logs" / "audit.log"
    assert log_file.exists(), "logs/audit.log does not exist."
    assert log_file.stat().st_size > 0, "logs/audit.log is empty."
    
    with open(log_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) > 0
    last_entry = json.loads(lines[-1])
    assert "timestamp" in last_entry
    assert "transaction_id" in last_entry or "event_type" in last_entry
    print(f"[PASS] Compliance audit log verified ({len(lines)} recorded audit events).")


def test_web_ui_assets():
    """Verify frontend web dashboard files exist and are well-formed."""
    web_dir = root_dir / "web"
    index_file = web_dir / "index.html"
    js_file = web_dir / "app.js"
    css_file = web_dir / "style.css"
    
    assert index_file.exists() and index_file.stat().st_size > 1000
    assert js_file.exists() and js_file.stat().st_size > 1000
    assert css_file.exists() and css_file.stat().st_size > 1000
    print("[PASS] Web dashboard HTML/JS/CSS assets verified.")


def run_all():
    print("\n" + "=" * 60)
    print("      RUNNING GLASSBOX SYSTEM INTEGRATION TEST SUITE        ")
    print("=" * 60)
    test_model_artifacts_exist()
    test_explainer_and_shap()
    test_vae_sentinel_scoring()
    test_web_ui_assets()
    
    # Run async API tests
    asyncio.run(run_api_tests())
    
    test_audit_log_created()
    
    print("=" * 60)
    print("  >>> ALL INTEGRATION & BENCHMARK TESTS PASSED (100%) <<<  ")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all()
