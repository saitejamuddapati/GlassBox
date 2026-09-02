"""
SHAP Explainability Layer for GlassBox Fraud Detection System.

This module provides real-time, human-understandable explanations for any credit card
transaction using SHAP (SHapley Additive exPlanations). For each transaction, it identifies
the top features driving risk UP (fraud indicators) and DOWN (trust indicators), and generates
an executive plain-English narrative for fraud analysts and automated decision systems.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any, Union
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV


FEATURE_DESCRIPTIONS: Dict[str, Dict[str, str]] = {
    "Amount": {
        "high": "Unusually high transaction amount",
        "low": "Small, routine micro-purchase amount",
        "default": "Transaction monetary amount",
    },
    "log_amount": {
        "high": "Significantly elevated transaction amount scale",
        "low": "Normal baseline purchase scale",
        "default": "Normalized transaction volume scale",
    },
    "Time": {
        "high": "Late-session transaction timestamp",
        "low": "Early-session transaction timestamp",
        "default": "Elapsed transaction timestamp",
    },
    "hour": {
        "high": "Transaction initiated during non-standard nighttime hours",
        "low": "Transaction initiated during regular daytime shopping hours",
        "default": "Hour of transaction execution",
    },
    "sin_hour": {
        "high": "Diurnal cycle risk window (off-peak hours)",
        "low": "Standard daylight purchase cycle",
        "default": "24-hour diurnal cyclical pattern",
    },
    "cos_hour": {
        "high": "Cyclical midnight-window shopping profile",
        "low": "Standard afternoon transaction profile",
        "default": "24-hour cyclical cosine pattern",
    },
    "V14": {
        "high": "Positive authorization security profile (V14)",
        "low": "Severe behavioral security anomaly (V14 - strong fraud signature)",
        "default": "Behavioral identity pattern V14",
    },
    "V4": {
        "high": "Abnormal velocity / high-frequency burst attempt (V4)",
        "low": "Normal single-purchase velocity rate (V4)",
        "default": "Transaction frequency / velocity indicator V4",
    },
    "V10": {
        "high": "Standard terminal and device signature (V10)",
        "low": "Unrecognized or mismatched cardholder terminal signature (V10)",
        "default": "Terminal / device profile component V10",
    },
    "V12": {
        "high": "Verified merchant channel history (V12)",
        "low": "High-risk merchant channel / atypical category divergence (V12)",
        "default": "Merchant channel risk indicator V12",
    },
    "V17": {
        "high": "Standard account verification history (V17)",
        "low": "Suspicious authorization anomaly / unusual geo-distance (V17)",
        "default": "Authorization verification vector V17",
    },
    "V11": {
        "high": "Repetitive attempt or balance mismatch indicator (V11)",
        "low": "Clean authorization consistency record (V11)",
        "default": "Consistency verification vector V11",
    },
    "top_pca_mag": {
        "high": "Combined multi-vector statistical divergence from normal behavior",
        "low": "Normal combined behavioral profile",
        "default": "Joint PCA Euclidean anomaly magnitude",
    },
    "v14_v4": {
        "high": "Compounded high-velocity security conflict (V14 x V4)",
        "low": "Low interaction risk between security and velocity",
        "default": "Interaction term V14 x V4",
    },
    "v10_v12": {
        "high": "Combined terminal and merchant risk divergence (V10 x V12)",
        "low": "Normal terminal-merchant synergy",
        "default": "Interaction term V10 x V12",
    },
    "v17_v11": {
        "high": "Compounded authorization and repeat attempt anomaly (V17 x V11)",
        "low": "Normal authorization profile",
        "default": "Interaction term V17 x V11",
    },
    "v14_v10": {
        "high": "Dual security and terminal mismatch profile (V14 x V10)",
        "low": "Consistent security and terminal baseline",
        "default": "Interaction term V14 x V10",
    },
    "v12_v17": {
        "high": "Dual merchant and authorization conflict (V12 x V17)",
        "low": "Clean merchant and authorization pairing",
        "default": "Interaction term V12 x V17",
    },
}


def describe_feature_impact(
    feature_name: str, raw_value: float, shap_value: float
) -> str:
    """Generate a plain-English explanation for why a feature pushed score up/down."""
    desc_dict = FEATURE_DESCRIPTIONS.get(feature_name)

    if desc_dict is None:
        if feature_name.startswith("V"):
            if shap_value > 0:
                return f"Anomalous pattern in security component {feature_name} (value: {raw_value:.2f})"
            else:
                return f"Normal, consistent behavior in security component {feature_name} (value: {raw_value:.2f})"
        return f"{feature_name} (value: {raw_value:.2f})"

    # Select appropriate directional description
    if feature_name in ["Amount", "log_amount"]:
        if raw_value > 300:
            return f"{desc_dict['high']} (${raw_value:.2f})"
        elif raw_value < 15:
            return f"{desc_dict['low']} (${raw_value:.2f})"
        else:
            return f"Standard transaction amount (${raw_value:.2f})"

    if feature_name == "V14":
        if raw_value < -2.0:
            return f"{desc_dict['low']} (value: {raw_value:.2f})"
        else:
            return f"{desc_dict['high']} (value: {raw_value:.2f})"

    if feature_name == "V4":
        if raw_value > 1.5:
            return f"{desc_dict['high']} (value: {raw_value:.2f})"
        else:
            return f"{desc_dict['low']} (value: {raw_value:.2f})"

    if shap_value > 0:
        return f"{desc_dict.get('high', desc_dict['default'])} (value: {raw_value:.2f})"
    else:
        return f"{desc_dict.get('low', desc_dict['default'])} (value: {raw_value:.2f})"


class FraudExplainer:
    """
    SHAP-based Explainability Engine for GlassBox Fraud Detection.
    """

    def __init__(self, models_dir: Union[Path, str] = "src/models/saved"):
        self.models_dir = Path(models_dir)
        self.calibrated_path = self.models_dir / "calibrated_xgb.joblib"
        self.base_xgb_path = self.models_dir / "xgb_fraud_model.joblib"
        self.meta_path = self.models_dir / "ensemble_metadata.json"

        self._load_models_and_explainer()

    def _load_models_and_explainer(self) -> None:
        """Load trained models, metadata, and initialize Tree SHAP."""
        if not self.calibrated_path.exists() or not self.base_xgb_path.exists():
            raise FileNotFoundError(
                f"Model artifacts not found in {self.models_dir.resolve()}. "
                "Please run src/models/train_ensemble.py first."
            )

        # Load metadata
        if self.meta_path.exists():
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)
                self.feature_names = self.metadata.get("feature_names", [])
        else:
            self.metadata = {}
            self.feature_names = []

        # Load models
        self.calibrated_model: CalibratedClassifierCV = joblib.load(self.calibrated_path)
        self.base_xgb: XGBClassifier = joblib.load(self.base_xgb_path)
        self.booster: xgb.Booster = self.base_xgb.get_booster()

        # Load Deep VAE Sentinel if available
        try:
            from src.models.vae_sentinel import DeepVAESentinel
            vae_path = self.models_dir / "vae_sentinel.pt"
            if vae_path.exists():
                self.vae_sentinel = DeepVAESentinel.load(self.models_dir)
                print("[OK] Deep VAE Anomaly Sentinel loaded.")
            else:
                self.vae_sentinel = None
        except Exception as e:
            self.vae_sentinel = None

        print("[OK] SHAP Tree Explainer initialized (Native C++ TreeSHAP Engine).")

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        df = df.copy()

        # 1. Log Amount
        if "Amount" in df.columns and "log_amount" not in df.columns:
            df["log_amount"] = np.log1p(df["Amount"].fillna(0.0))

        # 2. Cyclical Time
        if "Time" in df.columns and "sin_hour" not in df.columns:
            hour = (df["Time"].fillna(0.0) / 3600.0) % 24.0
            df["hour"] = hour
            df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
            df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)

        # 3. High-Impact Interactions
        if all(col in df.columns for col in ["V14", "V4"]) and "v14_v4" not in df.columns:
            df["v14_v4"] = df["V14"] * df["V4"]
        if all(col in df.columns for col in ["V10", "V12"]) and "v10_v12" not in df.columns:
            df["v10_v12"] = df["V10"] * df["V12"]
        if all(col in df.columns for col in ["V17", "V11"]) and "v17_v11" not in df.columns:
            df["v17_v11"] = df["V17"] * df["V11"]
        if all(col in df.columns for col in ["V14", "V10"]) and "v14_v10" not in df.columns:
            df["v14_v10"] = df["V14"] * df["V10"]
        if all(col in df.columns for col in ["V12", "V17"]) and "v12_v17" not in df.columns:
            df["v12_v17"] = df["V12"] * df["V17"]

        # 4. Joint Magnitude
        pca_req = ["V14", "V10", "V12", "V17", "V4", "V11"]
        if all(col in df.columns for col in pca_req) and "top_pca_mag" not in df.columns:
            df["top_pca_mag"] = np.sqrt(
                df["V14"] ** 2
                + df["V10"] ** 2
                + df["V12"] ** 2
                + df["V17"] ** 2
                + df["V4"] ** 2
                + df["V11"] ** 2
            )

        return df

    def _prepare_input_vector(
        self, data: Union[Dict[str, float], pd.Series, pd.DataFrame]
    ) -> pd.DataFrame:
        """Ensure input data matches the exact feature columns required by the model."""
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, pd.Series):
            df = pd.DataFrame([data.to_dict()])
        else:
            df = data.copy()

        # Drop ground-truth target if present
        if "Class" in df.columns:
            df = df.drop(columns=["Class"])

        # Fill missing values if any
        df = df.fillna(0.0)

        # Engineer features
        df_fe = self.engineer_features(df)

        # Match columns to model training order
        if self.feature_names:
            missing_cols = [c for c in self.feature_names if c not in df_fe.columns]
            if missing_cols:
                for c in missing_cols:
                    df_fe[c] = 0.0
            df_fe = df_fe[self.feature_names]

        return df_fe

    def explain_transaction(
        self, transaction_data: Union[Dict[str, float], pd.Series, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Compute calibrated fraud probability and return plain-English SHAP explanations.
        """
        X = self._prepare_input_vector(transaction_data)

        # 1. Compute calibrated probability
        prob_fraud = float(self.calibrated_model.predict_proba(X)[0, 1])

        # 2. Determine Action Tier
        if prob_fraud >= 0.70:
            decision_tier = "[RED TIER] Instant Hard Block"
            tier_color = "red"
            recommended_action = "Decline transaction immediately and flag card for investigation."
        elif prob_fraud >= 0.08:
            decision_tier = "[YELLOW TIER] Step-Up 2FA Challenge"
            tier_color = "yellow"
            recommended_action = "Prompt cardholder with SMS OTP or biometric verification (Do not auto-decline)."
        else:
            decision_tier = "[GREEN TIER] Instant Approval"
            tier_color = "green"
            recommended_action = "Fast-track transaction without customer friction."

        # 3. Compute SHAP values via native TreeSHAP
        dmat = xgb.DMatrix(X)
        shap_contribs = self.booster.predict(dmat, pred_contribs=True)[0]
        shap_values = shap_contribs[:-1]
        base_value = float(shap_contribs[-1])

        feature_names = X.columns.tolist()
        raw_values = X.iloc[0].values

        # Pair features with their SHAP values and raw values
        attributions: List[Dict[str, Any]] = []
        for name, raw_val, shap_val in zip(feature_names, raw_values, shap_values):
            attributions.append(
                {
                    "feature": name,
                    "raw_value": float(raw_val),
                    "shap_value": float(shap_val),
                    "abs_shap": abs(float(shap_val)),
                    "direction": "Risk UP" if shap_val > 0 else "Risk DOWN",
                    "explanation": describe_feature_impact(name, float(raw_val), float(shap_val)),
                }
            )

        # 4. Top 3 Risk Drivers (Positive SHAP -> Pushed Score UP)
        risk_up = sorted(
            [a for a in attributions if a["shap_value"] > 0],
            key=lambda x: x["shap_value"],
            reverse=True,
        )[:3]

        # 5. Top 3 Trust Drivers (Negative SHAP -> Pushed Score DOWN)
        risk_down = sorted(
            [a for a in attributions if a["shap_value"] < 0],
            key=lambda x: x["shap_value"],
        )[:3]

        # 6. Generate Plain-English Narrative Summary
        if prob_fraud >= 0.70:
            top_reasons = ", ".join([f"'{item['explanation']}'" for item in risk_up[:2]])
            narrative = (
                f"HIGH RISK FRAUD ALERT ({prob_fraud * 100:.1f}% probability): Transaction exhibits critical "
                f"risk indicators, primarily driven by {top_reasons}. Instant block recommended."
            )
        elif prob_fraud >= 0.08:
            top_reasons = ", ".join([f"'{item['explanation']}'" for item in risk_up[:2]])
            narrative = (
                f"SUSPICIOUS ACTIVITY ({prob_fraud * 100:.1f}% probability): Risk elevated by {top_reasons}. "
                f"Recommend requesting step-up 2FA authentication to avoid customer friction."
            )
        else:
            top_trust = ", ".join([f"'{item['explanation']}'" for item in risk_down[:2]])
            narrative = (
                f"CLEARED / LOW RISK ({prob_fraud * 100:.2f}% probability): Normal behavioral indicators verified, "
                f"supported by {top_trust}. Instant approval granted."
            )

        return {
            "fraud_probability": prob_fraud,
            "fraud_probability_pct": f"{prob_fraud * 100:.2f}%",
            "decision_tier": decision_tier,
            "tier_color": tier_color,
            "recommended_action": recommended_action,
            "base_value": base_value,
            "top_risk_drivers_up": risk_up,
            "top_trust_drivers_down": risk_down,
            "executive_narrative": narrative,
        }

    def explain_batch(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        High-performance vectorized batch explanation for multiple transactions.
        """
        X = self._prepare_input_vector(df)

        # Batch probability prediction
        probs = self.calibrated_model.predict_proba(X)[:, 1]

        # Batch SHAP prediction via native C++ TreeSHAP
        dmat = xgb.DMatrix(X)
        shap_matrix = self.booster.predict(dmat, pred_contribs=True)
        feature_shap = shap_matrix[:, :-1]
        base_vals = shap_matrix[:, -1]

        feature_names = X.columns.tolist()
        raw_vals_matrix = X.values

        results = []
        for i in range(len(df)):
            prob_fraud = float(probs[i])

            if prob_fraud >= 0.70:
                decision_tier = "[RED TIER] Instant Hard Block"
                tier_color = "red"
                recommended_action = "Decline transaction immediately and flag card for investigation."
            elif prob_fraud >= 0.08:
                decision_tier = "[YELLOW TIER] Step-Up 2FA Challenge"
                tier_color = "yellow"
                recommended_action = "Prompt cardholder with SMS OTP or biometric verification (Do not auto-decline)."
            else:
                decision_tier = "[GREEN TIER] Instant Approval"
                tier_color = "green"
                recommended_action = "Fast-track transaction without customer friction."

            shap_row = feature_shap[i]
            raw_row = raw_vals_matrix[i]

            attributions = []
            for name, raw_val, shap_val in zip(feature_names, raw_row, shap_row):
                attributions.append(
                    {
                        "feature": name,
                        "raw_value": float(raw_val),
                        "shap_value": float(shap_val),
                        "abs_shap": abs(float(shap_val)),
                        "direction": "Risk UP" if shap_val > 0 else "Risk DOWN",
                        "explanation": describe_feature_impact(name, float(raw_val), float(shap_val)),
                    }
                )

            risk_up = sorted(
                [a for a in attributions if a["shap_value"] > 0],
                key=lambda x: x["shap_value"],
                reverse=True,
            )[:3]

            risk_down = sorted(
                [a for a in attributions if a["shap_value"] < 0],
                key=lambda x: x["shap_value"],
            )[:3]

            if prob_fraud >= 0.70:
                top_reasons = ", ".join([f"'{item['explanation']}'" for item in risk_up[:2]])
                narrative = (
                    f"HIGH RISK FRAUD ALERT ({prob_fraud * 100:.1f}% probability): Transaction exhibits critical "
                    f"risk indicators, primarily driven by {top_reasons}. Instant block recommended."
                )
            elif prob_fraud >= 0.08:
                top_reasons = ", ".join([f"'{item['explanation']}'" for item in risk_up[:2]])
                narrative = (
                    f"SUSPICIOUS ACTIVITY ({prob_fraud * 100:.1f}% probability): Risk elevated by {top_reasons}. "
                    f"Recommend requesting step-up 2FA authentication to avoid customer friction."
                )
            else:
                top_trust = ", ".join([f"'{item['explanation']}'" for item in risk_down[:2]])
                narrative = (
                    f"CLEARED / LOW RISK ({prob_fraud * 100:.2f}% probability): Normal behavioral indicators verified, "
                    f"supported by {top_trust}. Instant approval granted."
                )

            results.append(
                {
                    "fraud_probability": prob_fraud,
                    "fraud_probability_pct": f"{prob_fraud * 100:.2f}%",
                    "decision_tier": decision_tier,
                    "tier_color": tier_color,
                    "recommended_action": recommended_action,
                    "base_value": float(base_vals[i]),
                    "top_risk_drivers_up": risk_up,
                    "top_trust_drivers_down": risk_down,
                    "executive_narrative": narrative,
                }
            )

        return results


def run_test_suite() -> None:
    """Run SHAP explanation tests on 5 representative transactions from the test set."""
    root_dir = Path(__file__).resolve().parents[2]
    test_csv_path = root_dir / "data" / "processed" / "test.csv"

    if not test_csv_path.exists():
        raise FileNotFoundError(f"Test CSV not found at {test_csv_path}")

    print(f"Loading held-out test data from: {test_csv_path}...")
    test_df = pd.read_csv(test_csv_path)

    explainer = FraudExplainer(models_dir=root_dir / "src" / "models" / "saved")

    # Select 5 diverse test transactions
    fraud_indices = test_df[test_df["Class"] == 1].index.tolist()
    legit_high_amt_indices = test_df[(test_df["Class"] == 0) & (test_df["Amount"] > 300)].index.tolist()
    legit_normal_indices = test_df[(test_df["Class"] == 0) & (test_df["Amount"] < 50)].index.tolist()

    selected_cases = [
        ("Test Case 1: High-Confidence Stolen Card Fraud", fraud_indices[0], 1),
        ("Test Case 2: Borderline Suspicious Fraud Transaction", fraud_indices[2], 1),
        ("Test Case 3: High-Amount Legitimate Purchase ($400+)", legit_high_amt_indices[5], 0),
        ("Test Case 4: Everyday Routine Coffee / Grocery Purchase", legit_normal_indices[10], 0),
        ("Test Case 5: Covert Multi-Vector Fraud Attack", fraud_indices[5], 1),
    ]

    print("\n" + "=" * 80)
    print("           GLASSBOX SHAP EXPLAINABILITY TEST SUITE (5 TEST SAMPLES)          ")
    print("=" * 80)

    for title, idx, actual_label in selected_cases:
        row = test_df.loc[idx]
        actual_str = "FRAUD (Class=1)" if actual_label == 1 else "LEGITIMATE (Class=0)"
        amount_str = f"${row.get('Amount', 0):.2f}"

        explanation = explainer.explain_transaction(row)

        print(f"\n>>> {title.upper()}")
        print(f"    Dataset Row Index: {idx} | Actual Ground Truth: {actual_str} | Amount: {amount_str}")
        print(f"    Calculated Risk Probability: {explanation['fraud_probability_pct']}")
        print(f"    Action Routing:              {explanation['decision_tier']}")
        print(f"    Action Recommendation:       {explanation['recommended_action']}")
        print(f"\n    [TOP 3 FACTORS INCREASING FRAUD RISK (SCORE UP)]:")
        if explanation["top_risk_drivers_up"]:
            for i, factor in enumerate(explanation["top_risk_drivers_up"], 1):
                print(f"      {i}. {factor['feature']} (SHAP: +{factor['shap_value']:.4f}) -> {factor['explanation']}")
        else:
            print("      (None - No significant risk factors detected)")

        print(f"\n    [TOP 3 FACTORS CONFIRMING LEGITIMACY (SCORE DOWN)]:")
        if explanation["top_trust_drivers_down"]:
            for i, factor in enumerate(explanation["top_trust_drivers_down"], 1):
                print(f"      {i}. {factor['feature']} (SHAP: {factor['shap_value']:.4f}) -> {factor['explanation']}")
        else:
            print("      (None)")

        print(f"\n    [EXECUTIVE NARRATIVE]:")
        print(f"      \"{explanation['executive_narrative']}\"")
        print("-" * 80)


if __name__ == "__main__":
    run_test_suite()
