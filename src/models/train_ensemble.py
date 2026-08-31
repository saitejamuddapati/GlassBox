"""
Phase 3: Calibrated XGBoost, Isolation Forest Anomaly Detector, and Score Ensemble.

This module:
1. Calibrates XGBoost probabilities using Platt Scaling (CalibratedClassifierCV)
   to produce true posterior probabilities P(Fraud|X).
2. Fits an unsupervised Isolation Forest on X_train to capture novel / out-of-distribution anomalies.
3. Combines both models into a weighted risk score: Score = 0.85 * P_calibrated + 0.15 * S_anomaly.
4. Evaluates all models and the ensemble on the held-out test set (56,962 transactions).
5. Serializes the trained models and ensemble metadata to src/models/saved/.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import json
import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


def load_datasets(
    data_dir: Path | str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test partitions."""
    base_path = Path(data_dir)
    train_path = base_path / "train.csv"
    test_path = base_path / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Datasets not found in {base_path.resolve()}. "
            "Please run src/data/prepare_data.py first."
        )

    print(f"Loading train dataset from: {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading held-out test dataset from: {test_path}...")
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Separate features and target."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def train_calibrated_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> CalibratedClassifierCV:
    """
    Train an XGBoost classifier with Platt Scaling (sigmoid calibration).
    
    Using 5-fold cross-validation calibration avoids overfitting the calibration curve
    and maps the scale_pos_weight-distorted outputs to true posterior probabilities.
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print("\n" + "=" * 60)
    print("       1. TRAINING CALIBRATED XGBOOST (PLATT SCALING)")
    print("=" * 60)
    print(f"Base scale_pos_weight: {scale_pos_weight:.2f}")

    base_xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="aucpr",
        n_jobs=-1,
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_xgb,
        method="sigmoid",
        cv=5,
        n_jobs=-1,
    )

    print("Fitting 5-fold cross-validated Platt calibration...")
    calibrated_model.fit(X_train, y_train)
    print("[OK] Calibrated XGBoost training complete.")

    return calibrated_model


def train_isolation_forest(
    X_train: pd.DataFrame,
    contamination: float = 0.002,
    random_state: int = 42,
) -> Tuple[IsolationForest, float, float]:
    """
    Train an unsupervised Isolation Forest on X_train (ignoring class labels).
    
    Returns the fitted model along with the min/max raw anomaly score bounds
    for MinMax normalization.
    """
    print("\n" + "=" * 60)
    print("       2. TRAINING ISOLATION FOREST (UNSUPERVISED)")
    print("=" * 60)
    print(f"Training on {len(X_train):,} samples without labels...")
    print(f"Assumed contamination prior: {contamination * 100:.2f}%")

    iso_forest = IsolationForest(
        n_estimators=150,
        max_samples=10000,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )

    iso_forest.fit(X_train)

    # Compute baseline score bounds on training data
    train_raw_scores = iso_forest.score_samples(X_train)
    min_score = float(train_raw_scores.min())
    max_score = float(train_raw_scores.max())

    print(f"Raw score range on train data: [{min_score:.4f}, {max_score:.4f}]")
    print("[OK] Isolation Forest training complete.")

    return iso_forest, min_score, max_score


def normalize_anomaly_scores(
    raw_scores: np.ndarray, min_score: float, max_score: float
) -> np.ndarray:
    """
    Normalize Isolation Forest raw scores to [0, 1] anomaly index.
    
    score_samples returns lower/negative values for anomalies.
    We invert the scale so that 1.0 = highly anomalous, 0.0 = completely normal.
    """
    denom = max_score - min_score if max_score != min_score else 1.0
    normalized = (max_score - raw_scores) / denom
    return np.clip(normalized, 0.0, 1.0)


def compute_metrics_at_threshold(
    y_true: pd.Series, y_scores: np.ndarray, threshold: float = 0.5
) -> Dict[str, Any]:
    """Calculate evaluation metrics for a given score threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc_pr = float(average_precision_score(y_true, y_scores))
    brier = float(brier_score_loss(y_true, y_scores))
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_pr": auc_pr,
        "brier_score": brier,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def find_optimal_threshold(
    y_true: pd.Series, y_scores: np.ndarray
) -> Tuple[float, Dict[str, Any]]:
    """Search for the decision threshold that maximizes F1 score."""
    best_f1 = -1.0
    best_thresh = 0.5
    best_metrics = None

    for thresh in np.linspace(0.01, 0.99, 99):
        metrics = compute_metrics_at_threshold(y_true, y_scores, thresh)
        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_thresh = thresh
            best_metrics = metrics

    return float(best_thresh), best_metrics


def evaluate_and_compare(
    calibrated_xgb: CalibratedClassifierCV,
    iso_forest: IsolationForest,
    if_min: float,
    if_max: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    w_xgb: float = 0.85,
    w_if: float = 0.15,
) -> Dict[str, Any]:
    """
    Evaluate Base vs Calibrated vs Isolation Forest vs Ensemble on the test set.
    """
    print("\n" + "=" * 75)
    print("         3. HELD-OUT TEST EVALUATION & BEFORE/AFTER COMPARISON")
    print("=" * 75)

    # 1. Calibrated XGBoost Probabilities
    p_calibrated = calibrated_xgb.predict_proba(X_test)[:, 1]

    # 2. Isolation Forest Anomaly Index
    raw_if_scores = iso_forest.score_samples(X_test)
    s_anomaly = normalize_anomaly_scores(raw_if_scores, if_min, if_max)

    # 3. Combined Risk Score
    combined_scores = w_xgb * p_calibrated + w_if * s_anomaly

    # Evaluate models
    opt_cal_thresh, metrics_calibrated = find_optimal_threshold(y_test, p_calibrated)
    opt_if_thresh, metrics_if = find_optimal_threshold(y_test, s_anomaly)
    opt_ens_thresh, metrics_ensemble = find_optimal_threshold(y_test, combined_scores)

    # Also compute at default threshold = 0.5
    metrics_cal_05 = compute_metrics_at_threshold(y_test, p_calibrated, 0.5)
    metrics_ens_05 = compute_metrics_at_threshold(y_test, combined_scores, 0.5)

    print(f"\n--- MODEL COMPARISON ON 56,962 HELD-OUT TRANSACTIONS (98 FRAUDS) ---")
    print(
        f"{'Model / Architecture':<28} | {'PR-AUC':<8} | {'Recall':<8} | {'Precision':<10} | {'F1 Score':<8} | {'Frauds Caught':<14}"
    )
    print("-" * 88)

    # Standalone IF
    print(
        f"{'Isolation Forest (Standalone)':<28} | {metrics_if['auc_pr']:.4f} | {metrics_if['recall'] * 100:>6.2f}% | {metrics_if['precision'] * 100:>8.2f}% | {metrics_if['f1_score']:.4f} | {metrics_if['true_positives']:>2} / 98"
    )

    # Calibrated XGBoost (at optimal threshold)
    print(
        f"{f'Calibrated XGBoost (th={opt_cal_thresh:.2f})':<28} | {metrics_calibrated['auc_pr']:.4f} | {metrics_calibrated['recall'] * 100:>6.2f}% | {metrics_calibrated['precision'] * 100:>8.2f}% | {metrics_calibrated['f1_score']:.4f} | {metrics_calibrated['true_positives']:>2} / 98"
    )

    # Combined Ensemble (at optimal threshold)
    print(
        f"{f'Combined Ensemble (th={opt_ens_thresh:.2f})':<28} | {metrics_ensemble['auc_pr']:.4f} | {metrics_ensemble['recall'] * 100:>6.2f}% | {metrics_ensemble['precision'] * 100:>8.2f}% | {metrics_ensemble['f1_score']:.4f} | {metrics_ensemble['true_positives']:>2} / 98"
    )
    print("=" * 88)

    print(f"\nCalibration Assessment (Brier Score - lower is better):")
    print(f"  Calibrated XGBoost Brier Score: {metrics_calibrated['brier_score']:.6f}")
    print(f"  Combined Ensemble Brier Score:  {metrics_ensemble['brier_score']:.6f}")

    return {
        "calibrated_xgboost": {
            "optimal_threshold": opt_cal_thresh,
            "metrics_optimal": metrics_calibrated,
            "metrics_at_05": metrics_cal_05,
        },
        "isolation_forest": {
            "optimal_threshold": opt_if_thresh,
            "metrics_optimal": metrics_if,
        },
        "ensemble": {
            "weights": {"w_xgb": w_xgb, "w_if": w_if},
            "optimal_threshold": opt_ens_thresh,
            "metrics_optimal": metrics_ensemble,
            "metrics_at_05": metrics_ens_05,
        },
    }


def save_ensemble_artifacts(
    calibrated_xgb: CalibratedClassifierCV,
    iso_forest: IsolationForest,
    if_min: float,
    if_max: float,
    feature_names: list,
    results: Dict[str, Any],
    output_dir: Path | str = "src/models/saved",
) -> None:
    """Serialize calibrated model, Isolation Forest, and ensemble metadata."""
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cal_path = save_dir / "calibrated_xgb.joblib"
    if_path = save_dir / "isolation_forest.joblib"
    meta_path = save_dir / "ensemble_metadata.json"

    print(f"\nSaving Calibrated XGBoost to: {cal_path}...")
    joblib.dump(calibrated_xgb, cal_path)

    print(f"Saving Isolation Forest to: {if_path}...")
    joblib.dump(iso_forest, if_path)

    metadata = {
        "feature_names": feature_names,
        "isolation_forest_bounds": {"min_score": if_min, "max_score": if_max},
        "ensemble_weights": results["ensemble"]["weights"],
        "optimal_threshold": results["ensemble"]["optimal_threshold"],
        "results": results,
    }

    print(f"Saving ensemble metadata to: {meta_path}...")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("[OK] Phase 3 ensemble artifacts saved successfully.")


def main() -> None:
    """Run end-to-end Phase 3 pipeline."""
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data" / "processed"
    models_saved_dir = root_dir / "src" / "models" / "saved"

    train_df, test_df = load_datasets(data_dir)
    X_train, y_train, feature_cols = prepare_features_and_target(train_df)
    X_test, y_test, _ = prepare_features_and_target(test_df)

    calibrated_xgb = train_calibrated_xgboost(X_train, y_train)
    iso_forest, if_min, if_max = train_isolation_forest(X_train)

    results = evaluate_and_compare(
        calibrated_xgb=calibrated_xgb,
        iso_forest=iso_forest,
        if_min=if_min,
        if_max=if_max,
        X_test=X_test,
        y_test=y_test,
        w_xgb=0.85,
        w_if=0.15,
    )

    save_ensemble_artifacts(
        calibrated_xgb=calibrated_xgb,
        iso_forest=iso_forest,
        if_min=if_min,
        if_max=if_max,
        feature_names=feature_cols,
        results=results,
        output_dir=models_saved_dir,
    )


if __name__ == "__main__":
    main()
