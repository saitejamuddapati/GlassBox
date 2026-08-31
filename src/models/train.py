"""
XGBoost Fraud Classification Training & Evaluation for GlassBox.

This module trains an XGBoost classifier with cost-sensitive class weighting
(scale_pos_weight) to handle severe fraud imbalance (0.17%). It strictly evaluates
on the held-out test partition using Precision, Recall, F1, PR-AUC, and Confusion Matrix,
and serializes the model to src/models/saved/.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import json
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def load_datasets(
    data_dir: Path | str = "data/processed",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the pre-split train and test sets."""
    base_path = Path(data_dir)
    train_path = base_path / "train.csv"
    test_path = base_path / "test.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {base_path.resolve()}. "
            "Please run src/data/prepare_data.py first."
        )

    print(f"Loading train dataset: {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Loading held-out test dataset: {test_path}...")
    test_df = pd.read_csv(test_path)

    return train_df, test_df


def prepare_features_and_target(
    df: pd.DataFrame, target_col: str = "Class"
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """Separate feature matrix X and target vector y."""
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Tuple[XGBClassifier, float]:
    """
    Train an XGBoost model with cost-sensitive class weighting.
    
    scale_pos_weight = negative_count / positive_count
    This penalizes false negatives heavily, making the model sensitive to fraud.
    """
    neg_count = int((y_train == 0).sum())
    pos_count = int((y_train == 1).sum())
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print("\n" + "=" * 55)
    print("                MODEL TRAINING SETUP")
    print("=" * 55)
    print(f"Training Samples:        {len(X_train):,}")
    print(f"Legitimate Transactions: {neg_count:,}")
    print(f"Fraudulent Transactions: {pos_count:,}")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.2f}")
    print("=" * 55)

    model = XGBClassifier(
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

    print("Training XGBoost classifier...")
    model.fit(X_train, y_train)
    print("[OK] Model training complete.")

    return model, scale_pos_weight


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluate model performance strictly on held-out test data.
    
    Computes Precision, Recall, F1, PR-AUC, and Confusion Matrix.
    """
    print("\n" + "=" * 65)
    print("         HELD-OUT TEST SET EVALUATION (56,962 TRANSACTIONS)")
    print("=" * 65)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    auc_pr = float(average_precision_score(y_test, y_prob))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"{'Metric':<30} | {'Score / Count':<20}")
    print("-" * 65)
    print(f"{'PR-AUC (Average Precision)':<30} | {auc_pr:.4f} ({auc_pr * 100:.2f}%)")
    print(f"{'Precision':<30} | {precision:.4f} ({precision * 100:.2f}%)")
    print(f"{'Recall (Sensitivity)':<30} | {recall:.4f} ({recall * 100:.2f}%)")
    print(f"{'F1 Score':<30} | {f1:.4f}")
    print("-" * 65)
    print(f"{'True Positives (Frauds Caught)':<30} | {tp:,} / {tp + fn:,}")
    print(f"{'False Positives (False Alarms)':<30} | {fp:,}")
    print(f"{'False Negatives (Missed Frauds)':<30} | {fn:,}")
    print(f"{'True Negatives (Correct Legits)':<30} | {tn:,} / {tn + fp:,}")
    print("=" * 65 + "\n")

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_pr": auc_pr,
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        },
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


def save_model(
    model: XGBClassifier,
    feature_names: list,
    metrics: Dict[str, Any],
    scale_pos_weight: float,
    output_dir: Path | str = "src/models/saved",
) -> Tuple[Path, Path, Path]:
    """Save trained model artifacts and metadata."""
    save_dir = Path(output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    json_model_path = save_dir / "xgb_fraud_model.json"
    joblib_model_path = save_dir / "xgb_fraud_model.joblib"
    meta_path = save_dir / "model_metadata.json"

    print(f"Saving model in JSON format: {json_model_path}...")
    model.save_model(str(json_model_path))

    print(f"Saving model in Joblib format: {joblib_model_path}...")
    joblib.dump(model, joblib_model_path)

    metadata = {
        "model_type": "XGBClassifier",
        "scale_pos_weight": scale_pos_weight,
        "feature_names": feature_names,
        "metrics": metrics,
    }

    print(f"Saving model metadata: {meta_path}...")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print("[OK] All model artifacts saved successfully.")
    return json_model_path, joblib_model_path, meta_path


def main() -> None:
    """Run end-to-end model training, evaluation, and artifact saving."""
    root_dir = Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data" / "processed"
    models_saved_dir = root_dir / "src" / "models" / "saved"

    train_df, test_df = load_datasets(data_dir)
    X_train, y_train, feature_cols = prepare_features_and_target(train_df)
    X_test, y_test, _ = prepare_features_and_target(test_df)

    model, scale_pos_weight = train_xgboost(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, feature_cols, metrics, scale_pos_weight, models_saved_dir)


if __name__ == "__main__":
    main()
