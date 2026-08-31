"""
Data Loading, Quality Checks, and Stratified Splitting for GlassBox.

This module loads the raw credit card fraud dataset, inspects data hygiene
and missing values, and performs a stratified 80/20 train/test split to guarantee
that both subsets maintain the identical ~0.172% fraud class ratio.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path: Path | str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")
    
    print(f"Loading dataset from: {path.resolve()}...")
    df = pd.read_csv(path)
    print(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df


def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Check dataset for missing values, shape, data types, and target distribution.
    """
    print("\n" + "=" * 50)
    print("DATA QUALITY & INTEGRITY CHECKS")
    print("=" * 50)

    # 1. Missing values check
    missing_series = df.isnull().sum()
    total_missing = missing_series.sum()
    print(f"Total missing values across all columns: {total_missing}")
    if total_missing > 0:
        missing_cols = missing_series[missing_series > 0]
        print(f"Columns with missing values:\n{missing_cols}")
    else:
        print("[OK] Zero missing values detected across all columns.")

    # 2. Target distribution
    if "Class" not in df.columns:
        raise ValueError("Target column 'Class' not found in dataset.")

    class_counts = df["Class"].value_counts().to_dict()
    fraud_count = class_counts.get(1, 0)
    non_fraud_count = class_counts.get(0, 0)
    total_rows = len(df)
    fraud_ratio = (fraud_count / total_rows) * 100 if total_rows > 0 else 0

    print(f"Total Transactions: {total_rows:,}")
    print(f"Legitimate (Class 0): {non_fraud_count:,} ({100 - fraud_ratio:.4f}%)")
    print(f"Fraudulent (Class 1): {fraud_count:,} ({fraud_ratio:.4f}%)")
    print("=" * 50 + "\n")

    return {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "total_missing": int(total_missing),
        "fraud_count": fraud_count,
        "non_fraud_count": non_fraud_count,
        "fraud_ratio_pct": fraud_ratio,
    }


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform a stratified train/test split on the dataset.
    
    Stratification ensures that both the training and test sets maintain
    the exact same proportion of fraud cases as the original dataset.
    """
    print(f"Performing stratified split ({int((1 - test_size) * 100)}% train / {int(test_size * 100)}% test)...")
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    
    return train_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path | str = "data/processed",
) -> Tuple[Path, Path]:
    """Save the train and test partitions to the processed directory."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_path = out_path / "train.csv"
    test_path = out_path / "test.csv"

    print(f"Saving training set ({len(train_df):,} rows) to: {train_path}...")
    train_df.to_csv(train_path, index=False)

    print(f"Saving test set ({len(test_df):,} rows) to: {test_path}...")
    test_df.to_csv(test_path, index=False)

    print("[OK] Processed datasets saved successfully.")
    return train_path, test_path


def print_summary(
    total_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "Class",
) -> None:
    """Print a clean, comprehensive summary of the dataset splits."""
    total_len = len(total_df)
    total_fraud = int((total_df[target_col] == 1).sum())

    train_len = len(train_df)
    train_fraud = int((train_df[target_col] == 1).sum())
    train_fraud_pct = (train_fraud / train_len) * 100 if train_len > 0 else 0

    test_len = len(test_df)
    test_fraud = int((test_df[target_col] == 1).sum())
    test_fraud_pct = (test_fraud / test_len) * 100 if test_len > 0 else 0

    print("\n" + "=" * 60)
    print("                GLASSBOX DATASET SPLIT SUMMARY                ")
    print("=" * 60)
    print(f"{'Metric':<25} | {'Total':<10} | {'Train (80%)':<12} | {'Test (20%)':<10}")
    print("-" * 60)
    print(f"{'Total Rows':<25} | {total_len:<10,} | {train_len:<12,} | {test_len:<10,}")
    print(f"{'Fraud Count (Class=1)':<25} | {total_fraud:<10,} | {train_fraud:<12,} | {test_fraud:<10,}")
    print(f"{'Fraud Ratio (%)':<25} | {total_fraud / total_len * 100:<9.4f}% | {train_fraud_pct:<11.4f}% | {test_fraud_pct:<9.4f}%")
    print("=" * 60)
    print("IMPORTANT: Test set (data/processed/test.csv) is isolated and")
    print("must remain untouched until final model evaluation.")
    print("=" * 60 + "\n")


def main() -> None:
    """Run data loading, validation, stratified splitting, and persistence."""
    root_dir = Path(__file__).resolve().parents[2]
    raw_data_path = root_dir / "data" / "creditcard.csv"
    processed_dir = root_dir / "data" / "processed"

    df = load_data(raw_data_path)
    check_data_quality(df)
    train_df, test_df = stratified_split(df, target_col="Class", test_size=0.2, random_state=42)
    save_processed_data(train_df, test_df, processed_dir)
    print_summary(df, train_df, test_df, target_col="Class")


if __name__ == "__main__":
    main()
