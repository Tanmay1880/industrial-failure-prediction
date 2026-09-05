from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform

import numpy as np
import pandas as pd

from src.data.load_dataset import load_dataset


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

EXPERIMENT_DIR = PROJECT_ROOT / "experiments" / "experiment_00_dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "experiment_metadata"

TARGET = "Machine failure"

FAILURE_MODE_COLUMNS = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

CATEGORICAL_FEATURES = [
    "Type",
]

IDENTIFIER_COLUMNS = [
    "UDI",
    "Product ID",
]


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def calculate_file_hash(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# Experiment
# --------------------------------------------------

def main():
    print("=" * 60)
    print("EXPERIMENT 0 - DATASET INVESTIGATION")
    print("=" * 60)

    # Load dataset
    df = load_dataset(DATASET_PATH)

    print("\nDataset loaded successfully.")

    # --------------------------------------------------
    # 1. Basic dataset information
    # --------------------------------------------------

    n_rows, n_columns = df.shape

    print("\n--- Dataset shape ---")
    print(f"Rows: {n_rows}")
    print(f"Columns: {n_columns}")

    print("\n--- Columns ---")
    print(df.columns.tolist())

    print("\n--- Data types ---")
    print(df.dtypes)

    # --------------------------------------------------
    # 2. Data quality
    # --------------------------------------------------

    missing_by_column = df.isnull().sum()
    total_missing = int(missing_by_column.sum())

    duplicate_rows = int(df.duplicated().sum())

    print("\n--- Missing values ---")
    print(missing_by_column)

    print(f"\nTotal missing values: {total_missing}")

    print(f"\nDuplicate rows: {duplicate_rows}")

    # --------------------------------------------------
    # 3. Target distribution
    # --------------------------------------------------

    target_counts = df[TARGET].value_counts().sort_index()
    target_percentages = (
        df[TARGET]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    normal_count = int(target_counts.get(0, 0))
    failure_count = int(target_counts.get(1, 0))

    normal_percentage = float(target_percentages.get(0, 0))
    failure_percentage = float(target_percentages.get(1, 0))

    print("\n--- Target distribution ---")
    print(target_counts)

    print("\n--- Target percentages ---")
    print(target_percentages)

    # --------------------------------------------------
    # 4. Failure-mode investigation
    # --------------------------------------------------

    failure_mode_counts = df[FAILURE_MODE_COLUMNS].sum()

    any_failure_mode = df[FAILURE_MODE_COLUMNS].any(axis=1)

    failure_mode_crosstab = pd.crosstab(
        df[TARGET],
        any_failure_mode
    )

    disagreement_mask = df[TARGET].astype(bool) != any_failure_mode
    disagreement_count = int(disagreement_mask.sum())

    print("\n--- Failure-mode counts ---")
    print(failure_mode_counts)

    print("\n--- Machine failure vs any failure-mode indicator ---")
    print(failure_mode_crosstab)

    print(f"\nFailure-mode disagreements: {disagreement_count}")

    # --------------------------------------------------
    # 5. Identifier investigation
    # --------------------------------------------------

    print("\n--- Identifier information ---")

    for column in IDENTIFIER_COLUMNS:
        print(f"{column}:")
        print(f"  Unique values: {df[column].nunique()}")

    # --------------------------------------------------
    # 6. Categorical feature investigation
    # --------------------------------------------------

    print("\n--- Type distribution ---")
    print(df["Type"].value_counts())

    # --------------------------------------------------
    # 7. Numerical feature summary
    # --------------------------------------------------

    numerical_summary = df[NUMERICAL_FEATURES].describe()

    print("\n--- Numerical feature summary ---")
    print(numerical_summary)

    # --------------------------------------------------
    # 8. Numerical features by target
    # --------------------------------------------------

    means_by_target = df.groupby(TARGET)[NUMERICAL_FEATURES].mean()

    print("\n--- Numerical feature means by target ---")
    print(means_by_target)

    # --------------------------------------------------
    # 9. Save experiment artifacts
    # --------------------------------------------------

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset summary
    summary = pd.DataFrame({
        "property": [
            "n_rows",
            "n_columns",
            "total_missing_values",
            "duplicate_rows",
            "normal_count",
            "failure_count",
            "normal_percentage",
            "failure_percentage",
            "failure_mode_disagreements",
        ],
        "value": [
            n_rows,
            n_columns,
            total_missing,
            duplicate_rows,
            normal_count,
            failure_count,
            normal_percentage,
            failure_percentage,
            disagreement_count,
        ],
    })

    summary.to_csv(
        EXPERIMENT_DIR / "dataset_summary.csv",
        index=False
    )

    # Numerical summary
    numerical_summary.to_csv(
        EXPERIMENT_DIR / "numerical_summary.csv"
    )

    # Means by target
    means_by_target.to_csv(
        EXPERIMENT_DIR / "feature_means_by_target.csv"
    )

    # Failure mode counts
    failure_mode_counts.to_csv(
        EXPERIMENT_DIR / "failure_mode_counts.csv",
        header=["count"]
    )

    # --------------------------------------------------
    # 10. Experiment metadata
    # --------------------------------------------------

    metadata = {
        "experiment_id": "experiment_00_dataset",
        "timestamp": datetime.now().astimezone().isoformat(),
        "dataset": {
            "file": str(DATASET_PATH),
            "sha256": calculate_file_hash(DATASET_PATH),
            "rows": n_rows,
            "columns": n_columns,
        },
        "environment": {
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
        },
        "data_quality": {
            "total_missing_values": total_missing,
            "duplicate_rows": duplicate_rows,
        },
        "target": {
            "name": TARGET,
            "normal_count": normal_count,
            "failure_count": failure_count,
            "normal_percentage": normal_percentage,
            "failure_percentage": failure_percentage,
        },
        "feature_groups": {
            "identifier_columns": IDENTIFIER_COLUMNS,
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
            "failure_mode_columns": FAILURE_MODE_COLUMNS,
        },
        "failure_mode_investigation": {
            "disagreement_count": disagreement_count,
        },
        "research_observations": [
            "The dataset is highly imbalanced, with failures representing 3.39% of observations.",
            "No missing values were found.",
            "No duplicate rows were found.",
            "UDI and Product ID are identifier columns.",
            "Type is a categorical variable.",
            "The five physical/process measurements are numerical candidate predictors.",
            "Failure-mode indicators were investigated separately because they are directly related to failure information.",
            "The combined failure-mode indicators do not exactly match the Machine failure target in all rows.",
        ],
    }

    with open(
        RESULTS_DIR / "experiment_00_dataset.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\n--- Experiment 0 artifacts saved ---")
    print(EXPERIMENT_DIR)
    print(RESULTS_DIR)

    print("\nExperiment 0 investigation completed.")


if __name__ == "__main__":
    main()