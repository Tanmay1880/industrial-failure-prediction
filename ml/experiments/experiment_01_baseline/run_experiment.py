from pathlib import Path
import json
from datetime import datetime

import pandas as pd
import sklearn
import numpy

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data.load_dataset import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

RESULTS_DIR = PROJECT_ROOT / "experiments" / "experiment_01_baseline"
METADATA_DIR = PROJECT_ROOT / "results" / "experiment_metadata"


FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

CATEGORICAL_FEATURES = [
    "Type",
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                NUMERICAL_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        ),
    }


def create_pipeline(model):
    return Pipeline(
        steps=[
            (
                "preprocessing",
                create_preprocessor(),
            ),
            (
                "model",
                model,
            ),
        ]
    )


def main():
    print("=" * 60)
    print("EXPERIMENT 1 - BASELINE MODELS")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATASET_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    # ---------------------------------------------------------
    # 1. Development / final-test split
    # ---------------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    print("\n--- Development / Final Test Split ---")
    print(f"Development rows: {len(X_dev)}")
    print(f"Development failures: {(y_dev == 1).sum()}")
    print(f"Final test rows: {len(X_test)}")
    print(f"Final test failures: {(y_test == 1).sum()}")

    # ---------------------------------------------------------
    # 2. Stratified 5-fold cross-validation
    # ---------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    # ---------------------------------------------------------
    # 3. Create models
    # ---------------------------------------------------------

    models = create_models()

    all_results = []

    # ---------------------------------------------------------
    # 4. Evaluate every model
    # ---------------------------------------------------------

    for model_name, model in models.items():

        print("\n" + "=" * 60)
        print(f"MODEL: {model_name}")
        print("=" * 60)

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_dev, y_dev),
            start=1,
        ):

            X_train = X_dev.iloc[train_idx]
            X_val = X_dev.iloc[val_idx]

            y_train = y_dev.iloc[train_idx]
            y_val = y_dev.iloc[val_idx]

            pipeline = create_pipeline(model)

            # Fit only on this fold's training data.
            pipeline.fit(X_train, y_train)

            # Failure probability.
            y_probability = pipeline.predict_proba(X_val)[:, 1]

            # Baseline threshold.
            threshold = 0.5

            y_prediction = (
                y_probability >= threshold
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_val,
                y_prediction,
                labels=[0, 1],
            ).ravel()

            recall = recall_score(
                y_val,
                y_prediction,
                zero_division=0,
            )

            precision = precision_score(
                y_val,
                y_prediction,
                zero_division=0,
            )

            f1 = f1_score(
                y_val,
                y_prediction,
                zero_division=0,
            )

            roc_auc = roc_auc_score(
                y_val,
                y_probability,
            )

            result = {
                "model": model_name,
                "fold": fold,
                "threshold": threshold,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "recall": recall,
                "precision": precision,
                "f1": f1,
                "roc_auc": roc_auc,
            }

            all_results.append(result)

            print(f"\nFold {fold}")
            print(f"TN: {tn}")
            print(f"FP: {fp}")
            print(f"FN: {fn}")
            print(f"TP: {tp}")
            print(f"Recall: {recall:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"F1: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")

    # ---------------------------------------------------------
    # 5. Save fold-level results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(all_results)

    fold_results_path = (
        RESULTS_DIR / "baseline_fold_results.csv"
    )

    results_df.to_csv(
        fold_results_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 6. Calculate and save summary
    # ---------------------------------------------------------

    summary_df = (
        results_df
        .groupby("model")[
            [
                "recall",
                "precision",
                "f1",
                "roc_auc",
            ]
        ]
        .mean()
        .reset_index()
    )

    summary_path = (
        RESULTS_DIR / "baseline_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 60)

    print(summary_df.to_string(index=False))

    # ---------------------------------------------------------
    # 7. Save experiment metadata
    # ---------------------------------------------------------

    metadata = {
        "experiment_id": "experiment_01_baseline",
        "timestamp": datetime.now().isoformat(),

        "dataset": {
            "path": str(DATASET_PATH),
            "rows": len(df),
            "columns": len(df.columns),
        },

        "features": FEATURES,

        "target": TARGET,

        "split": {
            "development_fraction": 0.80,
            "final_test_fraction": 0.20,
            "stratified": True,
            "random_state": 42,
            "development_rows": len(X_dev),
            "development_failures": int((y_dev == 1).sum()),
            "final_test_rows": len(X_test),
            "final_test_failures": int((y_test == 1).sum()),
        },

        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        },

        "preprocessing": {
            "numerical": "StandardScaler",
            "categorical": "OneHotEncoder",
            "pipeline": True,
            "leakage_safe": True,
        },

        "models": {
            "Logistic Regression": {
                "max_iter": 1000,
                "random_state": 42,
            },
            "Decision Tree": {
                "random_state": 42,
            },
            "Random Forest": {
                "n_estimators": 100,
                "random_state": 42,
            },
        },

        "baseline_threshold": 0.5,

        "metrics": [
            "recall",
            "precision",
            "f1",
            "roc_auc",
            "TN",
            "FP",
            "FN",
            "TP",
        ],

        "software": {
            "python": "3.14.2",
            "pandas": pd.__version__,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
        },

        "observations": [
            "Decision Tree achieved the highest mean recall and F1 at threshold 0.5.",
            "Random Forest achieved the highest mean ROC-AUC and precision.",
            "Logistic Regression achieved substantially lower recall at threshold 0.5.",
            "Differences between default-threshold metrics and ROC-AUC motivate further threshold analysis.",
        ],

        "decision": (
            "No final model selected. "
            "Further experiments will investigate decision thresholds "
            "and asymmetric error costs."
        ),
    }

    metadata_path = (
        METADATA_DIR / "experiment_01_baseline.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\n--- Saved Results ---")
    print(f"Fold results: {fold_results_path}")
    print(f"Summary:      {summary_path}")
    print(f"Metadata:     {metadata_path}")


if __name__ == "__main__":
    main()