from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOF_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_02_probability"
    / "oof_predictions.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_03_threshold"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_metadata"
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TARGET = "actual_failure"

MODEL_PROBABILITY_COLUMNS = {
    "Logistic Regression": "logistic_regression_probability",
    "Decision Tree": "decision_tree_probability",
    "Random Forest": "random_forest_probability",
}


# Thresholds from 0.05 to 0.95
THRESHOLDS = np.arange(
    0.05,
    1.00,
    0.05,
)


# ---------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------

def calculate_metrics(y_true, y_probability, threshold):

    # Convert probability into class prediction
    y_prediction = (
        y_probability >= threshold
    ).astype(int)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_prediction,
        labels=[0, 1],
    ).ravel()

    # Calculate classification metrics
    recall = recall_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_prediction,
        zero_division=0,
    )

    # False Alarm Rate
    far = fp / (fp + tn)

    return {
        "threshold": float(threshold),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "recall": float(recall),
        "precision": float(precision),
        "f1": float(f1),
        "far": float(far),
    }


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Load OOF predictions
    # -----------------------------------------------------

    oof_df = pd.read_csv(
        OOF_PREDICTIONS_PATH
    )

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    expected_rows = 8000

    assert len(oof_df) == expected_rows

    assert TARGET in oof_df.columns

    for model_name, probability_column in (
        MODEL_PROBABILITY_COLUMNS.items()
    ):

        assert (
            probability_column
            in oof_df.columns
        )

        assert (
            oof_df[probability_column]
            .notna()
            .all()
        )

        assert (
            (
                oof_df[probability_column]
                >= 0
            ).all()
        )

        assert (
            (
                oof_df[probability_column]
                <= 1
            ).all()
        )

    # -----------------------------------------------------
    # Extract actual target
    # -----------------------------------------------------

    y_true = oof_df[TARGET].to_numpy()

    # -----------------------------------------------------
    # Threshold analysis
    # -----------------------------------------------------

    all_results = []

    for (
        model_name,
        probability_column,
    ) in MODEL_PROBABILITY_COLUMNS.items():

        print(
            f"\nAnalyzing thresholds for: "
            f"{model_name}"
        )

        y_probability = (
            oof_df[probability_column]
            .to_numpy()
        )

        for threshold in THRESHOLDS:

            metrics = calculate_metrics(
                y_true,
                y_probability,
                threshold,
            )

            metrics["model"] = model_name

            all_results.append(
                metrics
            )

            print(
                f"  Threshold "
                f"{threshold:.2f}: "
                f"Recall="
                f"{metrics['recall']:.4f}, "
                f"Precision="
                f"{metrics['precision']:.4f}, "
                f"F1="
                f"{metrics['f1']:.4f}, "
                f"FP="
                f"{metrics['FP']}, "
                f"FN="
                f"{metrics['FN']}"
            )

    # -----------------------------------------------------
    # Create results dataframe
    # -----------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    # Put model first
    results_df = results_df[
        [
            "model",
            "threshold",
            "TN",
            "FP",
            "FN",
            "TP",
            "recall",
            "precision",
            "f1",
            "far",
        ]
    ]

    # Sort results
    results_df = results_df.sort_values(
        by=[
            "model",
            "threshold",
        ]
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save complete threshold results
    # -----------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "threshold_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # -----------------------------------------------------
    # Create model-level best-F1 summary
    # -----------------------------------------------------

    best_f1_rows = []

    for model_name in MODEL_PROBABILITY_COLUMNS:

        model_results = results_df[
            results_df["model"]
            == model_name
        ]

        best_row = model_results.loc[
            model_results["f1"].idxmax()
        ]

        best_f1_rows.append(
            best_row.to_dict()
        )

    best_f1_df = pd.DataFrame(
        best_f1_rows
    )

    best_f1_path = (
        RESULTS_DIR
        / "best_f1_thresholds.csv"
    )

    best_f1_df.to_csv(
        best_f1_path,
        index=False,
    )

    # -----------------------------------------------------
    # Create threshold 0.50 comparison
    # -----------------------------------------------------

    baseline_threshold_df = results_df[
        results_df["threshold"] == 0.50
    ].copy()

    baseline_threshold_path = (
        RESULTS_DIR
        / "threshold_0_50_comparison.csv"
    )

    baseline_threshold_df.to_csv(
        baseline_threshold_path,
        index=False,
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {

        "experiment_id":
            "experiment_03_threshold",

        "timestamp":
            datetime.now().isoformat(),

        "purpose": (
            "Evaluate the effect of decision "
            "thresholds on model performance "
            "using out-of-fold probability "
            "predictions."
        ),

        "input": {
            "file":
                str(OOF_PREDICTIONS_PATH),
            "rows":
                len(oof_df),
            "prediction_type":
                "out-of-fold probabilities",
        },

        "target": TARGET,

        "models": list(
            MODEL_PROBABILITY_COLUMNS.keys()
        ),

        "thresholds": [
            float(threshold)
            for threshold in THRESHOLDS
        ],

        "number_of_thresholds":
            len(THRESHOLDS),

        "number_of_model_threshold_combinations":
            len(MODEL_PROBABILITY_COLUMNS)
            * len(THRESHOLDS),

        "evaluation_method": (
            "Pooled evaluation over all 8,000 "
            "development-set OOF predictions."
        ),

        "metrics": [
            "recall",
            "precision",
            "f1",
            "far",
            "TN",
            "FP",
            "FN",
            "TP",
        ],

        "false_alarm_rate_definition":
            "FP / (FP + TN)",

        "final_test_used": False,

        "costs_used": False,

        "outputs": {
            "complete_results":
                str(results_path),
            "best_f1_thresholds":
                str(best_f1_path),
            "threshold_0_50_comparison":
                str(
                    baseline_threshold_path
                ),
        },

        "software": {
            "python": "3.14.2",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn":
                sklearn.__version__,
        },

        "research_observation": (
            "Changing the decision threshold "
            "changes the classification of "
            "observations and therefore changes "
            "the trade-off between failure "
            "detection and false alarms."
        ),

        "decision": (
            "Use threshold results as the basis "
            "for minimum-recall and cost-based "
            "threshold selection in subsequent "
            "experiments."
        ),
    }

    metadata_path = (
        METADATA_DIR
        / "experiment_03_threshold.json"
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

    # -----------------------------------------------------
    # Display summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT 3 COMPLETE")
    print("=" * 60)

    print(
        f"\nModels analyzed: "
        f"{len(MODEL_PROBABILITY_COLUMNS)}"
    )

    print(
        f"Thresholds analyzed: "
        f"{len(THRESHOLDS)}"
    )

    print(
        f"Model-threshold combinations: "
        f"{len(results_df)}"
    )

    print(
        "\nBest F1 threshold for each model:"
    )

    print(
        best_f1_df[
            [
                "model",
                "threshold",
                "recall",
                "precision",
                "f1",
                "far",
                "FP",
                "FN",
            ]
        ].to_string(index=False)
    )

    print(
        "\nResults saved to:"
    )

    print(results_path)

    print(
        "\nMetadata saved to:"
    )

    print(metadata_path)


if __name__ == "__main__":
    main()