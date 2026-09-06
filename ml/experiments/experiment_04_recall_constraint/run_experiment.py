from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OOF_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_02_probability"
    / "oof_predictions.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent

METADATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_metadata"
)

METADATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

EXPERIMENT_ID = "experiment_04_min_recall"

MIN_RECALL_SCENARIOS = [
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]

MODEL_COLUMNS = {
    "Logistic Regression": "logistic_regression_probability",
    "Decision Tree": "decision_tree_probability",
    "Random Forest": "random_forest_probability",
}

THRESHOLDS = [
    round(value / 100, 2)
    for value in range(5, 100, 5)
]


# ---------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((y_true == 1) & (predictions == 1)).sum())
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    far = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    return {
        "threshold": threshold,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "far": far,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():
    print("Starting Experiment 4: Minimum Recall Constraint")

    # -----------------------------------------------------
    # Load OOF probabilities
    # -----------------------------------------------------

    if not OOF_PATH.exists():
        raise FileNotFoundError(
            f"OOF predictions not found: {OOF_PATH}"
        )

    oof = pd.read_csv(OOF_PATH)

    print(f"Loaded OOF predictions: {len(oof)} rows")

    # -----------------------------------------------------
    # Validate OOF data
    # -----------------------------------------------------

    required_columns = [
        "actual_failure",
        *MODEL_COLUMNS.values(),
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in oof.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if len(oof) != 8000:
        raise ValueError(
            f"Expected 8000 OOF rows, found {len(oof)}"
        )

    y_true = oof["actual_failure"]

    # -----------------------------------------------------
    # Generate threshold-level results
    # -----------------------------------------------------

    threshold_results = []

    for model_name, probability_column in MODEL_COLUMNS.items():

        probabilities = oof[probability_column]

        for threshold in THRESHOLDS:

            metrics = calculate_metrics(
                y_true,
                probabilities,
                threshold,
            )

            threshold_results.append(
                {
                    "model": model_name,
                    **metrics,
                }
            )

    threshold_results_df = pd.DataFrame(
        threshold_results
    )

    # -----------------------------------------------------
    # Apply minimum recall scenarios
    # -----------------------------------------------------

    scenario_results = []
    best_results = []

    for minimum_recall in MIN_RECALL_SCENARIOS:

        for model_name in MODEL_COLUMNS:

            model_results = threshold_results_df[
                threshold_results_df["model"] == model_name
            ].copy()

            feasible = model_results[
                model_results["recall"] >= minimum_recall
            ].copy()

            # -------------------------------------------------
            # Record every feasible threshold
            # -------------------------------------------------

            for _, row in feasible.iterrows():

                scenario_results.append(
                    {
                        "minimum_recall": minimum_recall,
                        **row.to_dict(),
                    }
                )

            # -------------------------------------------------
            # Select best feasible threshold by F1
            # -------------------------------------------------

            if feasible.empty:

                best_results.append(
                    {
                        "minimum_recall": minimum_recall,
                        "model": model_name,
                        "feasible": False,
                        "threshold": None,
                        "recall": None,
                        "precision": None,
                        "f1": None,
                        "far": None,
                        "tn": None,
                        "fp": None,
                        "fn": None,
                        "tp": None,
                    }
                )

            else:

                best_row = (
                    feasible
                    .sort_values(
                        by=["f1", "recall"],
                        ascending=[False, False],
                    )
                    .iloc[0]
                )

                best_results.append(
                    {
                        "minimum_recall": minimum_recall,
                        "model": model_name,
                        "feasible": True,
                        "threshold": best_row["threshold"],
                        "recall": best_row["recall"],
                        "precision": best_row["precision"],
                        "f1": best_row["f1"],
                        "far": best_row["far"],
                        "tn": best_row["tn"],
                        "fp": best_row["fp"],
                        "fn": best_row["fn"],
                        "tp": best_row["tp"],
                    }
                )

    scenario_results_df = pd.DataFrame(
        scenario_results
    )

    best_results_df = pd.DataFrame(
        best_results
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    threshold_results_df.to_csv(
        OUTPUT_DIR / "threshold_results_reused.csv",
        index=False,
    )

    scenario_results_df.to_csv(
        OUTPUT_DIR / "feasible_threshold_results.csv",
        index=False,
    )

    best_results_df.to_csv(
        OUTPUT_DIR / "best_feasible_choices.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    expected_models = len(MODEL_COLUMNS)
    expected_thresholds = len(THRESHOLDS)

    expected_threshold_rows = (
        expected_models * expected_thresholds
    )

    if len(threshold_results_df) != expected_threshold_rows:
        raise ValueError(
            "Unexpected number of threshold results."
        )

    for minimum_recall in MIN_RECALL_SCENARIOS:

        scenario = best_results_df[
            best_results_df["minimum_recall"]
            == minimum_recall
        ]

        if len(scenario) != expected_models:
            raise ValueError(
                f"Expected {expected_models} model results "
                f"for recall scenario {minimum_recall}."
            )

    # -----------------------------------------------------
    # Print summary
    # -----------------------------------------------------

    print("\nBest feasible choices:")
    print(
        best_results_df.to_string(index=False)
    )

    print("\nExperiment 4 COMPLETE")

    print(
        f"Threshold combinations evaluated: "
        f"{len(threshold_results_df)}"
    )

    print(
        f"Minimum recall scenarios: "
        f"{MIN_RECALL_SCENARIOS}"
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()