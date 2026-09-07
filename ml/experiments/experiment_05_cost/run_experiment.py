import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

OOF_FILE = (
    BASE_DIR
    / "experiments"
    / "experiment_02_probability"
    / "oof_predictions.csv"
)

OUTPUT_DIR = Path(__file__).resolve().parent

THRESHOLDS = [
    round(i / 100, 2)
    for i in range(5, 100, 5)
]

MINIMUM_RECALLS = [
    0.60,
    0.70,
    0.80,
    0.90,
    0.95,
]

COST_SCENARIOS = {
    "C1": {"fp_cost": 1, "fn_cost": 1},
    "C2": {"fp_cost": 1, "fn_cost": 2},
    "C3": {"fp_cost": 1, "fn_cost": 5},
    "C4": {"fp_cost": 1, "fn_cost": 10},
}

MODEL_PROBABILITY_COLUMNS = {
    "Logistic Regression": "logistic_regression_probability",
    "Decision Tree": "decision_tree_probability",
    "Random Forest": "random_forest_probability",
}


# --------------------------------------------------
# Metric calculation
# --------------------------------------------------

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((y_true == 1) & (predictions == 1)).sum())
    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

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


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading OOF predictions...")
    df = pd.read_csv(OOF_FILE)

    y_true = df["actual_failure"]

    # ==================================================
    # PART 1
    # Cost-sensitive analysis without recall constraint
    # ==================================================

    print("\nRunning cost-only analysis...")

    results = []

    for model_name, probability_column in MODEL_PROBABILITY_COLUMNS.items():

        probabilities = df[probability_column]

        for threshold in THRESHOLDS:

            metrics = calculate_metrics(
                y_true,
                probabilities,
                threshold,
            )

            for scenario_name, costs in COST_SCENARIOS.items():

                fp_cost = costs["fp_cost"]
                fn_cost = costs["fn_cost"]

                total_cost = (
                    fp_cost * metrics["fp"]
                    + fn_cost * metrics["fn"]
                )

                results.append({
                    "cost_scenario": scenario_name,
                    "fp_cost": fp_cost,
                    "fn_cost": fn_cost,
                    "model": model_name,
                    **metrics,
                    "total_cost": total_cost,
                })

    results_df = pd.DataFrame(results)

    results_file = OUTPUT_DIR / "cost_results.csv"
    results_df.to_csv(results_file, index=False)

    # --------------------------------------------------
    # Best cost choice for each model/scenario
    # --------------------------------------------------

    best_choices = []

    for (scenario, model), group in results_df.groupby(
        ["cost_scenario", "model"]
    ):
        best = group.sort_values(
            by=["total_cost", "recall"],
            ascending=[True, False],
        ).iloc[0]

        best_choices.append(best)

    best_choices_df = pd.DataFrame(best_choices)

    best_choices_df = best_choices_df.sort_values(
        by=["cost_scenario", "total_cost", "model"]
    )

    best_file = OUTPUT_DIR / "best_cost_choices.csv"
    best_choices_df.to_csv(best_file, index=False)

    print("\nBest cost choices:")
    print(
        best_choices_df[
            [
                "cost_scenario",
                "fp_cost",
                "fn_cost",
                "model",
                "threshold",
                "recall",
                "precision",
                "f1",
                "far",
                "fp",
                "fn",
                "total_cost",
            ]
        ].to_string(index=False)
    )

    # ==================================================
    # PART 2
    # Cost-sensitive analysis with minimum recall
    # ==================================================

    print("\nRunning cost + minimum recall analysis...")

    constrained_results = []

    for minimum_recall in MINIMUM_RECALLS:

        for scenario_name, costs in COST_SCENARIOS.items():

            fp_cost = costs["fp_cost"]
            fn_cost = costs["fn_cost"]

            for model_name, probability_column in MODEL_PROBABILITY_COLUMNS.items():

                probabilities = df[probability_column]

                model_threshold_results = []

                for threshold in THRESHOLDS:

                    metrics = calculate_metrics(
                        y_true,
                        probabilities,
                        threshold,
                    )

                    total_cost = (
                        fp_cost * metrics["fp"]
                        + fn_cost * metrics["fn"]
                    )

                    model_threshold_results.append({
                        "minimum_recall": minimum_recall,
                        "cost_scenario": scenario_name,
                        "fp_cost": fp_cost,
                        "fn_cost": fn_cost,
                        "model": model_name,
                        **metrics,
                        "total_cost": total_cost,
                    })

                model_results_df = pd.DataFrame(
                    model_threshold_results
                )

                # Keep only thresholds satisfying
                # the minimum recall requirement.
                feasible = model_results_df[
                    model_results_df["recall"] >= minimum_recall
                ]

                if feasible.empty:

                    constrained_results.append({
                        "minimum_recall": minimum_recall,
                        "cost_scenario": scenario_name,
                        "fp_cost": fp_cost,
                        "fn_cost": fn_cost,
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
                        "total_cost": None,
                    })

                else:

                    # Primary criterion:
                    #   lowest total cost
                    #
                    # Tie-break:
                    #   higher recall
                    best = feasible.sort_values(
                        by=["total_cost", "recall"],
                        ascending=[True, False],
                    ).iloc[0]

                    constrained_results.append({
                        **best.to_dict(),
                        "feasible": True,
                    })

    constrained_df = pd.DataFrame(constrained_results)

    constrained_file = (
        OUTPUT_DIR
        / "best_cost_choices_with_recall.csv"
    )

    constrained_df.to_csv(
        constrained_file,
        index=False,
    )

    print("\nBest cost choices with minimum recall constraints:")

    print(
        constrained_df[
            [
                "minimum_recall",
                "cost_scenario",
                "fp_cost",
                "fn_cost",
                "model",
                "feasible",
                "threshold",
                "recall",
                "precision",
                "f1",
                "far",
                "fp",
                "fn",
                "total_cost",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------
    # Files saved
    # --------------------------------------------------

    print("\nFiles saved:")
    print(results_file)
    print(best_file)
    print(constrained_file)


if __name__ == "__main__":
    main()