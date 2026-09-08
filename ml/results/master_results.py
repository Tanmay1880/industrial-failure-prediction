from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = Path("results")
EXPERIMENTS_DIR = Path("experiments")

OUTPUT_FILE = RESULTS_DIR / "master_results.csv"


# Final research policy selected in Experiment 6
FINAL_MIN_RECALL = 0.80
FINAL_FP_COST = 1
FINAL_FN_COST = 5
FINAL_COST_SCENARIO = "C3"
FINAL_THRESHOLD = 0.20
FINAL_MODEL = "Random Forest"


# Thresholds highlighted in the research story
RF_THRESHOLD_POINTS = [0.50, 0.30, 0.20, 0.10, 0.05]

# Recall constraints investigated
RECALL_CONSTRAINTS = [0.60, 0.70, 0.80, 0.90, 0.95]

# Cost scenarios investigated
COST_SCENARIOS = ["C1", "C2", "C3", "C4"]


# ============================================================
# HELPERS
# ============================================================

COLUMNS = [
    "experiment",
    "purpose",
    "result_type",
    "model",
    "threshold",
    "minimum_recall",
    "cost_scenario",
    "fp_cost",
    "fn_cost",
    "recall",
    "precision",
    "f1",
    "roc_auc",
    "far",
    "fp",
    "fn",
    "tp",
    "tn",
    "total_cost",
    "observation",
]


def make_row(
    experiment,
    purpose,
    result_type,
    observation,
    model=None,
    threshold=None,
    minimum_recall=None,
    cost_scenario=None,
    fp_cost=None,
    fn_cost=None,
    recall=None,
    precision=None,
    f1=None,
    roc_auc=None,
    far=None,
    fp=None,
    fn=None,
    tp=None,
    tn=None,
    total_cost=None,
):
    return {
        "experiment": experiment,
        "purpose": purpose,
        "result_type": result_type,
        "model": model,
        "threshold": threshold,
        "minimum_recall": minimum_recall,
        "cost_scenario": cost_scenario,
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
        "far": far,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "tn": tn,
        "total_cost": total_cost,
        "observation": observation,
    }


def require_columns(df, required_columns, filename):
    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{filename} is missing required columns: {missing}"
        )


def require_exactly_one(df, description):
    if len(df) != 1:
        raise ValueError(
            f"Expected exactly one row for {description}, "
            f"but found {len(df)}."
        )


# ============================================================
# LOAD EXPERIMENT RESULTS
# ============================================================

exp1 = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_01_baseline" / "baseline_summary.csv"
)

exp2 = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_02_probability" / "oof_predictions.csv"
)

exp3 = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_03_threshold" / "threshold_results.csv"
)

exp4 = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_04_recall_constraint"
    / "best_feasible_choices.csv"
)

exp5_cost = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_05_cost" / "best_cost_choices.csv"
)

exp5_recall_cost = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_05_cost"
    / "best_cost_choices_with_recall.csv"
)

exp6 = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_06_model_selection"
    / "final_selection.csv"
)

exp7 = pd.read_csv(
    EXPERIMENTS_DIR / "experiment_07_final_test"
    / "final_test_results.csv"
)


# ============================================================
# VALIDATE SOURCE SCHEMAS
# ============================================================

require_columns(
    exp1,
    ["model", "recall", "precision", "f1", "roc_auc"],
    "Experiment 1 baseline_summary.csv",
)

require_columns(
    exp2,
    [
        "row_index",
        "actual_failure",
        "logistic_regression_probability",
        "decision_tree_probability",
        "random_forest_probability",
        "fold",
    ],
    "Experiment 2 oof_predictions.csv",
)

require_columns(
    exp3,
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
    ],
    "Experiment 3 threshold_results.csv",
)

require_columns(
    exp4,
    [
        "minimum_recall",
        "model",
        "feasible",
        "threshold",
        "recall",
        "precision",
        "f1",
        "far",
        "tn",
        "fp",
        "fn",
        "tp",
    ],
    "Experiment 4 best_feasible_choices.csv",
)

require_columns(
    exp5_cost,
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
        "tn",
        "fp",
        "fn",
        "tp",
        "total_cost",
    ],
    "Experiment 5 best_cost_choices.csv",
)

require_columns(
    exp5_recall_cost,
    [
        "minimum_recall",
        "cost_scenario",
        "fp_cost",
        "fn_cost",
        "model",
        "threshold",
        "recall",
        "precision",
        "f1",
        "far",
        "tn",
        "fp",
        "fn",
        "tp",
        "total_cost",
        "feasible",
    ],
    "Experiment 5 best_cost_choices_with_recall.csv",
)

require_columns(
    exp6,
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
        "tn",
        "fp",
        "fn",
        "tp",
        "total_cost",
    ],
    "Experiment 6 final_selection.csv",
)

require_columns(
    exp7,
    [
        "model",
        "threshold",
        "test_rows",
        "actual_failures",
        "actual_normal",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "recall",
        "precision",
        "f1",
        "false_alarm_rate",
        "roc_auc",
        "fp_cost",
        "fn_cost",
        "total_cost",
    ],
    "Experiment 7 final_test_results.csv",
)


# ============================================================
# BUILD MASTER RESULTS
# ============================================================

rows = []


# ------------------------------------------------------------
# EXPERIMENT 1 — BASELINE
# ------------------------------------------------------------

for _, result in exp1.iterrows():

    rows.append(
        make_row(
            experiment="Experiment 1",
            purpose="Baseline model comparison at threshold 0.50",
            result_type="Baseline",
            model=result["model"],
            threshold=0.50,
            recall=result["recall"],
            precision=result["precision"],
            f1=result["f1"],
            roc_auc=result["roc_auc"],
            observation=(
                "Baseline comparison using the standard 0.50 "
                "classification threshold."
            ),
        )
    )


# ------------------------------------------------------------
# EXPERIMENT 2 — OUT-OF-FOLD PROBABILITIES
# ------------------------------------------------------------

rows.append(
    make_row(
        experiment="Experiment 2",
        purpose="Generate out-of-fold probabilities for threshold analysis",
        result_type="Methodological result",
        observation=(
            "Generated one out-of-fold probability for each of the "
            "8,000 development rows for each model. These predictions "
            "were used for threshold and cost analysis without "
            "touching the final test set."
        ),
    )
)


# ------------------------------------------------------------
# EXPERIMENT 3 — THRESHOLD ANALYSIS
# ------------------------------------------------------------

# Best F1 threshold for each model
for model in exp3["model"].unique():

    model_results = exp3[
        exp3["model"] == model
    ].copy()

    best = model_results.sort_values(
        by=["f1", "recall"],
        ascending=[False, False],
    ).iloc[0]

    rows.append(
        make_row(
            experiment="Experiment 3",
            purpose="Study the effect of classification threshold",
            result_type="Best F1 threshold",
            model=model,
            threshold=best["threshold"],
            recall=best["recall"],
            precision=best["precision"],
            f1=best["f1"],
            far=best["far"],
            fp=best["FP"],
            fn=best["FN"],
            tp=best["TP"],
            tn=best["TN"],
            observation=(
                "Threshold giving the highest F1 score for this model "
                "on pooled out-of-fold development predictions."
            ),
        )
    )


# RF threshold progression used in the main research story
rf_results = exp3[
    (exp3["model"] == FINAL_MODEL)
    & (exp3["threshold"].isin(RF_THRESHOLD_POINTS))
].copy()

for _, result in rf_results.sort_values("threshold", ascending=False).iterrows():

    rows.append(
        make_row(
            experiment="Experiment 3",
            purpose="Study recall versus false alarms as threshold changes",
            result_type="Random Forest threshold point",
            model=result["model"],
            threshold=result["threshold"],
            recall=result["recall"],
            precision=result["precision"],
            f1=result["f1"],
            far=result["far"],
            fp=result["FP"],
            fn=result["FN"],
            tp=result["TP"],
            tn=result["TN"],
            observation=(
                "Lowering the threshold increases failure detection "
                "but also increases false alarms."
            ),
        )
    )


# ------------------------------------------------------------
# EXPERIMENT 4 — MINIMUM RECALL CONSTRAINT
# ------------------------------------------------------------

for minimum_recall in RECALL_CONSTRAINTS:

    scenario = exp4[
        exp4["minimum_recall"] == minimum_recall
    ].copy()

    if scenario.empty:
        raise ValueError(
            f"No Experiment 4 results found for minimum recall "
            f"{minimum_recall}."
        )

    feasible_models = sorted(
        scenario.loc[
            scenario["feasible"] == True,
            "model",
        ].tolist()
    )

    infeasible_models = sorted(
        scenario.loc[
            scenario["feasible"] == False,
            "model",
        ].tolist()
    )

    feasible_text = (
        ", ".join(feasible_models)
        if feasible_models
        else "None"
    )

    infeasible_text = (
        ", ".join(infeasible_models)
        if infeasible_models
        else "None"
    )

    rows.append(
        make_row(
            experiment="Experiment 4",
            purpose="Determine which models can satisfy minimum recall requirements",
            result_type="Recall feasibility summary",
            minimum_recall=minimum_recall,
            observation=(
                f"Feasible models: {feasible_text}. "
                f"Infeasible models: {infeasible_text}."
            ),
        )
    )


# ------------------------------------------------------------
# EXPERIMENT 5 — COST-SENSITIVE ANALYSIS
# ------------------------------------------------------------

# best_cost_choices.csv contains one row per MODEL per COST SCENARIO.
# Therefore we independently choose the lowest-cost model/threshold
# within each cost scenario.

actual_cost_scenarios = sorted(
    exp5_cost["cost_scenario"].unique().tolist()
)

if actual_cost_scenarios != COST_SCENARIOS:
    raise ValueError(
        "Unexpected cost scenarios in Experiment 5. "
        f"Expected {COST_SCENARIOS}, "
        f"found {actual_cost_scenarios}."
    )


for scenario_name in COST_SCENARIOS:

    scenario_rows = exp5_cost[
        exp5_cost["cost_scenario"] == scenario_name
    ].copy()

    if scenario_rows.empty:
        raise ValueError(
            f"No Experiment 5 results found for {scenario_name}."
        )

    best = scenario_rows.sort_values(
        by=["total_cost", "recall"],
        ascending=[True, False],
    ).iloc[0]

    rows.append(
        make_row(
            experiment="Experiment 5",
            purpose="Study model and threshold selection under asymmetric costs",
            result_type="Cost-only winning configuration",
            model=best["model"],
            threshold=best["threshold"],
            cost_scenario=best["cost_scenario"],
            fp_cost=best["fp_cost"],
            fn_cost=best["fn_cost"],
            recall=best["recall"],
            precision=best["precision"],
            f1=best["f1"],
            far=best["far"],
            fp=best["fp"],
            fn=best["fn"],
            tp=best["tp"],
            tn=best["tn"],
            total_cost=best["total_cost"],
            observation=(
                "Lowest-cost configuration across the three models "
                "and tested thresholds for this cost scenario."
            ),
        )
    )


# Add the constrained configuration corresponding to the final policy
final_policy_candidates = exp5_recall_cost[
    (exp5_recall_cost["minimum_recall"] == FINAL_MIN_RECALL)
    & (exp5_recall_cost["cost_scenario"] == FINAL_COST_SCENARIO)
    & (exp5_recall_cost["fp_cost"] == FINAL_FP_COST)
    & (exp5_recall_cost["fn_cost"] == FINAL_FN_COST)
    & (exp5_recall_cost["feasible"] == True)
].copy()

if len(final_policy_candidates) == 0:
    raise ValueError(
        "Experiment 5 does not contain a feasible configuration "
        "for the final selection policy."
    )

best = final_policy_candidates.sort_values(
    by=["total_cost", "recall"],
    ascending=[True, False],
).iloc[0]

rows.append(
    make_row(
        experiment="Experiment 5",
        purpose="Apply recall constraint together with asymmetric cost",
        result_type="Final-policy candidate",
        model=best["model"],
        threshold=best["threshold"],
        minimum_recall=best["minimum_recall"],
        cost_scenario=best["cost_scenario"],
        fp_cost=best["fp_cost"],
        fn_cost=best["fn_cost"],
        recall=best["recall"],
        precision=best["precision"],
        f1=best["f1"],
        far=best["far"],
        fp=best["fp"],
        fn=best["fn"],
        tp=best["tp"],
        tn=best["tn"],
        total_cost=best["total_cost"],
        observation=(
            "Feasible configuration under the final operating policy: "
            "minimum recall 80% and FN cost five times FP cost."
        ),
    )
)


# ------------------------------------------------------------
# EXPERIMENT 6 — FINAL MODEL SELECTION
# ------------------------------------------------------------

require_exactly_one(
    exp6,
    "Experiment 6 final selection",
)

final_selection = exp6.iloc[0]

# Validate that Experiment 6 really represents the intended policy.
if final_selection["model"] != FINAL_MODEL:
    raise ValueError(
        f"Unexpected final model in Experiment 6: "
        f"{final_selection['model']}"
    )

if final_selection["threshold"] != FINAL_THRESHOLD:
    raise ValueError(
        f"Unexpected final threshold in Experiment 6: "
        f"{final_selection['threshold']}"
    )

if final_selection["cost_scenario"] != FINAL_COST_SCENARIO:
    raise ValueError(
        f"Unexpected cost scenario in Experiment 6: "
        f"{final_selection['cost_scenario']}"
    )

if final_selection["fp_cost"] != FINAL_FP_COST:
    raise ValueError(
        f"Unexpected FP cost in Experiment 6: "
        f"{final_selection['fp_cost']}"
    )

if final_selection["fn_cost"] != FINAL_FN_COST:
    raise ValueError(
        f"Unexpected FN cost in Experiment 6: "
        f"{final_selection['fn_cost']}"
    )

calculated_cost = (
    final_selection["fp"] * final_selection["fp_cost"]
    + final_selection["fn"] * final_selection["fn_cost"]
)

if calculated_cost != final_selection["total_cost"]:
    raise ValueError(
        "Experiment 6 total cost does not match "
        "FP * FP_cost + FN * FN_cost."
    )

rows.append(
    make_row(
        experiment="Experiment 6",
        purpose="Final model and operating-point selection",
        result_type="Final selection",
        model=final_selection["model"],
        threshold=final_selection["threshold"],
        minimum_recall=FINAL_MIN_RECALL,
        cost_scenario=final_selection["cost_scenario"],
        fp_cost=final_selection["fp_cost"],
        fn_cost=final_selection["fn_cost"],
        recall=final_selection["recall"],
        precision=final_selection["precision"],
        f1=final_selection["f1"],
        far=final_selection["far"],
        fp=final_selection["fp"],
        fn=final_selection["fn"],
        tp=final_selection["tp"],
        tn=final_selection["tn"],
        total_cost=final_selection["total_cost"],
        observation=(
            "Configuration frozen before evaluation on the untouched "
            "final test set."
        ),
    )
)


# ------------------------------------------------------------
# EXPERIMENT 7 — FINAL TEST
# ------------------------------------------------------------

require_exactly_one(
    exp7,
    "Experiment 7 final test result",
)

final_test = exp7.iloc[0]

if final_test["model"] != FINAL_MODEL:
    raise ValueError(
        f"Unexpected final-test model: {final_test['model']}"
    )

if final_test["threshold"] != FINAL_THRESHOLD:
    raise ValueError(
        f"Unexpected final-test threshold: {final_test['threshold']}"
    )

calculated_test_cost = (
    final_test["false_positives"] * final_test["fp_cost"]
    + final_test["false_negatives"] * final_test["fn_cost"]
)

if calculated_test_cost != final_test["total_cost"]:
    raise ValueError(
        "Experiment 7 total cost does not match "
        "FP * FP_cost + FN * FN_cost."
    )

rows.append(
    make_row(
        experiment="Experiment 7",
        purpose="Final evaluation on untouched test set",
        result_type="Final test result",
        model=final_test["model"],
        threshold=final_test["threshold"],
        minimum_recall=FINAL_MIN_RECALL,
        cost_scenario=FINAL_COST_SCENARIO,
        fp_cost=final_test["fp_cost"],
        fn_cost=final_test["fn_cost"],
        recall=final_test["recall"],
        precision=final_test["precision"],
        f1=final_test["f1"],
        roc_auc=final_test["roc_auc"],
        far=final_test["false_alarm_rate"],
        fp=final_test["false_positives"],
        fn=final_test["false_negatives"],
        tp=final_test["true_positives"],
        tn=final_test["true_negatives"],
        total_cost=final_test["total_cost"],
        observation=(
            "Final evaluation performed once on the untouched "
            f"{int(final_test['test_rows'])}-row test set using the "
            "frozen Random Forest model and threshold 0.20."
        ),
    )
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

master_results = pd.DataFrame(rows, columns=COLUMNS)


# ============================================================
# FINAL VALIDATION
# ============================================================

expected_row_count = 24

if len(master_results) != expected_row_count:
    raise ValueError(
        f"Expected {expected_row_count} master rows, "
        f"but created {len(master_results)}."
    )

expected_counts = {
    "Experiment 1": 3,
    "Experiment 2": 1,
    "Experiment 3": 8,
    "Experiment 4": 5,
    "Experiment 5": 5,
    "Experiment 6": 1,
    "Experiment 7": 1,
}

actual_counts = (
    master_results["experiment"]
    .value_counts()
    .sort_index()
    .to_dict()
)

if actual_counts != expected_counts:
    raise ValueError(
        "Unexpected rows by experiment.\n"
        f"Expected: {expected_counts}\n"
        f"Actual:   {actual_counts}"
    )


# Ensure final selection and final test exist exactly once.
if len(
    master_results[
        master_results["result_type"] == "Final selection"
    ]
) != 1:
    raise ValueError(
        "Master results must contain exactly one final selection."
    )

if len(
    master_results[
        master_results["result_type"] == "Final test result"
    ]
) != 1:
    raise ValueError(
        "Master results must contain exactly one final test result."
    )


# ============================================================
# SAVE
# ============================================================

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

master_results.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 40)
print("MASTER RESULTS CREATED")
print("=" * 40)

print(f"Rows    : {len(master_results)}")
print(f"Columns : {len(master_results.columns)}")
print(f"Saved   : {OUTPUT_FILE}")

print()
print("Rows by experiment:")

print(
    master_results["experiment"]
    .value_counts()
    .sort_index()
)


print()
print("=" * 40)
print("FINAL SELECTION CHECK")
print("=" * 40)

print(
    master_results[
        master_results["result_type"] == "Final selection"
    ].to_string(index=False)
)


print()
print("=" * 40)
print("FINAL TEST CHECK")
print("=" * 40)

print(
    master_results[
        master_results["result_type"] == "Final test result"
    ].to_string(index=False)
)


print()
print("Master results validation: PASSED")