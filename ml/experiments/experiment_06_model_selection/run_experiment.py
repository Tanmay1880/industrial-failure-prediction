from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

MIN_RECALL = 0.80

FP_COST = 1
FN_COST = 5

INPUT_FILE = Path(
    "experiments/experiment_05_cost/cost_results.csv"
)

OUTPUT_DIR = Path(
    "experiments/experiment_06_model_selection"
)

CANDIDATES_FILE = OUTPUT_DIR / "selection_candidates.csv"
FINAL_SELECTION_FILE = OUTPUT_DIR / "final_selection.csv"


# ============================================================
# Load Experiment 5 results
# ============================================================

results = pd.read_csv(INPUT_FILE)

print("Loaded Experiment 5 results:")
print(f"Rows: {len(results)}")


# ============================================================
# Apply minimum recall constraint
# ============================================================

feasible = results[
    results["recall"] >= MIN_RECALL
].copy()

print()
print("Feasible configurations:")
print(f"Rows: {len(feasible)}")


# ============================================================
# Calculate cost
# ============================================================

feasible["total_cost"] = (
    feasible["fp"] * FP_COST
    + feasible["fn"] * FN_COST
)


# ============================================================
# Save all feasible configurations
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

feasible.to_csv(
    CANDIDATES_FILE,
    index=False
)


# ============================================================
# Select the best configuration
# ============================================================

best = feasible.sort_values(
    by=["total_cost", "recall"],
    ascending=[True, False]
).iloc[0]

# Record the cost policy actually used for Experiment 6.
# The selected row comes from Experiment 5, so its original
# cost-scenario metadata may refer to a different scenario.
best["cost_scenario"] = "C3"
best["fp_cost"] = FP_COST
best["fn_cost"] = FN_COST

final_selection = pd.DataFrame([best])

final_selection.to_csv(
    FINAL_SELECTION_FILE,
    index=False
)


# ============================================================
# Display final decision
# ============================================================

print()
print("========================================")
print("EXPERIMENT 6 — FINAL MODEL SELECTION")
print("========================================")

print(f"Minimum recall : {MIN_RECALL:.0%}")
print(f"FP cost        : {FP_COST}")
print(f"FN cost        : {FN_COST}")

print()
print(f"Model          : {best['model']}")
print(f"Threshold      : {best['threshold']}")
print(f"Recall         : {best['recall']:.4f}")
print(f"Precision      : {best['precision']:.4f}")
print(f"F1             : {best['f1']:.4f}")
print(f"False positives: {int(best['fp'])}")
print(f"False negatives: {int(best['fn'])}")
print(f"Total cost     : {int(best['total_cost'])}")

print()
print("Saved:")
print(CANDIDATES_FILE)
print(FINAL_SELECTION_FILE)