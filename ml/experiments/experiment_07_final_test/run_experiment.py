from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

FINAL_THRESHOLD = 0.20

FP_COST = 1
FN_COST = 5

DATASET_PATH = Path("data/raw/ai4i2020.csv")

OUTPUT_DIR = Path(
    "experiments/experiment_07_final_test"
)

PREDICTIONS_FILE = OUTPUT_DIR / "final_test_predictions.csv"
RESULTS_FILE = OUTPUT_DIR / "final_test_results.csv"


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
    "Type"
]

NUMERICAL_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(DATASET_PATH)

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# Recreate the development/final-test split
# ============================================================

X_dev, X_test, y_dev, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)


# ============================================================
# Create preprocessing pipeline
# ============================================================

preprocessor = ColumnTransformer(
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


# ============================================================
# Create final Random Forest pipeline
# ============================================================

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


# ============================================================
# Train on ALL development data
# ============================================================

print("Training final Random Forest...")
print(f"Development rows: {len(X_dev)}")

model.fit(X_dev, y_dev)


# ============================================================
# Predict probabilities on untouched final test data
# ============================================================

test_probabilities = model.predict_proba(X_test)[:, 1]


# ============================================================
# Apply frozen threshold
# ============================================================

test_predictions = (
    test_probabilities >= FINAL_THRESHOLD
).astype(int)


# ============================================================
# Calculate confusion matrix
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()


# ============================================================
# Calculate metrics
# ============================================================

recall = recall_score(
    y_test,
    test_predictions,
    zero_division=0,
)

precision = precision_score(
    y_test,
    test_predictions,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_predictions,
    zero_division=0,
)

roc_auc = roc_auc_score(
    y_test,
    test_probabilities,
)

false_alarm_rate = fp / (fp + tn)

total_cost = (
    fp * FP_COST
    + fn * FN_COST
)


# ============================================================
# Save predictions
# ============================================================

test_predictions_df = X_test.copy()

test_predictions_df["actual_failure"] = y_test.values
test_predictions_df["failure_probability"] = test_probabilities
test_predictions_df["predicted_failure"] = test_predictions

test_predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False,
)


# ============================================================
# Save final results
# ============================================================

results = pd.DataFrame(
    [
        {
            "model": "Random Forest",
            "threshold": FINAL_THRESHOLD,
            "test_rows": len(X_test),
            "actual_failures": int(y_test.sum()),
            "actual_normal": int((y_test == 0).sum()),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "false_alarm_rate": false_alarm_rate,
            "roc_auc": roc_auc,
            "fp_cost": FP_COST,
            "fn_cost": FN_COST,
            "total_cost": total_cost,
        }
    ]
)

results.to_csv(
    RESULTS_FILE,
    index=False,
)


# ============================================================
# Display final results
# ============================================================

print()
print("========================================")
print("EXPERIMENT 7 — FINAL TEST EVALUATION")
print("========================================")

print(f"Model          : Random Forest")
print(f"Threshold      : {FINAL_THRESHOLD}")
print(f"Test rows      : {len(X_test)}")
print(f"Actual failures: {int(y_test.sum())}")
print()

print(f"True negatives : {tn}")
print(f"False positives: {fp}")
print(f"False negatives: {fn}")
print(f"True positives : {tp}")
print()

print(f"Recall         : {recall:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"F1             : {f1:.4f}")
print(f"False alarm rate: {false_alarm_rate:.4f}")
print(f"ROC-AUC        : {roc_auc:.4f}")
print(f"Total cost     : {total_cost}")

print()
print("Saved:")
print(PREDICTIONS_FILE)
print(RESULTS_FILE)