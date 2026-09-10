FEATURES = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TARGET = "Machine failure"

RANDOM_STATE = 42

MODEL_CONFIG = {
    "logistic_regression": {
        "max_iter": 1000,
        "random_state": RANDOM_STATE,
    },
    "decision_tree": {
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_STATE,
    },
}

FINAL_MODEL = "random_forest"
FINAL_THRESHOLD = 0.20
MINIMUM_RECALL = 0.80

FP_COST = 1
FN_COST = 5
COST_SCENARIO = "C3"