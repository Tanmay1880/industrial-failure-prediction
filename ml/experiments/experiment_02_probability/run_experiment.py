from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.data.load_dataset import load_dataset


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ai4i2020.csv"

RESULTS_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "experiment_02_probability"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "results"
    / "experiment_metadata"
)


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Pipeline
# ---------------------------------------------------------

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
    # Load dataset
    # -----------------------------------------------------

    df = load_dataset(DATASET_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    # -----------------------------------------------------
    # Development / final-test split
    # -----------------------------------------------------

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    # -----------------------------------------------------
    # Preserve original development-set indices
    # -----------------------------------------------------

    development_indices = X_dev.index.to_numpy()

    # -----------------------------------------------------
    # Create stratified 5-fold CV
    # -----------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = create_models()

    # -----------------------------------------------------
    # Storage for OOF predictions
    # -----------------------------------------------------

    oof_predictions = pd.DataFrame(
        {
            "row_index": development_indices,
            "actual_failure": y_dev.to_numpy(),
        }
    )

    # Store which fold generated each prediction
    oof_fold = np.empty(
        len(X_dev),
        dtype=int,
    )

    # -----------------------------------------------------
    # Generate OOF probabilities
    # -----------------------------------------------------

    for model_name, model in models.items():

        print(f"\nGenerating OOF probabilities for: {model_name}")

        model_column = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
            + "_probability"
        )

        # Array that will eventually contain
        # exactly one probability for every development row
        model_oof_probability = np.empty(
            len(X_dev),
            dtype=float,
        )

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(X_dev, y_dev),
            start=1,
        ):

            X_train = X_dev.iloc[train_idx]
            X_val = X_dev.iloc[val_idx]

            y_train = y_dev.iloc[train_idx]
            y_val = y_dev.iloc[val_idx]

            # Create a fresh pipeline for this fold
            pipeline = create_pipeline(model)

            # Train only on the fold's training data
            pipeline.fit(
                X_train,
                y_train,
            )

            # Generate failure probabilities
            # for the unseen validation fold
            probabilities = pipeline.predict_proba(
                X_val
            )[:, 1]

            # Put probabilities back into their
            # original development-set positions
            model_oof_probability[val_idx] = probabilities

            # Store fold number
            oof_fold[val_idx] = fold

            print(
                f"  Fold {fold}: "
                f"train={len(train_idx)}, "
                f"validation={len(val_idx)}"
            )

        # Add this model's OOF probabilities
        oof_predictions[model_column] = model_oof_probability

    # -----------------------------------------------------
    # Add fold information
    # -----------------------------------------------------

    oof_predictions["fold"] = oof_fold

    # Sort by original development-set index
    oof_predictions = oof_predictions.sort_values(
        by="row_index"
    ).reset_index(drop=True)

    # -----------------------------------------------------
    # Save OOF predictions
    # -----------------------------------------------------

    predictions_path = (
        RESULTS_DIR
        / "oof_predictions.csv"
    )

    oof_predictions.to_csv(
        predictions_path,
        index=False,
    )

    # -----------------------------------------------------
    # Basic validation checks
    # -----------------------------------------------------

    expected_rows = len(X_dev)

    assert len(oof_predictions) == expected_rows

    assert (
        oof_predictions["row_index"].nunique()
        == expected_rows
    )

    probability_columns = [
        "logistic_regression_probability",
        "decision_tree_probability",
        "random_forest_probability",
    ]

    for column in probability_columns:

        assert (
            oof_predictions[column].notna().all()
        )

        assert (
            (oof_predictions[column] >= 0).all()
        )

        assert (
            (oof_predictions[column] <= 1).all()
        )

    assert (
        oof_predictions["fold"]
        .value_counts()
        .sort_index()
        .to_dict()
        == {
            1: 1600,
            2: 1600,
            3: 1600,
            4: 1600,
            5: 1600,
        }
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = {
        "experiment_id": "experiment_02_probability",

        "timestamp": datetime.now().isoformat(),

        "purpose": (
            "Generate out-of-fold probability predictions "
            "for the development dataset for subsequent "
            "threshold analysis."
        ),

        "dataset": {
            "path": str(DATASET_PATH),
            "rows": len(df),
            "columns": len(df.columns),
        },

        "target": TARGET,

        "features": FEATURES,

        "development_test_split": {
            "development_size": len(X_dev),
            "final_test_size": len(X_test),
            "development_fraction": 0.80,
            "final_test_fraction": 0.20,
            "stratified": True,
            "random_state": 42,
        },

        "development_target_distribution": {
            "normal": int((y_dev == 0).sum()),
            "failure": int((y_dev == 1).sum()),
        },

        "final_test_target_distribution": {
            "normal": int((y_test == 0).sum()),
            "failure": int((y_test == 1).sum()),
        },

        "cross_validation": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42,
        },

        "preprocessing": {
            "numerical": {
                "features": NUMERICAL_FEATURES,
                "method": "StandardScaler",
            },
            "categorical": {
                "features": CATEGORICAL_FEATURES,
                "method": "OneHotEncoder",
                "handle_unknown": "ignore",
            },
            "implementation": (
                "ColumnTransformer inside Pipeline"
            ),
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

        "prediction_type": (
            "Out-of-fold probability of Machine failure"
        ),

        "positive_class": 1,

        "oof_rows": len(oof_predictions),

        "output_file": str(predictions_path),

        "software": {
            "python": "3.14.2",
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },

        "research_observation": (
            "Each development observation receives one "
            "probability from each model, generated by a "
            "model that was not trained on that observation."
        ),

        "decision": (
            "Use OOF probabilities for subsequent "
            "threshold, minimum-recall, and cost experiments. "
            "Keep the final test set untouched."
        ),
    }

    metadata_path = (
        METADATA_DIR
        / "experiment_02_probability.json"
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
    # Display results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("EXPERIMENT 2 COMPLETE")
    print("=" * 60)

    print(
        f"\nOOF predictions: "
        f"{len(oof_predictions)} rows"
    )

    print("\nColumns:")
    for column in oof_predictions.columns:
        print(f"  - {column}")

    print("\nFold distribution:")
    print(
        oof_predictions["fold"]
        .value_counts()
        .sort_index()
    )

    print("\nFirst five predictions:")
    print(
        oof_predictions.head()
    )

    print(
        f"\nSaved predictions to:\n"
        f"{predictions_path}"
    )

    print(
        f"\nSaved metadata to:\n"
        f"{metadata_path}"
    )


if __name__ == "__main__":
    main()