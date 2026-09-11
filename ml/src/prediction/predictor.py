from pathlib import Path

import pandas as pd
from joblib import load

from src.utils.config import FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIRECTORY = PROJECT_ROOT / "models"


class Predictor:

    def __init__(self):
        self.models = {
            "logistic_regression": load(
                MODEL_DIRECTORY / "logistic_regression.joblib"
            ),
            "decision_tree": load(
                MODEL_DIRECTORY / "decision_tree.joblib"
            ),
            "random_forest": load(
                MODEL_DIRECTORY / "random_forest.joblib"
            ),
        }

    def predict_all(self, machine_data):
        input_data = pd.DataFrame([machine_data])[FEATURES]

        results = {}

        for model_name, model in self.models.items():
            probability = model.predict_proba(input_data)[0][1]
            prediction = model.predict(input_data)[0]

            results[model_name] = {
                "probability": float(probability),
                "prediction": int(prediction),
            }

        return results