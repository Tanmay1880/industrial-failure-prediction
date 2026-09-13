from src.utils.config import FINAL_MODEL, FINAL_THRESHOLD
from src.api.schemas import (
    AdminPredictionResponse,
    ModelPrediction,
    UserPredictionResponse,
)


def create_status(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "FAILURE"
    return "NORMAL"


def create_user_response(results: dict) -> UserPredictionResponse:
    selected_result = results[FINAL_MODEL]

    probability = selected_result["probability"]

    return UserPredictionResponse(
        status=create_status(probability, FINAL_THRESHOLD),
        failure_probability=probability,
    )


def create_admin_response(results: dict) -> AdminPredictionResponse:
    selected_result = results[FINAL_MODEL]

    probability = selected_result["probability"]

    models = {
        model_name: ModelPrediction(
            probability=result["probability"],
            prediction=result["prediction"],
        )
        for model_name, result in results.items()
    }

    return AdminPredictionResponse(
        status=create_status(probability, FINAL_THRESHOLD),
        failure_probability=probability,
        models=models,
        selected_model=FINAL_MODEL,
        threshold=FINAL_THRESHOLD,
    )