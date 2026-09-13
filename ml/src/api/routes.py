from fastapi import APIRouter

from src.api.mappers import (
    create_admin_response,
    create_user_response,
)
from src.api.schemas import (
    AdminPredictionResponse,
    UserPredictionResponse,
)
from src.prediction.predictor import Predictor
from src.prediction.schemas import MachineInput


router = APIRouter()

predictor = Predictor()


@router.get("/health")
def health():
    return {"status": "UP"}


@router.post("/predict", response_model=UserPredictionResponse)
def predict(machine: MachineInput):
    results = predictor.predict_all(machine.to_model_input())

    return create_user_response(results)


@router.post("/admin/predict", response_model=AdminPredictionResponse)
def admin_predict(machine: MachineInput):
    results = predictor.predict_all(machine.to_model_input())

    return create_admin_response(results)