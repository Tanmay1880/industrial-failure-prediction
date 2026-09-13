from pydantic import BaseModel


class UserPredictionResponse(BaseModel):
    status: str
    failure_probability: float


class ModelPrediction(BaseModel):
    probability: float
    prediction: int


class AdminPredictionResponse(BaseModel):
    status: str
    failure_probability: float

    models: dict[str, ModelPrediction]

    selected_model: str
    threshold: float