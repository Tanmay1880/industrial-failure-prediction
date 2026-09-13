from fastapi import FastAPI

from src.api.routes import router


app = FastAPI(
    title="Industrial Failure Prediction API",
    description="ML service for industrial equipment failure prediction.",
    version="1.0.0",
)


app.include_router(router)