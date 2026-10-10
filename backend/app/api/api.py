from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from backend.app.api.schemas import ChurnPredictionRequest
from backend.app.ml.registry import load_registered_model_by_alias


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once during application startup."""
    app.state.churn_model = load_registered_model_by_alias(
        "churn-classifier",
        "champion",
    )
    yield
    app.state.churn_model = None


app = FastAPI(
    title="ModelForge API",
    description="Customer churn prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API health status."""
    return {"status": "healthy"}


@app.post("/predict")
def predict_churn(
    request: ChurnPredictionRequest,
    api_request: Request,
) -> dict[str, int]:
    """Predict whether a customer will churn."""
    try:
        model = api_request.app.state.churn_model
        customer = pd.DataFrame([request.model_dump()])
        prediction = int(model.predict(customer)[0])

        return {"churn_prediction": prediction}
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The churn prediction service is unavailable.",
        ) from exc
