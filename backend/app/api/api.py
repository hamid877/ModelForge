import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.app.api.schemas import ChurnPredictionRequest
from backend.app.config import settings
from backend.app.ml.registry import load_registered_model_by_alias

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the champion model once during application startup."""
    logger.info(
        "model_loading_started model_name=%s model_alias=%s",
        settings.model_name,
        settings.model_alias,
    )

    try:
        app.state.churn_model = load_registered_model_by_alias(
            settings.model_name,
            settings.model_alias,
        )
    except Exception:
        logger.exception(
            "model_loading_failed model_name=%s model_alias=%s",
            settings.model_name,
            settings.model_alias,
        )
        raise

    logger.info(
        "model_loading_succeeded model_name=%s model_alias=%s",
        settings.model_name,
        settings.model_alias,
    )

    yield

    app.state.churn_model = None
    logger.info("model_unloaded")


app = FastAPI(
    title="ModelForge API",
    description="Customer churn prediction API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return the API liveness status."""
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check(request: Request):
    """Report whether the API has a model available for predictions."""
    model = getattr(request.app.state, "churn_model", None)

    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )

    return {"status": "ready"}


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
    except Exception:
        logger.exception("prediction_failed")
        raise HTTPException(
            status_code=503,
            detail="The churn prediction service is unavailable.",
        ) from None
