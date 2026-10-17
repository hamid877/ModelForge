import logging
import time
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
    app.state.prediction_metrics = {
        "prediction_requests_total": 0,
        "predictions_succeeded_total": 0,
        "predictions_failed_total": 0,
        "prediction_latency_total_seconds": 0.0,
        "prediction_latency_max_seconds": 0.0,
    }
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
app.state.prediction_metrics = {
    "prediction_requests_total": 0,
    "predictions_succeeded_total": 0,
    "predictions_failed_total": 0,
}


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


@app.get("/metrics")
def prediction_metrics(request: Request) -> dict[str, int | float]:
    """Return process-local prediction counters."""
    return request.app.state.prediction_metrics.copy()


@app.post("/predict")
def predict_churn(
    request: ChurnPredictionRequest,
    api_request: Request,
) -> dict[str, int]:
    """Predict whether a customer will churn."""
    metrics = api_request.app.state.prediction_metrics
    metrics["prediction_requests_total"] += 1
    start_time = time.perf_counter()

    try:
        model = getattr(api_request.app.state, "churn_model", None)

        if model is None:
            metrics["predictions_failed_total"] += 1
            logger.error("prediction_failed reason=model_not_loaded")
            raise HTTPException(
                status_code=503,
                detail="The churn prediction service is unavailable.",
            )

        customer = pd.DataFrame([request.model_dump()])
        predictions = model.predict(customer)

        if len(predictions) != 1:
            raise ValueError("The model must return exactly one prediction.")

        raw_prediction = predictions[0]

        if raw_prediction not in (0, 1):
            raise ValueError("The model prediction must be either 0 or 1.")

        prediction = int(raw_prediction)
        metrics["predictions_succeeded_total"] += 1
        return {"churn_prediction": prediction}

    except HTTPException:
        raise
    except Exception:
        metrics["predictions_failed_total"] += 1
        logger.exception("prediction_failed")
        raise HTTPException(
            status_code=503,
            detail="The churn prediction service is unavailable.",
        ) from None
    finally:
        elapsed = time.perf_counter() - start_time
        metrics["prediction_latency_total_seconds"] += elapsed
        metrics["prediction_latency_max_seconds"] = max(
            metrics["prediction_latency_max_seconds"],
            elapsed,
        )
