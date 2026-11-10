import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.responses import Response

from backend.app.api.schemas import (
    ActiveJobResponse,
    ChurnPredictionRequest,
    DriftAssessmentResponse,
    DriftAssessRequest,
    DriftBatchRequest,
    PromotionAuditRecord,
    RetrainingJobResponse,
)
from backend.app.api.security import (
    SlidingWindowRateLimiter,
    enforce_rate_limit,
    require_role,
)
from backend.app.config import settings
from backend.app.ml.audit import AuditDatabaseError, get_promotion_audit_records
from backend.app.ml.drift import detect_data_drift
from backend.app.ml.jobs import get_active_job, get_job
from backend.app.ml.migrations import migrate_audit_database_idempotent
from backend.app.ml.registry import load_registered_model_by_alias
from backend.app.ml.retraining import evaluate_retraining_policy
from backend.app.ml.retraining_metrics import (
    format_retraining_prometheus_metrics,
    get_retraining_metrics_snapshot,
)
from backend.app.ml.tracking import setup_mlflow

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
    app.state.prediction_latency_samples = deque(maxlen=1000)
    if getattr(app.state, "rate_limiter", None) is None:
        app.state.rate_limiter = SlidingWindowRateLimiter(
            max_requests=settings.rate_limit_requests_per_minute,
            window_seconds=settings.rate_limit_window_seconds,
        )

    # Initialize audit database schema idempotently prior to MLflow setup
    migrate_audit_database_idempotent(settings.lifecycle_audit_db_path)

    logger.info(
        "model_loading_started model_name=%s model_alias=%s",
        settings.model_name,
        settings.model_alias,
    )
    setup_mlflow()

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
    app.state.rate_limiter = None
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


def _percentile(sorted_samples: list[float], percentile: float) -> float:
    """Calculate a linearly interpolated percentile."""
    if len(sorted_samples) == 1:
        return sorted_samples[0]

    position = (len(sorted_samples) - 1) * percentile / 100
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_samples) - 1)
    fraction = position - lower_index

    return (
        sorted_samples[lower_index] * (1 - fraction)
        + sorted_samples[upper_index] * fraction
    )


@app.get("/metrics")
def prediction_metrics(request: Request) -> dict[str, int | float]:
    """Return process-local prediction metrics."""
    metrics = request.app.state.prediction_metrics.copy()
    request_count = metrics["prediction_requests_total"]

    metrics["prediction_latency_average_seconds"] = (
        metrics["prediction_latency_total_seconds"] / request_count
        if request_count > 0
        else 0.0
    )

    samples = sorted(request.app.state.prediction_latency_samples)

    if samples:
        metrics["prediction_latency_p50_seconds"] = _percentile(samples, 50)
        metrics["prediction_latency_p95_seconds"] = _percentile(samples, 95)
        metrics["prediction_latency_p99_seconds"] = _percentile(samples, 99)
    else:
        metrics["prediction_latency_p50_seconds"] = 0.0
        metrics["prediction_latency_p95_seconds"] = 0.0
        metrics["prediction_latency_p99_seconds"] = 0.0

    return metrics


@app.get("/metrics/prometheus", include_in_schema=False)
def prometheus_metrics(request: Request) -> Response:
    """Expose prediction and retraining metrics in Prometheus text exposition format."""
    metrics = prediction_metrics(request)

    lines = [
        "# HELP modelforge_prediction_requests_total Total prediction requests.",
        "# TYPE modelforge_prediction_requests_total counter",
        (
            "modelforge_prediction_requests_total "
            f"{metrics['prediction_requests_total']}"
        ),
        "# HELP modelforge_predictions_succeeded_total Successful predictions.",
        "# TYPE modelforge_predictions_succeeded_total counter",
        (
            "modelforge_predictions_succeeded_total "
            f"{metrics['predictions_succeeded_total']}"
        ),
        "# HELP modelforge_predictions_failed_total Failed predictions.",
        "# TYPE modelforge_predictions_failed_total counter",
        (f"modelforge_predictions_failed_total {metrics['predictions_failed_total']}"),
        (
            "# HELP modelforge_prediction_latency_total_seconds "
            "Total prediction latency in seconds."
        ),
        "# TYPE modelforge_prediction_latency_total_seconds counter",
        (
            "modelforge_prediction_latency_total_seconds "
            f"{metrics['prediction_latency_total_seconds']}"
        ),
        (
            "# HELP modelforge_prediction_latency_max_seconds "
            "Maximum observed prediction latency in seconds."
        ),
        "# TYPE modelforge_prediction_latency_max_seconds gauge",
        (
            "modelforge_prediction_latency_max_seconds "
            f"{metrics['prediction_latency_max_seconds']}"
        ),
        (
            "# HELP modelforge_prediction_latency_average_seconds "
            "Average prediction latency in seconds."
        ),
        "# TYPE modelforge_prediction_latency_average_seconds gauge",
        (
            "modelforge_prediction_latency_average_seconds "
            f"{metrics['prediction_latency_average_seconds']}"
        ),
        (
            "# HELP modelforge_prediction_latency_p50_seconds "
            "50th percentile of recent prediction latency."
        ),
        "# TYPE modelforge_prediction_latency_p50_seconds gauge",
        (
            "modelforge_prediction_latency_p50_seconds "
            f"{metrics['prediction_latency_p50_seconds']}"
        ),
        (
            "# HELP modelforge_prediction_latency_p95_seconds "
            "95th percentile of recent prediction latency."
        ),
        "# TYPE modelforge_prediction_latency_p95_seconds gauge",
        (
            "modelforge_prediction_latency_p95_seconds "
            f"{metrics['prediction_latency_p95_seconds']}"
        ),
        (
            "# HELP modelforge_prediction_latency_p99_seconds "
            "99th percentile of recent prediction latency."
        ),
        "# TYPE modelforge_prediction_latency_p99_seconds gauge",
        (
            "modelforge_prediction_latency_p99_seconds "
            f"{metrics['prediction_latency_p99_seconds']}"
        ),
    ]

    # Append retraining and worker orchestration metrics
    db_path = getattr(settings, "lifecycle_audit_db_path", None)
    snapshot = get_retraining_metrics_snapshot(db_path)
    lines.extend(format_retraining_prometheus_metrics(snapshot))

    return Response(
        content="\n".join(lines) + "\n",
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post(
    "/predict",
    dependencies=[Depends(require_role("predict")), Depends(enforce_rate_limit)],
)
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
        api_request.app.state.prediction_latency_samples.append(elapsed)


@app.post(
    "/monitoring/drift",
    dependencies=[Depends(require_role("monitoring")), Depends(enforce_rate_limit)],
)
def monitor_data_drift(
    batch: DriftBatchRequest,
    api_request: Request,
) -> dict:
    """Compare a customer batch with the configured reference dataset."""
    reference_path = Path(settings.drift_reference_path)
    cached_path = getattr(api_request.app.state, "drift_reference_path", None)
    reference = getattr(api_request.app.state, "drift_reference", None)

    if reference is None or cached_path != str(reference_path):
        try:
            reference = pd.read_csv(reference_path)
        except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
            logger.exception("drift_reference_load_failed")
            raise HTTPException(
                status_code=503,
                detail="Drift monitoring reference data is unavailable.",
            ) from None

        # Validate the reference before caching it. An invalid reference is
        # a server-side data problem, not a client request problem.
        try:
            current = pd.DataFrame([record.model_dump() for record in batch.records])
            detect_data_drift(
                reference,
                current,
                min_samples=settings.drift_min_samples,
                p_value_threshold=settings.drift_p_value_threshold,
            )
        except ValueError:
            logger.exception("drift_reference_validation_failed")
            raise HTTPException(
                status_code=503,
                detail="Drift monitoring reference data is invalid.",
            ) from None

        api_request.app.state.drift_reference = reference
        api_request.app.state.drift_reference_path = str(reference_path)

    current = pd.DataFrame([record.model_dump() for record in batch.records])

    try:
        return detect_data_drift(
            reference,
            current,
            min_samples=settings.drift_min_samples,
            p_value_threshold=settings.drift_p_value_threshold,
        )
    except ValueError:
        logger.exception("drift_detection_failed")
        raise HTTPException(
            status_code=422,
            detail="Drift detection could not process the supplied datasets.",
        ) from None


@app.post(
    "/monitoring/drift/assess",
    response_model=DriftAssessmentResponse,
    status_code=200,
    dependencies=[Depends(require_role("monitoring")), Depends(enforce_rate_limit)],
    responses={
        409: {
            "model": DriftAssessmentResponse,
            "description": "Active retraining job already exists in the system.",
        }
    },
)
def drift_assess_and_queue(
    request: DriftAssessRequest,
    api_request: Request,
) -> DriftAssessmentResponse | JSONResponse:
    """Evaluate drift and retraining eligibility; queue a job when appropriate.

    This endpoint extends ``POST /monitoring/drift`` with stateful retraining
    policy evaluation.  The existing ``/monitoring/drift`` endpoint is preserved
    unchanged and returns a raw drift report without any job-queue interaction.

    Policy decision sequence:
    1. Insufficient data → ``status='insufficient_data'``, no job queued.
    2. No drift detected → ``status='stable'``, alert persisted, no job queued.
    3. Active job exists → ``status='active_job_exists'``, alert persisted,
       returns HTTP 409 with ``active_job_id``, no job queued.
    4. Cooldown not elapsed → ``status='cooldown_active'``, alert persisted,
       no job queued.
    5. Otherwise → retraining job queued → ``status='queued'``.

    The drift detection, active-job check, cooldown evaluation, and job
    insertion are executed inside a single ``IMMEDIATE`` transaction in
    ``evaluate_retraining_policy``.  This prevents duplicate jobs under
    concurrent requests without introducing a competing transaction layer.

    Returns a :class:`~backend.app.api.schemas.DriftAssessmentResponse` with
    the drift report and policy outcome.  Never triggers synchronous training or
    model promotion.
    """
    # Load and validate reference data (same caching logic as /monitoring/drift).
    reference_path = Path(settings.drift_reference_path)
    cached_path = getattr(api_request.app.state, "drift_reference_path", None)
    reference = getattr(api_request.app.state, "drift_reference", None)

    if reference is None or cached_path != str(reference_path):
        try:
            reference = pd.read_csv(reference_path)
        except (OSError, ValueError, pd.errors.ParserError, UnicodeDecodeError):
            logger.exception("drift_reference_load_failed")
            raise HTTPException(
                status_code=503,
                detail="Drift monitoring reference data is unavailable.",
            ) from None

        try:
            current_probe = pd.DataFrame(
                [record.model_dump() for record in request.records]
            )
            detect_data_drift(
                reference,
                current_probe,
                min_samples=settings.drift_min_samples,
                p_value_threshold=settings.drift_p_value_threshold,
            )
        except ValueError:
            logger.exception("drift_reference_validation_failed")
            raise HTTPException(
                status_code=503,
                detail="Drift monitoring reference data is invalid.",
            ) from None

        api_request.app.state.drift_reference = reference
        api_request.app.state.drift_reference_path = str(reference_path)

    current = pd.DataFrame([record.model_dump() for record in request.records])

    try:
        policy_result = evaluate_retraining_policy(
            reference=reference,
            current=current,
            dataset_id=request.dataset_id,
            trigger_source=request.trigger_source,
            alert_id=request.alert_id,
            database_path=settings.lifecycle_audit_db_path,
        )
    except ValueError:
        logger.exception("drift_assess_policy_evaluation_failed")
        raise HTTPException(
            status_code=422,
            detail="Drift assessment could not process the supplied datasets.",
        ) from None
    except Exception:
        logger.exception("drift_assess_internal_error")
        raise HTTPException(
            status_code=503,
            detail="Drift assessment service is temporarily unavailable.",
        ) from None

    if policy_result["status"] == "active_job_exists":
        return JSONResponse(
            status_code=409,
            content=DriftAssessmentResponse.model_validate(policy_result).model_dump(
                mode="json"
            ),
        )

    return DriftAssessmentResponse.model_validate(policy_result)


@app.get(
    "/audit/promotions",
    response_model=list[PromotionAuditRecord],
    dependencies=[Depends(require_role("audit")), Depends(enforce_rate_limit)],
)
def get_promotion_audits(
    limit: Annotated[int, Query(ge=1, le=100)] = 10,
) -> list[PromotionAuditRecord]:
    """Retrieve recent model promotion audit records newest first."""
    try:
        records = get_promotion_audit_records(
            database_path=settings.promotion_audit_db_path,
            limit=limit,
        )
    except AuditDatabaseError:
        logger.exception("promotion_audit_retrieval_failed")
        raise HTTPException(
            status_code=503,
            detail="The promotion audit service is unavailable.",
        ) from None

    return [PromotionAuditRecord.model_validate(record) for record in records]


@app.get(
    "/retraining/jobs/active",
    response_model=ActiveJobResponse,
    status_code=200,
    dependencies=[Depends(require_role("retraining")), Depends(enforce_rate_limit)],
    responses={
        503: {
            "description": "Retraining job service unavailable.",
        }
    },
)
def get_active_retraining_job() -> ActiveJobResponse:
    """Retrieve the current active retraining job if one exists.

    Returns HTTP 200 with ``active=false`` and ``job=null`` when no job is active.
    Strictly read-only; does not mutate leases, heartbeats, or attempts.
    """
    try:
        active_job = get_active_job(
            database_path=settings.lifecycle_audit_db_path,
        )
    except Exception:
        logger.exception("active_retraining_job_retrieval_failed")
        raise HTTPException(
            status_code=503,
            detail="The retraining job service is unavailable.",
        ) from None

    if active_job is None:
        return ActiveJobResponse(active=False, job=None)

    return ActiveJobResponse(
        active=True,
        job=RetrainingJobResponse.model_validate(active_job),
    )


@app.get(
    "/retraining/jobs/{job_id}",
    response_model=RetrainingJobResponse,
    status_code=200,
    dependencies=[Depends(require_role("retraining")), Depends(enforce_rate_limit)],
    responses={
        404: {
            "description": "Retraining job not found.",
        },
        503: {
            "description": "Retraining job service unavailable.",
        },
    },
)
def get_retraining_job_by_id(job_id: str) -> RetrainingJobResponse:
    """Retrieve retraining job details by unique job ID.

    Returns HTTP 404 when the specified job does not exist.
    Strictly read-only; does not mutate leases, heartbeats, or attempts.
    """
    clean_job_id = job_id.strip() if job_id else ""
    if not clean_job_id:
        raise HTTPException(
            status_code=404,
            detail=f"Retraining job '{job_id}' not found.",
        )

    try:
        job = get_job(
            database_path=settings.lifecycle_audit_db_path,
            job_id=clean_job_id,
        )
    except Exception:
        logger.exception("retraining_job_by_id_retrieval_failed")
        raise HTTPException(
            status_code=503,
            detail="The retraining job service is unavailable.",
        ) from None

    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Retraining job '{clean_job_id}' not found.",
        )

    return RetrainingJobResponse.model_validate(job)
