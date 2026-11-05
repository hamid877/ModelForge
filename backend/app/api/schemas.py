from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class ChurnPredictionRequest(BaseModel):
    """Customer features required for churn prediction."""

    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0)
    monthly_charges: float = Field(gt=0)
    contract_type: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ]
    support_tickets: int = Field(ge=0)


class DriftBatchRequest(BaseModel):
    """Customer feature records used for batch drift detection."""

    records: list[ChurnPredictionRequest] = Field(
        min_length=1,
        max_length=10_000,
    )


class PromotionAuditRecord(BaseModel):
    """Persisted record of a model promotion evaluation."""

    audit_id: str
    timestamp: str
    model_name: str
    candidate_version: str | None = None
    candidate_run_id: str | None = None
    champion_version: str | None = None
    approved: bool
    reason: str
    candidate_f1: float
    champion_f1: float | None = None
    benchmark_version: str
    benchmark_digest: str
    margin: float
    payload: dict[str, Any]


class DriftAssessRequest(BaseModel):
    """Request body for the drift-assess-and-queue endpoint.

    ``POST /monitoring/drift/assess`` accepts the same customer feature records
    as ``POST /monitoring/drift`` but additionally evaluates retraining
    eligibility policy and, when appropriate, enqueues a retraining job.
    """

    records: list[ChurnPredictionRequest] = Field(
        min_length=1,
        max_length=10_000,
    )
    dataset_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(
        description=(
            "Dataset identifier forwarded to the retraining job when one is queued."
        ),
    )
    trigger_source: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1)
    ] = Field(
        default="drift_assess",
        description="Source label recorded on the retraining job.",
    )
    alert_id: str | None = Field(
        default=None,
        description="Optional drift alert ID to link to the new retraining job.",
    )


class QueuedJobSummary(BaseModel):
    """Minimal summary of a newly queued retraining job."""

    job_id: str
    status: str
    dataset_id: str
    trigger_source: str
    created_at: str
    alert_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_job_dict(cls, data: Any) -> Any:
        """Accept a raw job dict or a mapping with the expected keys."""
        return data


class DriftAssessmentResponse(BaseModel):
    """Response from ``POST /monitoring/drift/assess``.

    Combines the drift detection outcome with the retraining policy decision.

    ``status`` values:
    - ``'insufficient_data'`` – batch too small; policy not evaluated; no job queued.
    - ``'stable'`` – no drift detected; no job queued.
    - ``'active_job_exists'`` – drift detected but an active job already exists.
    - ``'cooldown_active'`` – drift detected but cooldown has not elapsed.
    - ``'queued'`` – drift detected and a retraining job was successfully queued.
    """

    status: str
    retraining_recommended: bool
    drift_detected: bool | None = None
    drift_report: dict[str, Any]
    job: QueuedJobSummary | None = None
    active_job_id: str | None = None
    alert_id: str | None = None
    cooldown_remaining_seconds: int | None = None
    reason: str


class RetrainingJobResponse(BaseModel):
    """Public read-only representation of a retraining job.

    Excludes worker lease tokens, worker IDs, lease timestamps, and raw payloads.
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    status: str
    is_active: bool
    dataset_id: str
    trigger_source: str
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    attempt_count: int
    max_attempts: int
    candidate_version: str | None = None
    candidate_run_id: str | None = None
    previous_champion_version: str | None = None
    promotion_started_at: str | None = None
    promotion_approved: bool | None = None
    promotion_reason: str | None = None
    promotion_audit_id: str | None = None
    failure_reason: str | None = None
    alert_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _extract_public_fields(cls, data: Any) -> Any:
        if isinstance(data, Mapping):
            required_keys = (
                "job_id",
                "status",
                "dataset_id",
                "trigger_source",
                "created_at",
            )
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"Missing required job field: {key}")

            raw_is_active = data.get("is_active")
            is_active_val = bool(raw_is_active) if raw_is_active is not None else False

            return {
                "job_id": str(data["job_id"]),
                "status": str(data["status"]),
                "is_active": is_active_val,
                "dataset_id": str(data["dataset_id"]),
                "trigger_source": str(data["trigger_source"]),
                "created_at": str(data["created_at"]),
                "started_at": data.get("started_at"),
                "completed_at": data.get("completed_at"),
                "attempt_count": data.get("attempt_count", 0),
                "max_attempts": data.get("max_attempts", 3),
                "candidate_version": data.get("candidate_version"),
                "candidate_run_id": data.get("candidate_run_id"),
                "previous_champion_version": data.get("previous_champion_version"),
                "promotion_started_at": data.get("promotion_started_at"),
                "promotion_approved": data.get("promotion_approved"),
                "promotion_reason": data.get("promotion_reason"),
                "promotion_audit_id": data.get("promotion_audit_id"),
                "failure_reason": data.get("failure_reason"),
                "alert_id": data.get("alert_id"),
            }
        return data


class ActiveJobResponse(BaseModel):
    """Response envelope for GET /retraining/jobs/active."""

    model_config = ConfigDict(extra="forbid")

    active: bool
    job: RetrainingJobResponse | None = None
