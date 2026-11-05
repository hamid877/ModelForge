import pytest
from pydantic import ValidationError

from backend.app.api.schemas import (
    ActiveJobResponse,
    ChurnPredictionRequest,
    DriftAssessmentResponse,
    DriftAssessRequest,
    PromotionAuditRecord,
    QueuedJobSummary,
    RetrainingJobResponse,
)


def test_valid_prediction_request():
    request = ChurnPredictionRequest(
        age=35,
        tenure=24,
        monthly_charges=79.99,
        contract_type="month-to-month",
        support_tickets=2,
    )

    assert request.age == 35
    assert request.tenure == 24
    assert request.monthly_charges == 79.99


def test_invalid_age_is_rejected():
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(
            age=17,
            tenure=24,
            monthly_charges=79.99,
            contract_type="month-to-month",
            support_tickets=2,
        )


def test_negative_tenure_is_rejected():
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(
            age=35,
            tenure=-1,
            monthly_charges=79.99,
            contract_type="month-to-month",
            support_tickets=2,
        )


def test_zero_monthly_charges_are_rejected():
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(
            age=35,
            tenure=24,
            monthly_charges=0,
            contract_type="month-to-month",
            support_tickets=2,
        )


def test_empty_contract_type_is_rejected():
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(
            age=35,
            tenure=24,
            monthly_charges=79.99,
            contract_type="",
            support_tickets=2,
        )


def test_whitespace_only_contract_type_is_rejected():
    with pytest.raises(ValidationError):
        ChurnPredictionRequest(
            age=35,
            tenure=24,
            monthly_charges=79.99,
            contract_type="   ",
            support_tickets=2,
        )


def test_valid_promotion_audit_record():
    record = PromotionAuditRecord(
        audit_id="audit-123",
        timestamp="2026-10-28T12:00:00Z",
        model_name="churn-classifier",
        candidate_version="2",
        candidate_run_id="run-456",
        champion_version="1",
        approved=True,
        reason="candidate_meets_quality_gate",
        candidate_f1=0.85,
        champion_f1=0.80,
        benchmark_version="v1",
        benchmark_digest="sha256abc",
        margin=0.01,
        payload={"gate_outcomes": {"quality_gate": {"passed": True}}},
    )
    assert record.audit_id == "audit-123"
    assert record.approved is True
    assert record.candidate_version == "2"
    assert record.champion_version == "1"
    assert record.champion_f1 == 0.80
    assert record.payload["gate_outcomes"]["quality_gate"]["passed"] is True


def test_promotion_audit_record_nullable_fields():
    record = PromotionAuditRecord(
        audit_id="audit-123",
        timestamp="2026-10-28T12:00:00Z",
        model_name="churn-classifier",
        approved=False,
        reason="candidate_f1_below_champion",
        candidate_f1=0.75,
        benchmark_version="v1",
        benchmark_digest="sha256abc",
        margin=0.01,
        payload={},
    )
    assert record.candidate_version is None
    assert record.candidate_run_id is None
    assert record.champion_version is None
    assert record.champion_f1 is None
    assert record.approved is False


def test_promotion_audit_record_missing_required_fields():
    with pytest.raises(ValidationError):
        PromotionAuditRecord(
            audit_id="audit-123",
            # missing timestamp, model_name, approved, reason, etc.
        )


def test_valid_drift_assess_request():
    req = DriftAssessRequest(
        records=[
            ChurnPredictionRequest(
                age=35,
                tenure=24,
                monthly_charges=79.99,
                contract_type="month-to-month",
                support_tickets=2,
            )
        ],
        dataset_id="batch_2026_11",
        trigger_source="custom_trigger",
        alert_id="alert-99",
    )
    assert len(req.records) == 1
    assert req.dataset_id == "batch_2026_11"
    assert req.trigger_source == "custom_trigger"
    assert req.alert_id == "alert-99"


def test_drift_assess_request_default_trigger_source():
    req = DriftAssessRequest(
        records=[
            ChurnPredictionRequest(
                age=35,
                tenure=24,
                monthly_charges=79.99,
                contract_type="month-to-month",
                support_tickets=2,
            )
        ],
        dataset_id="batch_2026_11",
    )
    assert req.trigger_source == "drift_assess"
    assert req.alert_id is None


def test_drift_assess_request_rejects_empty_records():
    with pytest.raises(ValidationError):
        DriftAssessRequest(
            records=[],
            dataset_id="batch_2026_11",
        )


def test_drift_assess_request_rejects_empty_dataset_id():
    with pytest.raises(ValidationError):
        DriftAssessRequest(
            records=[
                ChurnPredictionRequest(
                    age=35,
                    tenure=24,
                    monthly_charges=79.99,
                    contract_type="month-to-month",
                    support_tickets=2,
                )
            ],
            dataset_id="   ",
        )


def test_valid_drift_assessment_response():
    resp = DriftAssessmentResponse(
        status="queued",
        retraining_recommended=True,
        drift_detected=True,
        drift_report={"status": "drift_detected", "drift_detected": True},
        job=QueuedJobSummary(
            job_id="job-123",
            status="queued",
            dataset_id="ds-1",
            trigger_source="drift_assess",
            created_at="2026-11-05T12:00:00Z",
            alert_id=None,
        ),
        cooldown_remaining_seconds=None,
        reason="feature_drift_detected_job_queued",
    )
    assert resp.status == "queued"
    assert resp.retraining_recommended is True
    assert resp.job is not None
    assert resp.job.job_id == "job-123"
    assert resp.cooldown_remaining_seconds is None


def test_drift_assessment_response_cooldown_active():
    resp = DriftAssessmentResponse(
        status="cooldown_active",
        retraining_recommended=False,
        drift_detected=True,
        drift_report={"status": "drift_detected", "drift_detected": True},
        job=None,
        cooldown_remaining_seconds=1200,
        reason="cooldown_active",
    )
    assert resp.status == "cooldown_active"
    assert resp.retraining_recommended is False
    assert resp.job is None
    assert resp.cooldown_remaining_seconds == 1200


def test_drift_assessment_response_active_job_exists():
    resp = DriftAssessmentResponse(
        status="active_job_exists",
        retraining_recommended=False,
        drift_detected=True,
        drift_report={"status": "drift_detected", "drift_detected": True},
        job=None,
        active_job_id="job-pre-456",
        alert_id="alert-789",
        cooldown_remaining_seconds=None,
        reason="active_job_exists",
    )
    assert resp.status == "active_job_exists"
    assert resp.retraining_recommended is False
    assert resp.job is None
    assert resp.active_job_id == "job-pre-456"
    assert resp.alert_id == "alert-789"
    assert resp.cooldown_remaining_seconds is None


def test_retraining_job_response_sanitization_and_exclusions():
    """Verify RetrainingJobResponse excludes secrets and internal fields."""
    source_dict = {
        "job_id": "job-abc-123",
        "created_at": "2026-11-05T08:00:00+00:00",
        "started_at": "2026-11-05T08:01:00+00:00",
        "completed_at": None,
        "status": "running",
        "is_active": 1,
        "dataset_id": "churn-ds",
        "trigger_source": "drift_assess",
        "worker_id": "worker-1",
        "lease_token": "lease-secret-token",
        "heartbeat_at": "2026-11-05T08:02:00+00:00",
        "lease_expires_at": "2026-11-05T08:03:00+00:00",
        "attempt_count": 1,
        "max_attempts": 3,
        "candidate_version": "v2",
        "candidate_run_id": "run-xyz",
        "previous_champion_version": "v1",
        "promotion_started_at": None,
        "promotion_approved": None,
        "promotion_reason": None,
        "promotion_audit_id": None,
        "failure_reason": None,
        "alert_id": "alert-456",
        "payload": {"arbitrary": "internal_data"},
    }
    source_copy = source_dict.copy()

    job_resp = RetrainingJobResponse.model_validate(source_dict)

    # Verify source dictionary was NOT mutated
    assert source_dict == source_copy

    # Verify public fields
    assert job_resp.job_id == "job-abc-123"
    assert job_resp.status == "running"
    assert job_resp.is_active is True
    assert job_resp.dataset_id == "churn-ds"
    assert job_resp.trigger_source == "drift_assess"
    assert job_resp.created_at == "2026-11-05T08:00:00+00:00"
    assert job_resp.started_at == "2026-11-05T08:01:00+00:00"
    assert job_resp.attempt_count == 1
    assert job_resp.max_attempts == 3
    assert job_resp.candidate_version == "v2"
    assert job_resp.alert_id == "alert-456"

    # Verify mandatory excluded fields are absent
    dumped = job_resp.model_dump()
    excluded_fields = (
        "lease_token",
        "worker_id",
        "heartbeat_at",
        "lease_expires_at",
        "payload",
    )
    for excluded in excluded_fields:
        assert excluded not in dumped


def test_retraining_job_response_missing_required_fields_fails():
    """Verify RetrainingJobResponse fails when essential fields are missing."""
    with pytest.raises(ValidationError):
        RetrainingJobResponse.model_validate(
            {
                "status": "queued",
                "dataset_id": "churn-ds",
            }
        )


def test_active_job_response_envelope():
    """Verify ActiveJobResponse envelope formatting for active and inactive states."""
    # When inactive
    resp_inactive = ActiveJobResponse(active=False, job=None)
    assert resp_inactive.active is False
    assert resp_inactive.job is None
    assert resp_inactive.model_dump() == {"active": False, "job": None}

    # When active
    job = RetrainingJobResponse.model_validate(
        {
            "job_id": "job-1",
            "status": "queued",
            "is_active": 1,
            "dataset_id": "ds",
            "trigger_source": "manual",
            "created_at": "2026-11-05T08:00:00+00:00",
        }
    )
    resp_active = ActiveJobResponse(active=True, job=job)
    assert resp_active.active is True
    assert resp_active.job is not None
    assert resp_active.job.job_id == "job-1"
