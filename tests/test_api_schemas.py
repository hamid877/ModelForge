import pytest
from pydantic import ValidationError

from backend.app.api.schemas import (
    ChurnPredictionRequest,
    DriftAssessmentResponse,
    DriftAssessRequest,
    PromotionAuditRecord,
    QueuedJobSummary,
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
