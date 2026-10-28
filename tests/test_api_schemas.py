import pytest
from pydantic import ValidationError

from backend.app.api.schemas import ChurnPredictionRequest, PromotionAuditRecord


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
