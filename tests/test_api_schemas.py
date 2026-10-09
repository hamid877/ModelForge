import pytest
from pydantic import ValidationError

from backend.app.api.schemas import ChurnPredictionRequest


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
