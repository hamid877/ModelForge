import pandas as pd
import pytest

from backend.app.data.validator import validate_churn_data


def test_valid_churn_data():
    df = pd.DataFrame(
        {
            "age": [30],
            "tenure": [12],
            "monthly_charges": [75.0],
            "contract_type": ["monthly"],
            "support_tickets": [2],
            "churn": [1],
        }
    )

    validate_churn_data(df)


def test_missing_required_column():
    df = pd.DataFrame(
        {
            "age": [30],
            "tenure": [12],
            "monthly_charges": [75.0],
            "contract_type": ["monthly"],
            "support_tickets": [2],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_churn_data(df)


def test_invalid_churn_value():
    df = pd.DataFrame(
        {
            "age": [30],
            "tenure": [12],
            "monthly_charges": [75.0],
            "contract_type": ["monthly"],
            "support_tickets": [2],
            "churn": [5],
        }
    )

    with pytest.raises(ValueError, match="Churn must contain only 0 or 1"):
        validate_churn_data(df)
