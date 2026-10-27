import numpy as np
import pandas as pd
import pytest

from backend.app.data.validator import (
    MIN_SAMPLE_COUNT,
    REQUIRED_COLUMNS,
    VALID_CONTRACT_TYPES,
    validate_churn_data,
)


@pytest.fixture
def valid_churn_df() -> pd.DataFrame:
    """Provide a valid churn dataframe satisfying schema and split constraints."""
    return pd.DataFrame(
        {
            "age": [25, 45, 30, 50, 35, 60, 28, 40, 22, 55],
            "tenure": [12, 24, 6, 36, 18, 48, 8, 30, 4, 42],
            "monthly_charges": [
                70.0,
                85.0,
                65.0,
                90.0,
                75.0,
                95.0,
                80.0,
                60.0,
                88.0,
                55.0,
            ],
            "contract_type": [
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
            ],
            "support_tickets": [2, 1, 3, 0, 4, 1, 5, 2, 3, 0],
            "churn": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )


def test_valid_churn_data(valid_churn_df):
    validate_churn_data(valid_churn_df)


def test_missing_required_column(valid_churn_df):
    df = valid_churn_df.drop(columns=["churn"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_churn_data(df)


def test_empty_dataset():
    df = pd.DataFrame(columns=list(REQUIRED_COLUMNS))

    with pytest.raises(ValueError, match="Dataset must not be empty"):
        validate_churn_data(df)


def test_missing_values(valid_churn_df):
    df = valid_churn_df.copy()
    df.loc[0, "age"] = np.nan

    with pytest.raises(ValueError, match="Dataset contains missing values"):
        validate_churn_data(df)


@pytest.mark.parametrize(
    ("column", "bad_value", "expected_match"),
    [
        ("age", 17, "Age must be between 18 and 100"),
        ("age", 101, "Age must be between 18 and 100"),
        ("tenure", -1, "Tenure cannot be negative"),
        ("monthly_charges", 0.0, "Monthly charges must be greater than zero"),
        ("monthly_charges", -5.0, "Monthly charges must be greater than zero"),
        ("support_tickets", -1, "Support tickets cannot be negative"),
    ],
)
def test_invalid_feature_ranges(valid_churn_df, column, bad_value, expected_match):
    df = valid_churn_df.copy()
    df.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=expected_match):
        validate_churn_data(df)


@pytest.mark.parametrize(
    "invalid_contract",
    ["month-to-month", "weekly", "biannual", "12_months"],
)
def test_invalid_contract_type_category(valid_churn_df, invalid_contract):
    df = valid_churn_df.copy()
    df.loc[0, "contract_type"] = invalid_contract

    with pytest.raises(ValueError, match="Invalid contract_type values found"):
        validate_churn_data(df)


def test_contract_type_error_mentions_supported_values(valid_churn_df):
    df = valid_churn_df.copy()
    df.loc[0, "contract_type"] = "unsupported"

    with pytest.raises(ValueError) as exc_info:
        validate_churn_data(df)

    assert "unsupported" in str(exc_info.value)
    for supported in VALID_CONTRACT_TYPES:
        assert supported in str(exc_info.value)


def test_invalid_column_types(valid_churn_df):
    df_str_age = valid_churn_df.copy()
    df_str_age["age"] = ["twenty"] * len(df_str_age)
    with pytest.raises(ValueError, match="Column 'age' must be numeric"):
        validate_churn_data(df_str_age)

    df_bool_tenure = valid_churn_df.copy()
    df_bool_tenure["tenure"] = [True] * len(df_bool_tenure)
    with pytest.raises(ValueError, match="Column 'tenure' must be numeric"):
        validate_churn_data(df_bool_tenure)

    df_num_contract = valid_churn_df.copy()
    df_num_contract["contract_type"] = [1] * len(df_num_contract)
    with pytest.raises(
        ValueError, match="Column 'contract_type' must be string/categorical"
    ):
        validate_churn_data(df_num_contract)


@pytest.mark.parametrize("invalid_target", [5, -1, 2, 99])
def test_invalid_target_values(valid_churn_df, invalid_target):
    df = valid_churn_df.copy()
    df.loc[0, "churn"] = invalid_target

    with pytest.raises(ValueError, match="Churn must contain only 0 or 1"):
        validate_churn_data(df)


def test_missing_classes_all_zeros(valid_churn_df):
    df = valid_churn_df.copy()
    df["churn"] = 0

    with pytest.raises(
        ValueError, match=r"Dataset must contain both classes \(0 and 1\) in 'churn'"
    ):
        validate_churn_data(df)


def test_missing_classes_all_ones(valid_churn_df):
    df = valid_churn_df.copy()
    df["churn"] = 1

    with pytest.raises(
        ValueError, match=r"Dataset must contain both classes \(0 and 1\) in 'churn'"
    ):
        validate_churn_data(df)


def test_undersized_dataset(valid_churn_df):
    df = valid_churn_df.iloc[: MIN_SAMPLE_COUNT - 1]

    with pytest.raises(
        ValueError,
        match=f"Dataset must contain at least {MIN_SAMPLE_COUNT} samples",
    ):
        validate_churn_data(df)


def test_underrepresented_class(valid_churn_df):
    df = valid_churn_df.copy()
    df["churn"] = 0
    df.loc[0, "churn"] = 1

    with pytest.raises(
        ValueError,
        match=r"Each class in 'churn' must have at least 2 samples",
    ):
        validate_churn_data(df)
