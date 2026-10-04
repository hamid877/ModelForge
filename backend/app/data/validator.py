import pandas as pd

REQUIRED_COLUMNS = {
    "age",
    "tenure",
    "monthly_charges",
    "contract_type",
    "support_tickets",
    "churn",
}


def validate_churn_data(df: pd.DataFrame) -> None:
    """Validate the structure and values of the churn dataset."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    if not df["age"].between(18, 100).all():
        raise ValueError("Age must be between 18 and 100.")

    if not df["tenure"].ge(0).all():
        raise ValueError("Tenure cannot be negative.")

    if not df["monthly_charges"].gt(0).all():
        raise ValueError("Monthly charges must be greater than zero.")

    if not df["support_tickets"].ge(0).all():
        raise ValueError("Support tickets cannot be negative.")

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 or 1.")
