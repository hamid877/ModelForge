import pandas as pd

REQUIRED_COLUMNS = {
    "age",
    "tenure",
    "monthly_charges",
    "contract_type",
    "support_tickets",
    "churn",
}

NUMERIC_COLUMNS = {
    "age",
    "tenure",
    "monthly_charges",
    "support_tickets",
    "churn",
}

VALID_CONTRACT_TYPES = {
    "monthly",
    "annual",
}

MIN_SAMPLE_COUNT = 10


def validate_churn_data(df: pd.DataFrame) -> None:
    """Validate structure, types, values, and splitting readiness of churn data."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df.empty:
        raise ValueError("Dataset must not be empty.")

    if df[list(REQUIRED_COLUMNS)].isnull().any().any():
        raise ValueError("Dataset contains missing values.")

    for col in sorted(NUMERIC_COLUMNS):
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            raise ValueError(
                f"Column '{col}' must be numeric, but got dtype '{df[col].dtype}'."
            )

    if not (
        pd.api.types.is_string_dtype(df["contract_type"])
        or pd.api.types.is_object_dtype(df["contract_type"])
    ) or pd.api.types.is_bool_dtype(df["contract_type"]):
        dtype = df["contract_type"].dtype
        raise ValueError(
            f"Column 'contract_type' must be string/categorical, but got '{dtype}'."
        )

    if not df["age"].between(18, 100).all():
        raise ValueError("Age must be between 18 and 100.")

    if not df["tenure"].ge(0).all():
        raise ValueError("Tenure cannot be negative.")

    if not df["monthly_charges"].gt(0).all():
        raise ValueError("Monthly charges must be greater than zero.")

    if not df["support_tickets"].ge(0).all():
        raise ValueError("Support tickets cannot be negative.")

    invalid_contract_types = sorted(
        {str(x) for x in df["contract_type"].unique() if x not in VALID_CONTRACT_TYPES}
    )
    if invalid_contract_types:
        raise ValueError(
            f"Invalid contract_type values found: {invalid_contract_types}. "
            f"Supported values are: {sorted(VALID_CONTRACT_TYPES)}."
        )

    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 or 1.")

    if len(df) < MIN_SAMPLE_COUNT:
        raise ValueError(
            f"Dataset must contain at least {MIN_SAMPLE_COUNT} samples for "
            f"stratified splitting, but got {len(df)}."
        )

    unique_classes = set(df["churn"].unique())
    if unique_classes != {0, 1}:
        raise ValueError(
            "Dataset must contain both classes (0 and 1) in 'churn', "
            f"but found: {sorted(unique_classes)}."
        )

    class_counts = df["churn"].value_counts()
    if (class_counts < 2).any():
        underrepresented = class_counts[class_counts < 2].to_dict()
        raise ValueError(
            "Each class in 'churn' must have at least 2 samples for "
            f"stratified splitting. Underrepresented classes: {underrepresented}."
        )
