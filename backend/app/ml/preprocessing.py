import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

FEATURE_COLUMNS = [
    "age",
    "tenure",
    "monthly_charges",
    "contract_type",
    "support_tickets",
]

TARGET_COLUMN = "churn"

NUMERICAL_COLUMNS = [
    "age",
    "tenure",
    "monthly_charges",
    "support_tickets",
]

CATEGORICAL_COLUMNS = [
    "contract_type",
]


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return X, y


def create_preprocessor() -> ColumnTransformer:
    """Create the preprocessing transformer for model training."""
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLUMNS,
            ),
        ],
        remainder="passthrough",
    )
