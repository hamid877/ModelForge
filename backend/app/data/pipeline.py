from pathlib import Path

import pandas as pd

from backend.app.data.loader import load_churn_data
from backend.app.data.validator import validate_churn_data


def load_and_validate_churn_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the customer churn dataset."""
    df = load_churn_data(path)
    validate_churn_data(df)
    return df
