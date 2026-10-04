from pathlib import Path

import pandas as pd


def load_churn_data(path: str | Path) -> pd.DataFrame:
    """Load the raw customer churn dataset."""
    return pd.read_csv(path)
