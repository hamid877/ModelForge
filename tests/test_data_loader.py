from pathlib import Path

from backend.app.data.loader import load_churn_data

DATA_PATH = Path("data/raw/churn.csv")


def test_load_churn_data():
    df = load_churn_data(DATA_PATH)

    assert len(df) == 20
    assert "churn" in df.columns
    assert "contract_type" in df.columns
