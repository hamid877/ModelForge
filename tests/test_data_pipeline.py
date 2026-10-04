from pathlib import Path

from backend.app.data.pipeline import load_and_validate_churn_data

DATA_PATH = Path("data/raw/churn.csv")


def test_load_and_validate_churn_data():
    df = load_and_validate_churn_data(DATA_PATH)

    assert len(df) == 20
    assert df["churn"].isin([0, 1]).all()
