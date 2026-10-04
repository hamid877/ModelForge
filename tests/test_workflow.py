from pathlib import Path

from backend.app.data.pipeline import load_and_validate_churn_data
from backend.app.ml.workflow import run_training

DATA_PATH = Path("data/raw/churn.csv")


def test_run_training(tmp_path: Path):
    df = load_and_validate_churn_data(DATA_PATH)

    model_path = tmp_path / "churn_model.joblib"

    metrics = run_training(df, model_path)

    assert model_path.exists()
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
