from pathlib import Path

from backend.app.data.pipeline import load_and_validate_churn_data
from backend.app.ml.workflow import run_training

DATA_PATH = Path("data/raw/churn.csv")
MODEL_PATH = Path("models/churn_model.joblib")


def main() -> None:
    """Train the churn model and display evaluation metrics."""
    df = load_and_validate_churn_data(DATA_PATH)

    metrics = run_training(df, MODEL_PATH)

    print("Model training complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 score: {metrics['f1_score']:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
