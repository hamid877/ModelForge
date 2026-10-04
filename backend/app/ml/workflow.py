from pathlib import Path

import pandas as pd

from backend.app.ml.evaluation import evaluate_churn_model
from backend.app.ml.training import (
    save_model,
    split_training_data,
    train_churn_model,
)


def run_training(
    df: pd.DataFrame,
    model_path: str | Path,
) -> dict[str, float]:
    """Train, evaluate, and save a churn model."""

    X_train, X_test, y_train, y_test = split_training_data(df)

    train_df = X_train.copy()
    train_df["churn"] = y_train

    model = train_churn_model(train_df)

    metrics = evaluate_churn_model(model, X_test, y_test)

    save_model(model, model_path)

    return metrics
