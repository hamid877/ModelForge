from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from backend.app.ml.evaluation import evaluate_churn_model
from backend.app.ml.tracking import setup_mlflow
from backend.app.ml.training import (
    TrainingConfig,
    save_model,
    split_training_data,
    train_churn_model,
)


def run_training(
    df: pd.DataFrame,
    model_path: str | Path,
    config: TrainingConfig | None = None,
    register_model: bool = False,
) -> dict[str, float]:
    """Train, evaluate, and save a churn model."""
    config = config or TrainingConfig()

    setup_mlflow()
    run_name = f"logistic-regression-c{config.c:g}"
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model_family", "logistic_regression")
        mlflow.set_tag("dataset", "churn")
        mlflow.set_tag("stage", "experimentation")

        X_train, X_test, y_train, y_test = split_training_data(
            df,
            config,
        )

        train_df = X_train.copy()
        train_df["churn"] = y_train

        model = train_churn_model(train_df, config)

        metrics = evaluate_churn_model(model, X_test, y_test)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("test_size", config.test_size)
        mlflow.log_param("random_state", config.random_state)
        mlflow.log_param("max_iter", config.max_iter)
        mlflow.log_param("C", config.c)

        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1_score", metrics["f1_score"])

        save_model(model, model_path)

        mlflow.log_artifact(model_path)

        if register_model:
            mlflow.sklearn.log_model(
                model,
                name="churn_model",
                registered_model_name="churn-classifier",
            )
    return metrics
