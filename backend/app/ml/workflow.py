from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from backend.app.config import settings
from backend.app.ml.evaluation import evaluate_churn_model
from backend.app.ml.promotion import (
    PromotionDecision,
    load_benchmark_dataset,
    promote_model,
)
from backend.app.ml.provenance import (
    build_split_details,
    extract_dataset_provenance,
)
from backend.app.ml.tracking import (
    log_dataset_provenance_to_mlflow,
    setup_mlflow,
)
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

        X_train, X_test, y_train, y_test = split_training_data(df, config)

        train_df = X_train.copy()
        train_df["churn"] = y_train
        test_df = X_test.copy()
        test_df["churn"] = y_test

        split_info = build_split_details(
            train_df=train_df,
            test_df=test_df,
            test_size=config.test_size,
            random_state=config.random_state,
            stratified=True,
        )
        provenance = extract_dataset_provenance(df, split_details=split_info)
        log_dataset_provenance_to_mlflow(provenance.to_dict())

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
                registered_model_name=settings.model_name,
            )

    return metrics


def run_training_and_promote(
    df: pd.DataFrame,
    model_path: str | Path,
    config: TrainingConfig | None = None,
    *,
    benchmark_df: pd.DataFrame | None = None,
    benchmark_version: str | None = None,
) -> tuple[dict[str, float], PromotionDecision]:
    """Train, register, and evaluate candidate promotion on shared benchmark."""
    config = config or TrainingConfig()

    setup_mlflow()
    run_name = f"logistic-regression-c{config.c:g}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_family", "logistic_regression")
        mlflow.set_tag("dataset", "churn")
        mlflow.set_tag("stage", "candidate")

        X_train, X_test, y_train, y_test = split_training_data(df, config)

        train_df = X_train.copy()
        train_df["churn"] = y_train
        test_df = X_test.copy()
        test_df["churn"] = y_test

        split_info = build_split_details(
            train_df=train_df,
            test_df=test_df,
            test_size=config.test_size,
            random_state=config.random_state,
            stratified=True,
        )
        provenance = extract_dataset_provenance(df, split_details=split_info)
        log_dataset_provenance_to_mlflow(provenance.to_dict())

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

        model_info = mlflow.sklearn.log_model(
            model,
            name="churn_model",
            registered_model_name=settings.model_name,
        )

        candidate_version = model_info.registered_model_version
        if candidate_version is None:
            raise RuntimeError(
                "MLflow did not return a registered model version; "
                "promotion was not attempted."
            )

        # Resolve shared benchmark dataset if not explicitly supplied
        resolved_benchmark_df = benchmark_df
        resolved_benchmark_version = benchmark_version or settings.benchmark_version
        if (
            resolved_benchmark_df is None
            and Path(settings.benchmark_data_path).exists()
        ):
            try:
                benchmark = load_benchmark_dataset(
                    settings.benchmark_data_path,
                    version=resolved_benchmark_version,
                )
                resolved_benchmark_df = benchmark.data
                resolved_benchmark_version = benchmark.version
            except Exception:
                resolved_benchmark_df = None

        candidate_run_id = getattr(getattr(run, "info", None), "run_id", None)
        decision = promote_model(
            candidate_version=candidate_version,
            candidate_f1=metrics["f1_score"],
            candidate_training_df=df,
            candidate_model=model,
            benchmark_df=resolved_benchmark_df,
            benchmark_version=resolved_benchmark_version,
            model_name=settings.model_name,
            alias=settings.model_alias,
            candidate_run_id=candidate_run_id,
        )

    return metrics, decision
