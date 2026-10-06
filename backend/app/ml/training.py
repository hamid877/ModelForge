from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from backend.app.ml.preprocessing import create_preprocessor, split_features_target


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for churn model training."""

    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 1000
    c: float = 1.0


def train_churn_model(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> Pipeline:
    """Train a logistic regression churn model."""

    X, y = split_features_target(df)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", LogisticRegression(C=config.c, max_iter=config.max_iter)),
        ]
    )

    pipeline.fit(X, y)

    return pipeline


def split_training_data(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the dataset into training and testing sets."""

    X, y = split_features_target(df)

    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )


def save_model(model: Pipeline, path: str | Path) -> None:
    """Save a trained model to disk."""

    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, path)
