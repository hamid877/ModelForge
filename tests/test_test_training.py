from pathlib import Path

import pandas as pd

from backend.app.ml.training import (
    TrainingConfig,
    save_model,
    split_training_data,
    train_churn_model,
)


def test_train_churn_model():
    df = pd.DataFrame(
        {
            "age": [22, 45, 31, 52, 28, 39],
            "tenure": [3, 36, 8, 48, 6, 24],
            "monthly_charges": [79.5, 65.2, 91.7, 58.3, 85.4, 72.1],
            "contract_type": [
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
            ],
            "support_tickets": [5, 1, 4, 0, 6, 2],
            "churn": [1, 0, 1, 0, 1, 0],
        }
    )

    config = TrainingConfig()
    model = train_churn_model(df, config)

    predictions = model.predict(df.drop(columns=["churn"]))

    assert len(predictions) == len(df)
    assert set(predictions).issubset({0, 1})


def test_split_training_data():
    df = pd.DataFrame(
        {
            "age": [22, 45, 31, 52, 28, 39, 24, 61, 35, 48],
            "tenure": [3, 36, 8, 48, 6, 24, 4, 60, 12, 42],
            "monthly_charges": [
                79.5,
                65.2,
                91.7,
                58.3,
                85.4,
                72.1,
                95.8,
                54.9,
                88.6,
                63.7,
            ],
            "contract_type": [
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
                "monthly",
                "annual",
            ],
            "support_tickets": [5, 1, 4, 0, 6, 2, 7, 1, 3, 1],
            "churn": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
        }
    )
    config = TrainingConfig()
    X_train, X_test, y_train, y_test = split_training_data(df, config)

    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    assert set(y_train).issubset({0, 1})
    assert set(y_test).issubset({0, 1})


def test_save_model(tmp_path: Path):
    df = pd.DataFrame(
        {
            "age": [22, 45, 31, 52],
            "tenure": [3, 36, 8, 48],
            "monthly_charges": [79.5, 65.2, 91.7, 58.3],
            "contract_type": ["monthly", "annual", "monthly", "annual"],
            "support_tickets": [5, 1, 4, 0],
            "churn": [1, 0, 1, 0],
        }
    )
    config = TrainingConfig()
    model = train_churn_model(df, config)

    model_path = tmp_path / "churn_model.joblib"

    save_model(model, model_path)

    assert model_path.exists()
