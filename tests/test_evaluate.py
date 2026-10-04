import pandas as pd

from backend.app.ml.evaluation import evaluate_churn_model
from backend.app.ml.training import train_churn_model


def test_evaluate_churn_model():
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

    model = train_churn_model(df)

    metrics = evaluate_churn_model(
        model,
        df.drop(columns=["churn"]),
        df["churn"],
    )

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
