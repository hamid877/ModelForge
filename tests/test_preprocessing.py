import pandas as pd

from backend.app.ml.preprocessing import (
    create_preprocessor,
    split_features_target,
)


def test_split_features_target():
    df = pd.DataFrame(
        {
            "age": [30],
            "tenure": [12],
            "monthly_charges": [75.0],
            "contract_type": ["monthly"],
            "support_tickets": [2],
            "churn": [1],
        }
    )

    X, y = split_features_target(df)

    assert "churn" not in X.columns
    assert "churn" in y.name
    assert len(X) == 1
    assert len(y) == 1


def test_preprocessor():
    df = pd.DataFrame(
        {
            "age": [30, 40],
            "tenure": [12, 24],
            "monthly_charges": [75.0, 60.0],
            "contract_type": ["monthly", "annual"],
            "support_tickets": [2, 1],
        }
    )

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] == 6
