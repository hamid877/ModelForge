import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from backend.app.ml.preprocessing import (
    NUMERICAL_COLUMNS,
    create_preprocessor,
    split_features_target,
)
from backend.app.ml.training import TrainingConfig, train_churn_model


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


def test_numerical_scaling():
    df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "tenure": [5, 15, 25, 35],
            "monthly_charges": [50.0, 70.0, 90.0, 110.0],
            "contract_type": ["monthly", "annual", "monthly", "annual"],
            "support_tickets": [1, 2, 3, 4],
        }
    )

    preprocessor = create_preprocessor()
    transformed = preprocessor.fit_transform(df)

    scaler = preprocessor.named_transformers_["numerical"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_ is not None

    # First 4 columns correspond to NUMERICAL_COLUMNS
    scaled_numerical = transformed[:, : len(NUMERICAL_COLUMNS)]
    means = np.mean(scaled_numerical, axis=0)
    stds = np.std(scaled_numerical, axis=0)

    np.testing.assert_allclose(means, 0.0, atol=1e-7)
    np.testing.assert_allclose(stds, 1.0, atol=1e-7)


def test_scaling_fitted_only_on_training_data_in_pipeline():
    train_df = pd.DataFrame(
        {
            "age": [20, 30, 40, 50, 60, 70],
            "tenure": [6, 12, 18, 24, 30, 36],
            "monthly_charges": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
            "contract_type": ["monthly", "annual"] * 3,
            "support_tickets": [1, 2, 3, 4, 5, 6],
            "churn": [1, 0, 1, 0, 1, 0],
        }
    )
    test_df = pd.DataFrame(
        {
            "age": [80, 90],
            "tenure": [48, 54],
            "monthly_charges": [120.0, 130.0],
            "contract_type": ["monthly", "annual"],
            "support_tickets": [7, 8],
        }
    )

    config = TrainingConfig(random_state=42)
    pipeline = train_churn_model(train_df, config)

    preprocessor = pipeline.named_steps["preprocessor"]
    fitted_scaler = preprocessor.named_transformers_["numerical"]
    expected_train_means = train_df[NUMERICAL_COLUMNS].mean().to_numpy()

    np.testing.assert_allclose(fitted_scaler.mean_, expected_train_means)

    # Preprocessor transform on new data should not modify the fitted scaler mean_
    pipeline.named_steps["preprocessor"].transform(test_df)
    np.testing.assert_allclose(fitted_scaler.mean_, expected_train_means)
