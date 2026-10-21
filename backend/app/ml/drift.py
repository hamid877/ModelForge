from typing import Any

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from backend.app.ml.preprocessing import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERICAL_COLUMNS,
)

DEFAULT_MIN_SAMPLES = 10
DEFAULT_P_VALUE_THRESHOLD = 0.05


def detect_data_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    p_value_threshold: float = DEFAULT_P_VALUE_THRESHOLD,
) -> dict[str, Any]:
    """Compare reference and current feature distributions."""
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")

    if not 0 < p_value_threshold < 1:
        raise ValueError("p_value_threshold must be between 0 and 1.")

    for name, frame in (("reference", reference), ("current", current)):
        missing = set(FEATURE_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} data is missing required features: {sorted(missing)}"
            )
        if frame.empty:
            raise ValueError(f"{name} data must not be empty.")
        if frame[FEATURE_COLUMNS].isnull().any().any():
            raise ValueError(f"{name} data contains missing feature values.")

    if len(reference) < min_samples or len(current) < min_samples:
        return {
            "status": "insufficient_data",
            "drift_detected": None,
            "reference_samples": len(reference),
            "current_samples": len(current),
            "min_samples": min_samples,
            "features": {},
        }

    features: dict[str, dict[str, Any]] = {}

    for column in NUMERICAL_COLUMNS:
        statistic, p_value = ks_2samp(
            reference[column],
            current[column],
            alternative="two-sided",
        )
        features[column] = {
            "test": "kolmogorov_smirnov",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < p_value_threshold),
        }

    for column in CATEGORICAL_COLUMNS:
        categories = sorted(
            set(reference[column].astype(str)) | set(current[column].astype(str))
        )
        reference_counts = reference[column].astype(str).value_counts()
        current_counts = current[column].astype(str).value_counts()

        observed = [
            [int(reference_counts.get(category, 0)) for category in categories],
            [int(current_counts.get(category, 0)) for category in categories],
        ]

        # With only one observed category, there is no meaningful
        # chi-squared comparison to perform.
        if len(categories) < 2:
            features[column] = {
                "test": "chi_squared",
                "statistic": 0.0,
                "p_value": 1.0,
                "drift_detected": False,
                "note": "Only one category is present across both samples.",
            }
            continue

        statistic, p_value, _, expected = chi2_contingency(
            observed,
            correction=False,
        )

        low_expected_counts = bool((expected < 5).any())
        features[column] = {
            "test": "chi_squared",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(p_value < p_value_threshold),
            "low_expected_counts": low_expected_counts,
        }

    drift_detected = any(feature["drift_detected"] for feature in features.values())

    return {
        "status": "drift_detected" if drift_detected else "stable",
        "drift_detected": drift_detected,
        "reference_samples": len(reference),
        "current_samples": len(current),
        "min_samples": min_samples,
        "p_value_threshold": p_value_threshold,
        "features": features,
    }
