from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


def _extract_positive_class_scores(
    model: Pipeline | Any,
    X: pd.DataFrame,
) -> np.ndarray | None:
    """Extract continuous probabilities or decision scores for positive class (1)."""
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if isinstance(proba, pd.DataFrame):
                proba = proba.to_numpy()
            if isinstance(proba, np.ndarray) and proba.ndim == 2:
                classes = getattr(model, "classes_", None)
                if classes is None and hasattr(model, "named_steps"):
                    last_step = list(model.named_steps.values())[-1]
                    classes = getattr(last_step, "classes_", None)

                if classes is not None:
                    classes_list = list(classes)
                    if len(classes_list) >= 2:
                        pos_idx = classes_list.index(1) if 1 in classes_list else 1
                        if pos_idx < proba.shape[1]:
                            return proba[:, pos_idx]
                elif proba.shape[1] >= 2:
                    return proba[:, 1]
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            decision = model.decision_function(X)
            if isinstance(decision, (pd.DataFrame, pd.Series)):
                decision = decision.to_numpy()
            if isinstance(decision, np.ndarray):
                if decision.ndim == 1:
                    return decision
                if decision.ndim == 2 and decision.shape[1] == 1:
                    return decision.ravel()
        except Exception:
            pass

    return None


def evaluate_churn_model(
    model: Pipeline | Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Evaluate a trained churn model on test data."""
    if len(y_test) == 0:
        raise ValueError("Evaluation requires at least one test sample.")

    predictions = model.predict(X_test)

    acc = float(accuracy_score(y_test, predictions))
    prec = float(precision_score(y_test, predictions, zero_division=0))
    rec = float(recall_score(y_test, predictions, zero_division=0))
    f1 = float(f1_score(y_test, predictions, zero_division=0))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    roc_auc: float | None = None
    y_test_arr = np.asarray(y_test)
    unique_classes = np.unique(y_test_arr)

    if len(unique_classes) == 2:
        scores = _extract_positive_class_scores(model, X_test)
        if scores is not None and len(scores) == len(y_test_arr):
            try:
                score_val = float(roc_auc_score(y_test_arr, scores))
                if not (np.isnan(score_val) or np.isinf(score_val)):
                    roc_auc = score_val
            except (ValueError, TypeError):
                roc_auc = None

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        },
    }
