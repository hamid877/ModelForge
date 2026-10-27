import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from backend.app.ml.evaluation import evaluate_churn_model
from backend.app.ml.training import TrainingConfig, train_churn_model


class MockEstimator:
    """Mock estimator for deterministic, independently verifiable evaluation tests."""

    def __init__(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray | None = None,
        classes: np.ndarray | None = None,
        decision_scores: np.ndarray | None = None,
    ):
        self._predictions = np.asarray(predictions)
        self._probabilities = (
            np.asarray(probabilities) if probabilities is not None else None
        )
        self.classes_ = (
            np.asarray(classes)
            if classes is not None
            else np.array([0, 1])
            if probabilities is not None
            else None
        )
        self._decision_scores = (
            np.asarray(decision_scores) if decision_scores is not None else None
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._probabilities is None:
            raise AttributeError("predict_proba not supported")
        return self._probabilities

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        if self._decision_scores is None:
            raise AttributeError("decision_function not supported")
        return self._decision_scores


def test_evaluate_churn_model_held_out_samples():
    """Evaluate pipeline on held-out samples and verify exact independent metrics."""
    train_df = pd.DataFrame(
        {
            "age": [20, 22, 24, 26, 45, 50, 55, 60],
            "tenure": [2, 4, 3, 5, 24, 36, 48, 60],
            "monthly_charges": [70.0, 85.0, 90.0, 95.0, 50.0, 45.0, 40.0, 35.0],
            "contract_type": [
                "monthly",
                "monthly",
                "monthly",
                "monthly",
                "annual",
                "annual",
                "annual",
                "annual",
            ],
            "support_tickets": [4, 5, 6, 7, 0, 1, 0, 2],
            "churn": [1, 1, 1, 1, 0, 0, 0, 0],
        }
    )

    # Distinct held-out test data (never seen in training)
    test_df = pd.DataFrame(
        {
            "age": [21, 23, 48, 52],
            "tenure": [3, 4, 30, 40],
            "monthly_charges": [80.0, 92.0, 48.0, 42.0],
            "contract_type": ["monthly", "monthly", "annual", "annual"],
            "support_tickets": [5, 6, 1, 0],
            "churn": [1, 1, 0, 0],
        }
    )

    config = TrainingConfig(random_state=42)
    model = train_churn_model(train_df, config)

    X_test = test_df.drop(columns=["churn"])
    y_test = test_df["churn"]

    # Verify disjointness: none of the test samples are present in training set
    common_rows = pd.merge(train_df, test_df, how="inner")
    assert common_rows.empty

    metrics = evaluate_churn_model(model, X_test, y_test)

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Independent ground-truth calculations
    expected_tn = int(((y_test == 0) & (preds == 0)).sum())
    expected_fp = int(((y_test == 0) & (preds == 1)).sum())
    expected_fn = int(((y_test == 1) & (preds == 0)).sum())
    expected_tp = int(((y_test == 1) & (preds == 1)).sum())

    total = len(y_test)
    expected_acc = (expected_tp + expected_tn) / total
    expected_prec = (
        expected_tp / (expected_tp + expected_fp)
        if (expected_tp + expected_fp) > 0
        else 0.0
    )
    expected_rec = (
        expected_tp / (expected_tp + expected_fn)
        if (expected_tp + expected_fn) > 0
        else 0.0
    )
    expected_f1 = (
        2 * (expected_prec * expected_rec) / (expected_prec + expected_rec)
        if (expected_prec + expected_rec) > 0
        else 0.0
    )
    expected_roc_auc = float(roc_auc_score(y_test, proba))

    # Assert specific, independently verifiable values
    assert metrics["accuracy"] == pytest.approx(expected_acc)
    assert metrics["precision"] == pytest.approx(expected_prec)
    assert metrics["recall"] == pytest.approx(expected_rec)
    assert metrics["f1_score"] == pytest.approx(expected_f1)
    assert metrics["f1"] == pytest.approx(expected_f1)
    assert metrics["roc_auc"] == pytest.approx(expected_roc_auc)

    assert metrics["tn"] == expected_tn
    assert metrics["fp"] == expected_fp
    assert metrics["fn"] == expected_fn
    assert metrics["tp"] == expected_tp
    assert metrics["TN"] == expected_tn
    assert metrics["FP"] == expected_fp
    assert metrics["FN"] == expected_fn
    assert metrics["TP"] == expected_tp

    assert metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"] == total
    assert metrics["confusion_matrix"]["tn"] == expected_tn
    assert metrics["confusion_matrix"]["tp"] == expected_tp


def test_evaluate_churn_model_known_fixed_predictions():
    """Assert exact hand-calculated metrics on deterministic synthetic test set."""
    X = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    # Sample 0: true=0, pred=0 -> TN (prob_pos = 0.2)
    # Sample 1: true=0, pred=1 -> FP (prob_pos = 0.6)
    # Sample 2: true=1, pred=0 -> FN (prob_pos = 0.3)
    # Sample 3: true=1, pred=1 -> TP (prob_pos = 0.9)
    model = MockEstimator(
        predictions=np.array([0, 1, 0, 1]),
        probabilities=np.array(
            [
                [0.8, 0.2],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.1, 0.9],
            ]
        ),
        classes=np.array([0, 1]),
    )

    metrics = evaluate_churn_model(model, X, y)

    # Exact hand-calculated values:
    # TN=1, FP=1, FN=1, TP=1
    # Accuracy = (1 + 1) / 4 = 0.5
    # Precision = 1 / (1 + 1) = 0.5
    # Recall = 1 / (1 + 1) = 0.5
    # F1 = 2 * (0.5 * 0.5) / (0.5 + 0.5) = 0.5
    # ROC-AUC: pairs (0.2, 0.3) concordant, (0.2, 0.9) concordant,
    #                (0.6, 0.3) discordant, (0.6, 0.9) concordant -> 3/4 = 0.75
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["roc_auc"] == 0.75

    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tp"] == 1
    assert metrics["TN"] == 1
    assert metrics["FP"] == 1
    assert metrics["FN"] == 1
    assert metrics["TP"] == 1


def test_evaluate_churn_model_single_class_predictions_all_zeros():
    """Verify zero_division=0 handling when model predicts only negative class (0)."""
    X = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    # Model predicts only 0 for all instances
    model = MockEstimator(
        predictions=np.array([0, 0, 0, 0]),
        probabilities=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4]]),
        classes=np.array([0, 1]),
    )

    metrics = evaluate_churn_model(model, X, y)

    # TN=2, FP=0, FN=2, TP=0
    # TP + FP = 0 -> Precision is 0/0 -> 0.0 with zero_division=0
    # Recall = 0 / 2 = 0.0
    # F1 = 0.0
    # Accuracy = 2 / 4 = 0.5
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["f1"] == 0.0

    assert metrics["tn"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 2
    assert metrics["tp"] == 0


def test_evaluate_churn_model_single_class_predictions_all_ones():
    """Verify metrics when model predicts only positive class (1)."""
    X = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    # Model predicts only 1 for all instances
    model = MockEstimator(
        predictions=np.array([1, 1, 1, 1]),
        probabilities=np.array([[0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.4, 0.6]]),
        classes=np.array([0, 1]),
    )

    metrics = evaluate_churn_model(model, X, y)

    # TN=0, FP=2, FN=0, TP=2
    # Accuracy = (0 + 2) / 4 = 0.5
    # Precision = 2 / (2 + 2) = 0.5
    # Recall = 2 / (2 + 0) = 1.0
    # F1 = 2 * (0.5 * 1.0) / (0.5 + 1.0) = 2/3
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(2 / 3)

    assert metrics["tn"] == 0
    assert metrics["fp"] == 2
    assert metrics["fn"] == 0
    assert metrics["tp"] == 2


def test_evaluate_churn_model_unavailable_roc_auc_single_class_labels():
    """ROC-AUC must be explicitly None when test set contains only one class."""
    X = pd.DataFrame({"feat": [1, 2, 3]})
    # Only class 0 present in y_test
    y = pd.Series([0, 0, 0])

    model = MockEstimator(
        predictions=np.array([0, 0, 0]),
        probabilities=np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]),
        classes=np.array([0, 1]),
    )

    metrics = evaluate_churn_model(model, X, y)

    # ROC-AUC is undefined for single-class ground truth; must be explicitly None
    assert metrics["roc_auc"] is None

    # Other metrics are still valid
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["tn"] == 3
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["tp"] == 0


def test_evaluate_churn_model_unavailable_roc_auc_no_score_method():
    """ROC-AUC must be explicitly None when model does not provide scores."""
    X = pd.DataFrame({"feat": [1, 2]})
    y = pd.Series([0, 1])

    # Model without predict_proba and without decision_function
    class PredictOnlyEstimator:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([0, 1])

    model = PredictOnlyEstimator()
    metrics = evaluate_churn_model(model, X, y)

    assert metrics["roc_auc"] is None
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["tn"] == 1
    assert metrics["tp"] == 1


def test_evaluate_churn_model_unavailable_roc_auc_single_class_model():
    """ROC-AUC must be None when model was trained on only a single class."""
    X = pd.DataFrame({"feat": [1, 2]})
    y = pd.Series([0, 1])

    # Model predict_proba with only 1 column
    model = MockEstimator(
        predictions=np.array([0, 0]),
        probabilities=np.array([[1.0], [1.0]]),
        classes=np.array([0]),
    )

    metrics = evaluate_churn_model(model, X, y)
    assert metrics["roc_auc"] is None


def test_evaluate_churn_model_decision_function_roc_auc():
    """ROC-AUC computed via decision_function when predict_proba is absent."""
    X = pd.DataFrame({"feat": [1, 2, 3, 4]})
    y = pd.Series([0, 0, 1, 1])

    # Model providing decision_function instead of predict_proba
    class DecisionEstimator:
        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([0, 1, 0, 1])

        def decision_function(self, X: pd.DataFrame) -> np.ndarray:
            return np.array([-1.5, 0.5, -0.5, 2.0])

    model = DecisionEstimator()
    metrics = evaluate_churn_model(model, X, y)

    # Negative class scores: -1.5, 0.5
    # Positive class scores: -0.5, 2.0
    # Pairs: (-1.5, -0.5) concordant, (-1.5, 2.0) concordant,
    #        (0.5, -0.5) discordant, (0.5, 2.0) concordant -> 3/4 = 0.75
    assert metrics["roc_auc"] == 0.75
    assert metrics["accuracy"] == 0.5


def test_evaluate_churn_model_empty_input_raises_error():
    """Evaluating on empty dataset must raise a clear ValueError."""
    X = pd.DataFrame()
    y = pd.Series([], dtype=int)

    model = MockEstimator(predictions=np.array([]))
    with pytest.raises(
        ValueError, match="Evaluation requires at least one test sample"
    ):
        evaluate_churn_model(model, X, y)
