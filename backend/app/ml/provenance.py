"""Dataset provenance extraction and metadata utilities."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def compute_dataframe_sha256(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hex digest of a DataFrame's contents."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


@dataclass(frozen=True)
class DatasetProvenance:
    """Provenance metadata describing dataset without sensitive row-level data."""

    dataset_sha256: str
    row_count: int
    feature_names: list[str]
    feature_count: int
    feature_types: dict[str, str]
    target_column: str | None
    target_classes: list[Any] | None
    class_distribution: dict[str, int] | None
    split_performed: bool
    split_details: dict[str, Any] | None = None
    gap_documentation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance metadata to a JSON-serializable dictionary."""
        split_dict: dict[str, Any] = {
            "split_performed": self.split_performed,
        }
        if self.split_details:
            split_dict.update(self.split_details)
        if self.gap_documentation:
            split_dict["gap_documentation"] = self.gap_documentation

        return {
            "dataset_sha256": self.dataset_sha256,
            "row_count": self.row_count,
            "feature_names": self.feature_names,
            "feature_count": self.feature_count,
            "feature_types": self.feature_types,
            "target_column": self.target_column,
            "target_classes": self.target_classes,
            "class_distribution": self.class_distribution,
            "split": split_dict,
        }


def build_split_details(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratified: bool = True,
) -> dict[str, Any]:
    """Summarize train/test split partitions without sensitive record leakage."""
    return {
        "split_performed": True,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "stratified": bool(stratified),
        "train_sha256": compute_dataframe_sha256(train_df),
        "test_sha256": compute_dataframe_sha256(test_df),
    }


def extract_dataset_provenance(
    df: pd.DataFrame,
    *,
    split_details: dict[str, Any] | None = None,
    target_column: str = "churn",
) -> DatasetProvenance:
    """Extract dataset provenance summary.

    Reports SHA-256 digest, row count, feature schema, target distribution,
    and split details. When no split was performed, explicitly documents the gap.
    Does not log sensitive row-level records.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df).__name__}.")

    dataset_sha256 = compute_dataframe_sha256(df)
    row_count = int(len(df))
    has_target = target_column in df.columns
    feature_names = sorted([col for col in df.columns if col != target_column])
    feature_types = {col: str(df[col].dtype) for col in feature_names}

    target_classes = None
    class_distribution = None
    if has_target:
        unique_vals = df[target_column].dropna().unique()
        target_classes = sorted(
            [int(x) if isinstance(x, (int, np.integer)) else x for x in unique_vals]
        )
        class_distribution = {
            str(k): int(v)
            for k, v in df[target_column].value_counts().to_dict().items()
        }

    split_performed = split_details is not None and bool(
        split_details.get("split_performed", False)
    )
    gap_documentation = None
    if not split_performed:
        gap_documentation = (
            "Current training pipeline was executed without a caller train/test split; "
            "model was trained directly or split downstream."
        )

    return DatasetProvenance(
        dataset_sha256=dataset_sha256,
        row_count=row_count,
        feature_names=feature_names,
        feature_count=len(feature_names),
        feature_types=feature_types,
        target_column=target_column if has_target else None,
        target_classes=target_classes,
        class_distribution=class_distribution,
        split_performed=split_performed,
        split_details=split_details if split_performed else None,
        gap_documentation=gap_documentation,
    )
