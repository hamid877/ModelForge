# ModelForge Provenance & Promotion Audit Reference

This document details the audit schemas, cryptographic provenance fields, gate outcomes, and safety mechanisms introduced in ModelForge Phase 4.

---

## 1. Dataset Provenance Schema

Dataset provenance provides cryptographic identity and partition metadata for every training and promotion run without leaking sensitive row-level records.

### Fields

| Field | Type | Description |
|---|---|---|
| `dataset_sha256` | `str` | SHA-256 hex digest of the full input DataFrame. |
| `row_count` | `int` | Total number of observations in the dataset. |
| `feature_names` | `list[str]` | Sorted list of input feature column names (excluding target). |
| `feature_count` | `int` | Total count of feature columns. |
| `feature_types` | `dict[str, str]` | Mapping of feature column name to pandas dtype string. |
| `target_column` | `str \| null` | Name of the target column (default `"churn"`). |
| `target_classes` | `list[int \| str] \| null` | Sorted list of unique target class values (e.g. `[0, 1]`). |
| `class_distribution` | `dict[str, int] \| null` | Count of records per target class (e.g. `{"0": 50, "1": 50}`). |
| `split.split_performed` | `bool` | `true` if a caller train/test split was executed; `false` otherwise. |
| `split.train_rows` | `int` (optional) | Number of training set partition rows. |
| `split.test_rows` | `int` (optional) | Number of test set partition rows. |
| `split.test_size` | `float` (optional) | Configured test partition proportion (e.g. `0.2`). |
| `split.random_state` | `int` (optional) | Seed used for partition reproducibility. |
| `split.stratified` | `bool` (optional) | Whether partitioning was stratified on target. |
| `split.train_sha256` | `str` (optional) | SHA-256 hex digest of the training partition DataFrame. |
| `split.test_sha256` | `str` (optional) | SHA-256 hex digest of the test partition DataFrame. |
| `split.gap_documentation` | `str` (optional) | Explicit explanation when no split was performed. |

### Privacy & Gap Policy
- **No Row Leakage:** Raw observation data, customer identifiers, and individual feature values are never captured in provenance or audit payloads.
- **No Fabricated Splits:** If a pipeline caller does not execute a train/test split, `split_performed` is set to `false` and `gap_documentation` describes the exact execution mode rather than estimating or inventing partition values.

---

## 2. Promotion Audit Schema

Promotion decisions—both **approved promotions and rejected attempts**—are persisted to the SQLite audit database (`promotion_audit` table in `data/lifecycle_audit.db`).

### SQLite Table DDL (`promotion_audit`)

```sql
CREATE TABLE IF NOT EXISTS promotion_audit (
    audit_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    candidate_version TEXT,
    candidate_run_id TEXT,
    champion_version TEXT,
    approved INTEGER NOT NULL,
    reason TEXT NOT NULL,
    candidate_f1 REAL NOT NULL,
    champion_f1 REAL,
    benchmark_version TEXT NOT NULL,
    benchmark_digest TEXT NOT NULL,
    margin REAL NOT NULL,
    payload TEXT NOT NULL
);
```

### Column Reference

| Column | Type | Nullable | Description |
|---|---|---|---|
| `audit_id` | `TEXT` | No | UUIDv4 identifying the promotion evaluation event. |
| `timestamp` | `TEXT` | No | ISO-8601 UTC timestamp of the evaluation. |
| `model_name` | `TEXT` | No | Registered model name (e.g. `"churn-classifier"`). |
| `candidate_version` | `TEXT` | Yes | Registered version number of the candidate model. |
| `candidate_run_id` | `TEXT` | Yes | MLflow run ID associated with the candidate model. |
| `champion_version` | `TEXT` | Yes | Version number of the active champion model (if present). |
| `approved` | `INTEGER` | No | `1` if candidate was promoted; `0` if rejected or failed. |
| `reason` | `TEXT` | No | Machine-readable gate outcome or rejection reason. |
| `candidate_f1` | `REAL` | No | Candidate F1 score evaluated on the shared benchmark. |
| `champion_f1` | `REAL` | Yes | Champion F1 score evaluated on the shared benchmark (`NULL` if no champion exists). |
| `benchmark_version` | `TEXT` | No | Version string of the evaluation benchmark (default `"v1"`). |
| `benchmark_digest` | `TEXT` | No | SHA-256 hex digest of the benchmark dataset. |
| `margin` | `REAL` | No | Configured minimum F1 improvement margin (e.g. `0.01`). |
| `payload` | `TEXT` | No | JSON-serialized dictionary containing complete evaluation metrics, gate outcomes, and dataset provenance. |

---

## 3. Rejection Reasons & Gate Outcomes

Every promotion attempt evaluates five sequential gates. Rejections across any gate are fully captured in the audit log:

| Gate | Failure Reason | Description |
|---|---|---|
| **Benchmark Integrity** | `benchmark_missing` | Benchmark file not found on disk or empty DataFrame passed. |
| | `benchmark_invalid` | Benchmark fails schema validation or column type constraints. |
| | `benchmark_digest_mismatch` | Actual benchmark SHA-256 does not match configured digest. |
| | `benchmark_insufficient_samples` | Benchmark row count is below configured minimum (`benchmark_min_samples`). |
| **Separation & Overlap** | `separation_not_established` | Candidate training data was not supplied; overlap cannot be verified. |
| | `training_benchmark_overlap` | Exact match found between candidate training rows and benchmark rows. |
| **Candidate Evaluation** | `candidate_model_missing` | Candidate model object was not provided for evaluation. |
| **Champion Evaluation** | `champion_benchmark_evaluation_failed` | Active champion model could not be evaluated on the benchmark. |
| **Quality Gate** | `candidate_below_minimum_f1` | Candidate F1 on benchmark is below `promotion_minimum_f1` (0.70). |
| | `candidate_f1_below_champion` | Candidate F1 is lower than the active champion's F1. |
| | `candidate_f1_tied_champion` | Candidate F1 equals champion F1 (ties do not promote). |
| | `candidate_f1_insufficient_improvement` | Candidate F1 beats champion by less than `promotion_f1_margin` (0.01). |
| | `benchmark_version_mismatch` | Candidate and champion were evaluated on differing benchmark versions. |
| **Audit & Logging Safety** | `audit_logging_failed` | SQLite persistence or MLflow traceability logging failed; promotion aborted. |

---

## 4. Fail-Closed Promotion Safety

To guarantee safe continuous deployment:
1. **Audit Precedes Promotion:** Promotion audit persistence to SQLite and MLflow must succeed **before** `set_registered_model_alias` is invoked.
2. **Fail-Closed on Logging Error:** If an I/O error, disk-full condition, or database lock occurs during audit recording, the promotion is immediately aborted with `approved=False` and `reason="audit_logging_failed"`.
3. **Alias Protection:** The live champion alias is never touched unless all evaluation gates pass and the audit record is safely committed.

---

## 5. Known Limitations

- **20-Row Synthetic Benchmark:** The default benchmark dataset (`data/benchmark/churn_benchmark.csv`) contains exactly 20 synthetic rows with SHA-256 `68f772d75783db0ded9aca953f8fbab70718a531048a88bffcfb5d2d5bc715fe`. It validates data pipeline evaluability, gate mechanics, and cryptographic integrity. It does **not** establish statistical sufficiency or statistical power for real-world model performance claims.
- **Improvement Margin vs. Significance:** The configured F1 margin (`0.01`) prevents promotions on noise and ties, but does not substitute for formal statistical hypothesis testing or real-time canary analysis.
