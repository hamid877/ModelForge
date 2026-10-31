from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MODELFORGE_",
        extra="ignore",
    )

    model_name: str = Field(
        default="churn-classifier",
        min_length=1,
    )
    model_alias: str = Field(
        default="champion",
        min_length=1,
    )
    mlflow_tracking_uri: str = Field(
        default="http://127.0.0.1:5001",
        min_length=1,
    )
    promotion_minimum_f1: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )
    drift_reference_path: str = Field(
        default="data/raw/churn.csv",
        min_length=1,
    )
    drift_min_samples: int = Field(default=10, ge=2)
    drift_p_value_threshold: float = Field(default=0.05, gt=0, lt=1)

    lifecycle_audit_db_path: str = Field(
        default="data/lifecycle_audit.db",
        min_length=1,
    )
    promotion_audit_db_path: str = Field(
        default="data/lifecycle_audit.db",
        min_length=1,
    )
    sqlite_busy_timeout_ms: int = Field(
        default=5000,
        ge=0,
    )
    worker_lock_path: str = Field(
        default="data/worker.lock",
        min_length=1,
    )
    retraining_lease_duration_seconds: int = Field(
        default=60,
        ge=1,
    )
    retraining_heartbeat_interval_seconds: int = Field(
        default=15,
        ge=1,
    )
    retraining_max_attempts: int = Field(
        default=3,
        ge=1,
    )
    retraining_poll_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
    )

    # Promotion quality gate and shared benchmark settings
    # Note: Requiring an F1 improvement margin prevents accidental promotions
    # on noise and ties, but does NOT claim statistical significance.
    promotion_f1_margin: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
    )
    benchmark_data_path: str = Field(
        default="data/benchmark/churn_benchmark.csv",
        min_length=1,
    )
    # Note: Meeting the minimum sample count validates data presence
    # and basic evaluability, but does NOT establish statistical sufficiency
    # or power for comparative benchmarking.
    benchmark_min_samples: int = Field(
        default=10,
        ge=1,
    )
    benchmark_version: str = Field(
        default="v1",
        min_length=1,
    )
    benchmark_expected_sha256: str | None = Field(
        default="68f772d75783db0ded9aca953f8fbab70718a531048a88bffcfb5d2d5bc715fe",
    )


settings = Settings()
