import pytest
from pydantic import ValidationError

from backend.app.config import Settings


def test_settings_use_defaults(monkeypatch):
    """Use default values when environment variables are absent."""
    monkeypatch.delenv("MODELFORGE_MODEL_NAME", raising=False)
    monkeypatch.delenv("MODELFORGE_MODEL_ALIAS", raising=False)

    config = Settings(_env_file=None)

    assert config.model_name == "churn-classifier"
    assert config.model_alias == "champion"


def test_settings_read_environment_variables(monkeypatch):
    """Override defaults using environment variables."""
    monkeypatch.setenv("MODELFORGE_MODEL_NAME", "test-classifier")
    monkeypatch.setenv("MODELFORGE_MODEL_ALIAS", "staging")

    config = Settings(_env_file=None)

    assert config.model_name == "test-classifier"
    assert config.model_alias == "staging"


def test_settings_reject_empty_model_name(monkeypatch):
    """Reject an empty model name."""
    monkeypatch.setenv("MODELFORGE_MODEL_NAME", "")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_reject_empty_model_alias(monkeypatch):
    """Reject an empty model alias."""
    monkeypatch.setenv("MODELFORGE_MODEL_ALIAS", "")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_benchmark_settings_use_defaults(monkeypatch):
    """Verify default values for safe promotion and benchmark configuration."""
    monkeypatch.delenv("MODELFORGE_PROMOTION_F1_MARGIN", raising=False)
    monkeypatch.delenv("MODELFORGE_BENCHMARK_DATA_PATH", raising=False)
    monkeypatch.delenv("MODELFORGE_BENCHMARK_MIN_SAMPLES", raising=False)
    monkeypatch.delenv("MODELFORGE_BENCHMARK_VERSION", raising=False)
    monkeypatch.delenv("MODELFORGE_BENCHMARK_EXPECTED_SHA256", raising=False)

    config = Settings(_env_file=None)

    assert config.promotion_f1_margin == 0.01
    assert config.benchmark_data_path == "data/benchmark/churn_benchmark.csv"
    assert config.benchmark_min_samples == 10
    assert config.benchmark_version == "v1"
    assert (
        config.benchmark_expected_sha256
        == "68f772d75783db0ded9aca953f8fbab70718a531048a88bffcfb5d2d5bc715fe"
    )


def test_benchmark_settings_read_environment_variables(monkeypatch):
    """Verify overriding benchmark configuration via environment variables."""
    monkeypatch.setenv("MODELFORGE_PROMOTION_F1_MARGIN", "0.05")
    monkeypatch.setenv("MODELFORGE_BENCHMARK_DATA_PATH", "data/test_benchmark.csv")
    monkeypatch.setenv("MODELFORGE_BENCHMARK_MIN_SAMPLES", "50")
    monkeypatch.setenv("MODELFORGE_BENCHMARK_VERSION", "2026-q4")
    monkeypatch.setenv(
        "MODELFORGE_BENCHMARK_EXPECTED_SHA256",
        "custom_sha256_digest_value",
    )

    config = Settings(_env_file=None)

    assert config.promotion_f1_margin == 0.05
    assert config.benchmark_data_path == "data/test_benchmark.csv"
    assert config.benchmark_min_samples == 50
    assert config.benchmark_version == "2026-q4"
    assert config.benchmark_expected_sha256 == "custom_sha256_digest_value"


def test_promotion_audit_settings(monkeypatch):
    """Verify default and overridden values for promotion audit database path."""
    monkeypatch.delenv("MODELFORGE_PROMOTION_AUDIT_DB_PATH", raising=False)
    config = Settings(_env_file=None)
    assert config.promotion_audit_db_path == "data/lifecycle_audit.db"

    monkeypatch.setenv("MODELFORGE_PROMOTION_AUDIT_DB_PATH", "data/custom_audit.db")
    custom_config = Settings(_env_file=None)
    assert custom_config.promotion_audit_db_path == "data/custom_audit.db"


def test_retraining_orchestration_settings(monkeypatch):
    """Verify default and overridden values for retraining orchestration settings."""
    env_vars = [
        "MODELFORGE_SQLITE_BUSY_TIMEOUT_MS",
        "MODELFORGE_WORKER_LOCK_PATH",
        "MODELFORGE_RETRAINING_LEASE_DURATION_SECONDS",
        "MODELFORGE_RETRAINING_HEARTBEAT_INTERVAL_SECONDS",
        "MODELFORGE_RETRAINING_MAX_ATTEMPTS",
        "MODELFORGE_RETRAINING_POLL_INTERVAL_SECONDS",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)

    config = Settings(_env_file=None)
    assert config.sqlite_busy_timeout_ms == 5000
    assert config.worker_lock_path == "data/worker.lock"
    assert config.retraining_lease_duration_seconds == 60
    assert config.retraining_heartbeat_interval_seconds == 15
    assert config.retraining_max_attempts == 3
    assert config.retraining_poll_interval_seconds == 5.0

    monkeypatch.setenv("MODELFORGE_SQLITE_BUSY_TIMEOUT_MS", "10000")
    monkeypatch.setenv("MODELFORGE_WORKER_LOCK_PATH", "data/custom_worker.lock")
    monkeypatch.setenv("MODELFORGE_RETRAINING_LEASE_DURATION_SECONDS", "120")
    monkeypatch.setenv("MODELFORGE_RETRAINING_HEARTBEAT_INTERVAL_SECONDS", "30")
    monkeypatch.setenv("MODELFORGE_RETRAINING_MAX_ATTEMPTS", "5")
    monkeypatch.setenv("MODELFORGE_RETRAINING_POLL_INTERVAL_SECONDS", "2.5")

    custom = Settings(_env_file=None)
    assert custom.sqlite_busy_timeout_ms == 10000
    assert custom.worker_lock_path == "data/custom_worker.lock"
    assert custom.retraining_lease_duration_seconds == 120
    assert custom.retraining_heartbeat_interval_seconds == 30
    assert custom.retraining_max_attempts == 5
    assert custom.retraining_poll_interval_seconds == 2.5
