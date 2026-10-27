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
