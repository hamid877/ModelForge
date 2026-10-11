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
