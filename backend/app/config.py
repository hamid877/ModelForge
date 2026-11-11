import hashlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_API_ROLES: frozenset[str] = frozenset(
    {"predict", "monitoring", "retraining", "audit", "admin"}
)


class ClientApiKeyConfig(BaseModel):
    """Configuration for an authenticated API client credential."""

    model_config = ConfigDict(
        hide_input_in_errors=True,
        frozen=True,
        extra="forbid",
    )

    client_id: str = Field(min_length=1)
    token: SecretStr = Field(...)
    roles: tuple[str, ...] = Field(min_length=1)

    @field_validator("client_id")
    @classmethod
    def validate_client_id(cls, v: str) -> str:
        clean = v.strip()
        if not clean:
            raise ValueError("client_id must not be empty or whitespace.")
        return clean

    @field_validator("token")
    @classmethod
    def validate_token(cls, v: SecretStr) -> SecretStr:
        clean = v.get_secret_value().strip()
        if not clean:
            raise ValueError("token must not be empty or whitespace.")
        return SecretStr(clean)

    @field_validator("roles", mode="before")
    @classmethod
    def normalize_roles(cls, v: Any) -> tuple[str, ...]:
        if isinstance(v, (list, tuple, set)):
            return tuple(str(r).strip() for r in v)
        raise ValueError("roles must be a list or tuple of strings.")

    @field_validator("roles")
    @classmethod
    def validate_roles(cls, roles: tuple[str, ...]) -> tuple[str, ...]:
        if not roles:
            raise ValueError("Client must have at least one assigned role.")
        for role in roles:
            if role not in ALLOWED_API_ROLES:
                allowed_str = ", ".join(sorted(ALLOWED_API_ROLES))
                raise ValueError(
                    f"Invalid role '{role}'. Allowed roles are: {allowed_str}."
                )
        return roles


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="MODELFORGE_",
        extra="ignore",
        hide_input_in_errors=True,
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
    retraining_cooldown_seconds: int = Field(
        default=3600,
        ge=0,
    )
    retraining_job_timeout_seconds: float = Field(
        default=600.0,
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

    api_keys: tuple[ClientApiKeyConfig, ...] = Field(
        default=(),
    )

    # Process-local sliding-window rate limiting configuration
    rate_limit_enabled: bool = Field(
        default=True,
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        ge=1,
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
    )

    @field_validator("api_keys", mode="after")
    @classmethod
    def validate_api_keys(
        cls, api_keys: tuple[ClientApiKeyConfig, ...]
    ) -> tuple[ClientApiKeyConfig, ...]:
        client_ids: set[str] = set()
        token_hashes: set[str] = set()
        for client in api_keys:
            if client.client_id in client_ids:
                raise ValueError(
                    f"Duplicate client_id '{client.client_id}' configured."
                )
            client_ids.add(client.client_id)

            token_val = client.token.get_secret_value().strip()
            if not token_val:
                raise ValueError(
                    f"Token for client_id '{client.client_id}' must not be empty."
                )
            token_digest = hashlib.sha256(token_val.encode()).hexdigest()
            if token_digest in token_hashes:
                raise ValueError(
                    "Duplicate token detected across multiple configured clients."
                )
            token_hashes.add(token_digest)

        return api_keys


settings = Settings()
