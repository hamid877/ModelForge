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


settings = Settings()
