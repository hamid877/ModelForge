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


settings = Settings()
