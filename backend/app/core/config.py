"""
AgroVision centralized configuration management using pydantic-settings.
"""

from functools import lru_cache
import secrets
from typing import List, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment variables and/or .env file.
    Provides sensible defaults for local development.
    """

    # Application identity
    APP_NAME: str = "AgroVision API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # API Routing
    API_V1_PREFIX: str = "/api/v1"

    # Database configuration (PostgreSQL with Psycopg 3 driver)
    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@localhost:5432/agrovision"

    # JWT Authentication configuration
    # Uses a cryptographically random ephemeral secret in dev if not set in environment.
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        """Support comma-separated strings or list representations from environment variables."""
        if isinstance(v, str) and not v.strip().startswith("["):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return cached singleton instance of application settings."""
    return Settings()


settings = get_settings()
