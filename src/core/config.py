"""Configuration management for VibeCode AI Mentor.

Loads and validates environment variables using Pydantic BaseSettings.
Provides a singleton pattern for global configuration access.
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""

    # TiDB Configuration
    tidb_host: str = Field(..., description="TiDB host address")
    tidb_user: str = Field(..., description="TiDB username")
    tidb_password: str = Field(..., description="TiDB password")
    tidb_port: int = Field(default=4000, description="TiDB port")
    tidb_database: str = Field(default="vibecode", description="TiDB database name")
    tidb_ssl_ca: Optional[str] = Field(
        default="/etc/ssl/certs/ca-certificates.crt", description="SSL CA path"
    )

    # API Keys
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    openai_api_key: Optional[str] = Field(
        default=None, description="OpenAI API key (optional fallback)"
    )
    api_keys: Optional[str] = Field(
        default=None, description="Comma-separated list of valid API keys for authentication"
    )

    # Application Settings
    environment: str = Field(default="development", description="Environment name")
    api_host: str = Field(default="0.0.0.0", description="API host address")
    api_port: int = Field(default=8000, description="API port")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    workers: int = Field(default=4, description="Number of API workers")
    reload: bool = Field(default=True, description="Auto-reload on code changes")

    # Performance Settings
    batch_size: int = Field(default=100, description="Batch processing size")
    max_chunk_size: int = Field(default=2048, description="Maximum chunk size")
    min_chunk_size: int = Field(default=512, description="Minimum chunk size")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    enable_cache: bool = Field(default=True, description="Enable caching")

    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100, description="Rate limit requests per period"
    )
    rate_limit_period: int = Field(
        default=60, description="Rate limit period in seconds"
    )

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    metrics_port: int = Field(default=9090, description="Metrics port")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}")
        return v.upper()

    @field_validator("min_chunk_size", "max_chunk_size")
    @classmethod
    def validate_chunk_sizes(cls, v: int, info) -> int:
        """Validate chunk size constraints."""
        if v <= 0:
            raise ValueError("Chunk size must be positive")
        if v > 8192:
            raise ValueError("Chunk size too large (max 8192)")
        return v

    @property
    def tidb_connection_string(self) -> str:
        """Generate TiDB connection string."""
        return (
            f"mysql+pymysql://{self.tidb_user}:{self.tidb_password}@"
            f"{self.tidb_host}:{self.tidb_port}/{self.tidb_database}"
        )

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()


# Convenience function for direct access
settings = get_settings()