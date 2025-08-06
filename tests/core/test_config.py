"""Tests for configuration management."""

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from src.core.config import Settings, get_settings


class TestSettings:
    """Test configuration settings."""

    def test_settings_loads_from_env(self):
        """Test settings load from environment variables."""
        env_vars = {
            "TIDB_HOST": "test.tidb.com",
            "TIDB_USER": "testuser",
            "TIDB_PASSWORD": "testpass",
            "GEMINI_API_KEY": "test_gemini_key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.tidb_host == "test.tidb.com"
            assert settings.tidb_user == "testuser"
            assert settings.tidb_password == "testpass"
            assert settings.gemini_api_key == "test_gemini_key"

    def test_settings_defaults(self):
        """Test default values are applied."""
        env_vars = {
            "TIDB_HOST": "test.tidb.com",
            "TIDB_USER": "testuser",
            "TIDB_PASSWORD": "testpass",
            "GEMINI_API_KEY": "test_key",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.tidb_port == 4000
            assert settings.tidb_database == "vibecode"
            assert settings.api_port == 8000
            assert settings.debug is False
            assert settings.log_level == "INFO"
            assert settings.batch_size == 100
            assert settings.max_chunk_size == 2048
            assert settings.min_chunk_size == 512
            assert settings.cache_ttl == 3600
            assert settings.enable_cache is True

    def test_missing_required_fields(self):
        """Test validation error for missing required fields."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            errors = exc_info.value.errors()
            required_fields = {"tidb_host", "tidb_user", "tidb_password", "gemini_api_key"}
            error_fields = {error["loc"][0] for error in errors}
            assert required_fields.issubset(error_fields)

    def test_log_level_validation(self):
        """Test log level validation."""
        env_vars = {
            "TIDB_HOST": "test",
            "TIDB_USER": "test",
            "TIDB_PASSWORD": "test",
            "GEMINI_API_KEY": "test",
            "LOG_LEVEL": "invalid",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "Invalid log level" in str(exc_info.value)

    def test_chunk_size_validation(self):
        """Test chunk size validation."""
        base_env = {
            "TIDB_HOST": "test",
            "TIDB_USER": "test",
            "TIDB_PASSWORD": "test",
            "GEMINI_API_KEY": "test",
        }

        # Test negative chunk size
        with patch.dict(os.environ, {**base_env, "MIN_CHUNK_SIZE": "-1"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "Chunk size must be positive" in str(exc_info.value)

        # Test chunk size too large
        with patch.dict(os.environ, {**base_env, "MAX_CHUNK_SIZE": "10000"}, clear=True):
            with pytest.raises(ValidationError) as exc_info:
                Settings()
            assert "Chunk size too large" in str(exc_info.value)

    def test_connection_string_generation(self):
        """Test TiDB connection string generation."""
        env_vars = {
            "TIDB_HOST": "test.tidb.com",
            "TIDB_USER": "testuser",
            "TIDB_PASSWORD": "testpass",
            "TIDB_PORT": "3306",
            "TIDB_DATABASE": "testdb",
            "GEMINI_API_KEY": "test",
        }

        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            expected = "mysql+pymysql://testuser:testpass@test.tidb.com:3306/testdb"
            assert settings.tidb_connection_string == expected

    def test_environment_helpers(self):
        """Test environment helper properties."""
        base_env = {
            "TIDB_HOST": "test",
            "TIDB_USER": "test",
            "TIDB_PASSWORD": "test",
            "GEMINI_API_KEY": "test",
        }

        # Test production
        with patch.dict(os.environ, {**base_env, "ENVIRONMENT": "production"}, clear=True):
            settings = Settings()
            assert settings.is_production is True
            assert settings.is_development is False

        # Test development
        with patch.dict(os.environ, {**base_env, "ENVIRONMENT": "development"}, clear=True):
            settings = Settings()
            assert settings.is_production is False
            assert settings.is_development is True

    def test_singleton_pattern(self):
        """Test singleton pattern works correctly."""
        with patch.dict(
            os.environ,
            {
                "TIDB_HOST": "test",
                "TIDB_USER": "test",
                "TIDB_PASSWORD": "test",
                "GEMINI_API_KEY": "test",
            },
            clear=True,
        ):
            settings1 = get_settings()
            settings2 = get_settings()
            assert settings1 is settings2  # Same instance