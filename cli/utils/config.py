"""Configuration management for CLI."""

import os
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel, Field


class CLIConfig(BaseModel):
    """CLI-specific configuration."""
    
    # Database settings
    tidb_host: Optional[str] = None
    tidb_user: Optional[str] = None
    tidb_password: Optional[str] = None
    tidb_port: int = 4000
    tidb_database: str = "vibecode"
    
    # API settings
    api_host: str = "localhost"
    api_port: int = 8000
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    
    # CLI preferences
    default_language: str = "python"
    batch_size: int = 100
    max_workers: int = 4
    output_format: str = "table"  # table, json, yaml
    
    # File patterns
    include_patterns: list[str] = Field(default_factory=lambda: ["**/*.py"])
    exclude_patterns: list[str] = Field(default_factory=lambda: [
        "**/__pycache__/**",
        "**/.*",
        "**/node_modules/**",
        "**/venv/**",
        "**/env/**",
        "**/.git/**"
    ])
    
    # Analysis settings
    min_lines_for_analysis: int = 5
    complexity_threshold: int = 10
    similarity_threshold: float = 0.8
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "CLIConfig":
        """Load configuration from file and environment."""
        config_data = {}
        
        # Try to load from config file
        if config_path:
            config_file = Path(config_path)
        else:
            # Look for config in default locations
            config_file = None
            for location in [
                Path.home() / ".vibecode" / "config.yaml",
                Path.home() / ".vibecode" / "config.yml",
                Path.cwd() / ".vibecode.yaml",
                Path.cwd() / ".vibecode.yml",
            ]:
                if location.exists():
                    config_file = location
                    break
        
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f) or {}
                config_data.update(file_config)
        
        # Override with environment variables
        env_mapping = {
            "TIDB_HOST": "tidb_host",
            "TIDB_USER": "tidb_user", 
            "TIDB_PASSWORD": "tidb_password",
            "TIDB_PORT": "tidb_port",
            "TIDB_DATABASE": "tidb_database",
            "GEMINI_API_KEY": "gemini_api_key",
            "OPENAI_API_KEY": "openai_api_key",
            "VIBECODE_API_HOST": "api_host",
            "VIBECODE_API_PORT": "api_port",
            "VIBECODE_DEFAULT_LANGUAGE": "default_language",
            "VIBECODE_BATCH_SIZE": "batch_size",
            "VIBECODE_MAX_WORKERS": "max_workers",
            "VIBECODE_OUTPUT_FORMAT": "output_format",
        }
        
        for env_var, config_key in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert types as needed
                if config_key in ["tidb_port", "api_port", "batch_size", "max_workers", "min_lines_for_analysis", "complexity_threshold"]:
                    config_data[config_key] = int(env_value)
                elif config_key in ["similarity_threshold"]:
                    config_data[config_key] = float(env_value)
                else:
                    config_data[config_key] = env_value
        
        return cls(**config_data)
    
    def save(self, config_path: Optional[str] = None) -> Path:
        """Save configuration to file."""
        if config_path:
            config_file = Path(config_path)
        else:
            config_dir = Path.home() / ".vibecode"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "config.yaml"
        
        # Prepare config data (exclude None values)
        config_data = self.model_dump(exclude_none=True)
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=True)
        
        return config_file
    
    @property
    def has_database_config(self) -> bool:
        """Check if database configuration is complete."""
        return all([
            self.tidb_host,
            self.tidb_user,
            self.tidb_password
        ])
    
    @property
    def has_api_keys(self) -> bool:
        """Check if API keys are configured."""
        return self.gemini_api_key is not None or self.openai_api_key is not None
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert config to environment variables dict."""
        env_dict = {}
        
        if self.tidb_host:
            env_dict["TIDB_HOST"] = self.tidb_host
        if self.tidb_user:
            env_dict["TIDB_USER"] = self.tidb_user
        if self.tidb_password:
            env_dict["TIDB_PASSWORD"] = self.tidb_password
        if self.gemini_api_key:
            env_dict["GEMINI_API_KEY"] = self.gemini_api_key
        if self.openai_api_key:
            env_dict["OPENAI_API_KEY"] = self.openai_api_key
        
        env_dict.update({
            "TIDB_PORT": str(self.tidb_port),
            "TIDB_DATABASE": self.tidb_database,
            "VIBECODE_API_HOST": self.api_host,
            "VIBECODE_API_PORT": str(self.api_port),
        })
        
        return env_dict