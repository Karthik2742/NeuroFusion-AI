"""
NeuroFusion AI - Configuration Manager
Loads and validates YAML configuration with environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class Config:
    """Singleton configuration manager for NeuroFusion AI."""

    _instance: Config | None = None
    _config: dict[str, Any] = {}

    def __new__(cls) -> Config:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        config_path = Path(__file__).parent / "default.yaml"
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            self._config = {}
            return

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}

        logger.info(f"Configuration loaded from {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Dot-separated key path (e.g., 'app.name', 'training.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        # Check environment variable override first
        env_key = f"NEUROFUSION_{key.upper().replace('.', '_')}"
        env_val = os.environ.get(env_key)
        if env_val is not None:
            return self._parse_env_value(env_val)

        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable string to appropriate Python type."""
        if value.lower() in ("true", "yes", "1"):
            return True
        if value.lower() in ("false", "no", "0"):
            return False
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
        return value

    @property
    def raw(self) -> dict[str, Any]:
        """Access raw configuration dictionary."""
        return self._config.copy()

    def reload(self) -> None:
        """Reload configuration from disk."""
        self._load_config()


# Global config instance
config = Config()
