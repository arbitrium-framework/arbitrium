"""Configuration handling for Arbitrium Framework."""

from pathlib import Path
from typing import Any

import yaml

from arbitrium.logging import get_contextual_logger

# Module-level logger
logger = get_contextual_logger("arbitrium.config")


def validate_config(config_data: dict[str, Any]) -> bool:
    """Validates the configuration data."""
    has_errors = False
    schema = {
        "models": {"required": True, "type": dict, "non_empty": True},
        "retry": {"required": True, "type": dict},
        "features": {"required": True, "type": dict},
        "prompts": {"required": True, "type": dict},
    }

    logger.debug(f"Validating config sections: {list(config_data.keys())}")

    for section, rules in schema.items():
        logger.debug(f"Checking section '{section}': required={rules['required']}, present={section in config_data}")

        if rules["required"] and section not in config_data:
            logger.error(f"'{section}' section is missing in config.yml.")
            has_errors = True
        if rules.get("non_empty") and not config_data.get(section):
            logger.error(f"'{section}' section is empty in config.yml.")
            has_errors = True
        if "type" in rules:
            expected_type = rules["type"]
            section_value = config_data.get(section)
            expected_type_name = getattr(expected_type, "__name__", str(expected_type))
            section_type_name = type(section_value).__name__ if section_value is not None else "None"
            logger.debug(f"Section '{section}' type check: value={section_type_name}, expected={expected_type_name}")
            if section_value is not None and not isinstance(section_value, expected_type):  # type: ignore[arg-type]
                type_name = getattr(expected_type, "__name__", str(expected_type))
                logger.error(f"'{section}' section should be a {type_name}.")
                has_errors = True

    for model_name, model_config in config_data.get("models", {}).items():
        if "model_name" not in model_config:
            logger.error(f"'model_name' is missing for model '{model_name}' in config.yml.")
            has_errors = True
        if "provider" not in model_config:
            logger.error(f"'provider' is missing for model '{model_name}' in config.yml.")
            has_errors = True

    logger.debug(f"Config validation result: has_errors={has_errors}")
    return not has_errors


class Config:
    """Configuration manager for Arbitrium Framework."""

    def __init__(self, config_path: str = "config.yml") -> None:
        """Initialize configuration from the given path."""
        self.config_path = Path(config_path)
        self.config_data: dict[str, Any] = {}
        self.retry_settings: dict[str, Any] = {}
        self.feature_flags: dict[str, Any] = {}
        self.prompts: dict[str, Any] = {}

    def _load_config_file(self) -> bool:
        """Load and parse the YAML config file."""
        if not self.config_path.exists():
            logger.error(f"Config file not found at {self.config_path.resolve()}")
            return False

        with open(self.config_path, encoding="utf-8") as f:
            try:
                self.config_data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error(f"Failed to parse YAML config file: {e}")
                return False
        logger.info(f"Loaded configuration from {self.config_path}")
        return True

    def load(self) -> bool:
        """Load configuration from file."""
        try:
            if not self._load_config_file():
                return False

            if not validate_config(self.config_data):
                logger.error("Configuration validation failed. Halting execution.")
                return False

            self._setup_config_shortcuts()
            return True
        except (FileNotFoundError, yaml.YAMLError) as e:
            logger.error(f"Failed to load configuration: {e!s}")
            return False
        except Exception as e:
            logger.critical(f"Unexpected error loading config: {e!s}", exc_info=True)
            return False

    def _setup_config_shortcuts(self) -> None:
        """Set up shortcut attributes for commonly accessed config sections."""
        retry_config = self.config_data.get("retry", {})
        self.retry_settings = {
            "max_attempts": retry_config.get("max_attempts", 3),
            "initial_delay": retry_config.get("initial_delay", 15),
            "max_delay": retry_config.get("max_delay", 120),
        }
        self.feature_flags = self.config_data.get("features", {})
        self.prompts = self.config_data.get("prompts", {})

    def get_model_config(self, model_key: str) -> dict[str, Any]:
        """Get configuration for a specific model, with feature flags merged in."""
        base_config = self.config_data.get("models", {}).get(model_key, {})
        if not base_config:
            return {}

        model_config = base_config.copy()
        features = self.config_data.get("features", {})

        if "llm_compression" not in model_config:
            model_config["llm_compression"] = features.get("llm_compression", True)
        if "compression_model" not in model_config:
            model_config["compression_model"] = features.get("compression_model", "ollama/qwen:1.8b")

        result: dict[str, Any] = model_config
        return result

    def get_active_model_keys(self) -> list[str]:
        """Get list of all configured model keys."""
        return list(self.config_data.get("models", {}).keys())
