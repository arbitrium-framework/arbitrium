"""Utility functions for benchmarks to avoid code duplication."""

from arbitrium.config import Config
from arbitrium.core.comparison import ModelComparison
from arbitrium.models.base import LiteLLMModel
from arbitrium.models.factory import create_models_from_config
from arbitrium.runners.cli_handler import ConsoleEventHandler
from arbitrium.runners.cli_host import CliHost


def initialize_benchmark(config_path: str) -> tuple[dict[str, object], dict[str, LiteLLMModel], ModelComparison]:
    """
    Initialize benchmark components (config, models, comparison).

    Args:
        config_path: Path to configuration file

    Returns:
        Tuple of (config, models dict, ModelComparison instance)
    """
    config_obj = Config(config_path)
    if not config_obj.load():
        raise ValueError(f"Failed to load configuration from {config_path}")

    config = config_obj.config_data

    # Initialize models
    models = create_models_from_config(config)

    # Initialize event handler and host
    event_handler = ConsoleEventHandler()
    host = CliHost()

    # Initialize comparison
    comparison = ModelComparison(
        config=config,
        models=models,  # type: ignore[arg-type]
        event_handler=event_handler,
        host=host,
    )

    return config, models, comparison
