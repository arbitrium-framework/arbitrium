"""Factory functions for creating models."""

from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import LiteLLMModel

logger = get_contextual_logger("arbitrium.models.factory")


def create_models_from_config(config: dict[str, object]) -> dict[str, LiteLLMModel]:
    """Creates a dictionary of models from a configuration dictionary."""
    models: dict[str, LiteLLMModel] = {}
    logger.info("Creating models from config...")
    models_config = config["models"]
    if not isinstance(models_config, dict):
        return models
    for model_key, model_config in models_config.items():
        logger.info(f"Creating model: {model_key}")
        models[model_key] = LiteLLMModel.from_config(model_key, model_config)
    return models
