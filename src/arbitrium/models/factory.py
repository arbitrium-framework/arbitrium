"""Factory functions for creating models."""

from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import BaseModel, LiteLLMModel

logger = get_contextual_logger("arbitrium.models.factory")


def create_models_from_config(
    config: dict[str, object],
) -> dict[str, BaseModel]:
    """Creates a dictionary of models from a configuration dictionary."""
    models: dict[str, BaseModel] = {}
    logger.info("Creating models from config...")
    models_config = config["models"]
    if not isinstance(models_config, dict):
        return models
    for model_key, model_config in models_config.items():
        if not isinstance(model_config, dict):
            continue
        logger.info(f"Creating model: {model_key}")

        # Handle mock provider for testing
        provider = model_config.get("provider", "")
        if provider == "mock":
            # For mock provider, create a test-friendly mock model
            # Import here to avoid circular dependency
            try:
                from tests.integration.conftest import MockModel

                models[model_key] = MockModel(
                    model_name=str(model_config.get("model_name", model_key)),
                    display_name=str(
                        model_config.get("display_name", model_key)
                    ),
                    temperature=float(model_config.get("temperature", 0.7)),
                    max_tokens=int(model_config.get("max_tokens", 1000)),
                    context_window=int(
                        model_config.get("context_window", 4000)
                    ),
                )
                logger.info(f"Created MockModel for {model_key}")
            except ImportError:
                # If we can't import MockModel (not in test context), create a LiteLLMModel anyway
                # This will fail at runtime but allows the code to load
                logger.warning(
                    f"MockModel not available, creating LiteLLMModel for mock provider {model_key}"
                )
                models[model_key] = LiteLLMModel.from_config(
                    model_key, model_config
                )
        else:
            models[model_key] = LiteLLMModel.from_config(
                model_key, model_config
            )
    return models
