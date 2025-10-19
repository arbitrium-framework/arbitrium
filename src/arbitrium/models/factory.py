"""Factory functions for creating models."""

from arbitrium.config.defaults import select_model_with_highest_context
from arbitrium.logging import get_contextual_logger
from arbitrium.models.base import BaseModel

# Import providers to trigger registration
from arbitrium.models.providers import LiteLLMProvider  # noqa: F401
from arbitrium.models.registry import ProviderRegistry

logger = get_contextual_logger("arbitrium.models.factory")


def _create_single_model(
    model_key: str, model_config: dict[str, object]
) -> BaseModel | None:
    """Create a single model from configuration.

    Args:
        model_key: The key identifying the model
        model_config: Configuration dictionary for this model

    Returns:
        BaseModel instance or None if creation failed

    Raises:
        ValueError: If provider is not registered
    """
    provider = model_config.get("provider", "")
    if not provider or not isinstance(provider, str):
        logger.warning(f"No provider specified for {model_key}, skipping")
        return None

    try:
        model = ProviderRegistry.create(provider, model_key, model_config)
        logger.info(
            f"Created {provider} model for {model_key}: {model.display_name}"
        )
        return model
    except ValueError as e:
        logger.error(
            f"Failed to create model {model_key}: {e}. "
            f"Available providers: {ProviderRegistry.list_providers()}"
        )
        raise


def _setup_compression_models(models: dict[str, BaseModel]) -> None:
    """Auto-select and set compression model for models that need it.

    Args:
        models: Dictionary of created models
    """
    if not models:
        return

    # Check if any models need compression model selection
    needs_compression_selection = any(
        model.compression_model is None for model in models.values()
    )

    if not needs_compression_selection:
        return

    # Select model with highest context from active models
    compression_model_key = select_model_with_highest_context(models)

    if not compression_model_key:
        logger.warning(
            "Could not auto-select compression model: no models have context_window set"
        )
        return

    compression_model_name = models[compression_model_key].model_name
    logger.info(
        f"Auto-selected compression model from active models: "
        f"{compression_model_key} ({compression_model_name}) "
        f"with {models[compression_model_key].context_window:,} token context"
    )

    # Set compression model for all models that have None
    for model in models.values():
        if model.compression_model is None:
            model.compression_model = compression_model_name
            logger.debug(
                f"Set compression_model={compression_model_name} for {model.model_key}"
            )


def create_models_from_config(
    config: dict[str, object],
) -> dict[str, BaseModel]:
    """Creates a dictionary of models from a configuration dictionary.

    Uses the ProviderRegistry to create models. Production providers
    (openai, anthropic, etc.) are registered in models/providers.py.
    Test providers (mock) are registered in test fixtures.

    If compression_model is None in config, automatically selects the model
    with the highest context window from the created (active) models.

    Args:
        config: Configuration dictionary with "models" key

    Returns:
        Dictionary mapping model keys to BaseModel instances

    Raises:
        ValueError: If a provider is not registered
    """
    logger.info("Creating models from config...")
    models_config = config["models"]

    if not isinstance(models_config, dict):
        return {}

    models: dict[str, BaseModel] = {}

    for model_key, model_config in models_config.items():
        if not isinstance(model_config, dict):
            continue

        logger.info(f"Creating model: {model_key}")
        model = _create_single_model(model_key, model_config)

        if model is not None:
            models[model_key] = model

    # Auto-select compression model from active models if needed
    _setup_compression_models(models)

    return models
