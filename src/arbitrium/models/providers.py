"""Standard model providers for Arbitrium."""

from typing import Any

from arbitrium.models.base import BaseModel, LiteLLMModel
from arbitrium.models.registry import ProviderRegistry


@ProviderRegistry.register("openai")
@ProviderRegistry.register("anthropic")
@ProviderRegistry.register("vertex_ai")
@ProviderRegistry.register("google")
@ProviderRegistry.register("bedrock")
@ProviderRegistry.register("azure")
@ProviderRegistry.register("cohere")
@ProviderRegistry.register("replicate")
@ProviderRegistry.register("huggingface")
@ProviderRegistry.register("ollama")
@ProviderRegistry.register("together_ai")
@ProviderRegistry.register("anyscale")
@ProviderRegistry.register("palm")
@ProviderRegistry.register("mistral")
@ProviderRegistry.register("xai")
class LiteLLMProvider:
    """Provider for all LiteLLM-supported models.

    This provider handles all standard model providers that work through LiteLLM.
    Each provider name is registered separately for clarity in error messages.
    """

    @classmethod
    def from_config(cls, model_key: str, config: dict[str, Any]) -> BaseModel:
        """Create a LiteLLM model from configuration.

        Args:
            model_key: The key used for this model in the config
            config: The model configuration dictionary

        Returns:
            A new LiteLLMModel instance
        """
        return LiteLLMModel.from_config(model_key, config)
