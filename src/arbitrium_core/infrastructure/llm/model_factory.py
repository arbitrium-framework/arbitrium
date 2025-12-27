from typing import Any

from arbitrium_core.ports.llm import BaseModel
from arbitrium_core.shared.logging import get_contextual_logger

logger = get_contextual_logger("arbitrium.infrastructure.llm.model_factory")


async def ensure_model_instances(
    models: dict[str, Any],
) -> dict[str, Any]:
    from arbitrium_core.infrastructure.llm.litellm_adapter import LiteLLMModel

    result = {}
    for key, model in models.items():
        if isinstance(model, BaseModel):
            result[key] = model
        elif isinstance(model, dict):
            try:
                instance = await LiteLLMModel.from_config(
                    model_key=model.get("name", key),
                    model_config=model,
                )
                result[key] = instance
            except Exception as e:
                logger.error(f"Failed to create model from config {key}: {e}")
        else:
            logger.warning(f"Unknown model type for {key}: {type(model)}")
    return result


async def ensure_single_model_instance(model: Any, key: str = "model") -> Any:
    from arbitrium_core.infrastructure.llm.litellm_adapter import LiteLLMModel

    if isinstance(model, BaseModel):
        return model
    elif isinstance(model, dict):
        try:
            return await LiteLLMModel.from_config(
                model_key=model.get("name", key),
                model_config=model,
            )
        except Exception as e:
            logger.error(f"Failed to create model from config {key}: {e}")
            return None
    else:
        logger.warning(f"Unknown model type for {key}: {type(model)}")
        return None
