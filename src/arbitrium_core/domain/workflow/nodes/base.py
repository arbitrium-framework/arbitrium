import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, ClassVar, TypeVar

from arbitrium_core.support.logging import get_contextual_logger

T = TypeVar("T")

logger = get_contextual_logger(__name__)


class PortType(Enum):
    MODELS = "models"
    MODEL = "model"
    RESPONSES = "responses"
    SCORES = "scores"
    RANKINGS = "rankings"
    RESULTS = "results"
    INSIGHTS = "insights"
    BOOLEAN = "boolean"
    NUMBER = "number"
    INTEGER = "integer"
    STRING = "string"
    STRING_MATRIX = "string_matrix"  # 2D array: [[run1_llm1, run1_llm2], [run2_llm1, run2_llm2], ...]
    ANY = "any"


@dataclass
class Port:
    name: str
    port_type: PortType
    required: bool = True
    description: str = ""


@dataclass
class ExecutionContext:
    question: str = ""
    round_num: int = 0
    execution_id: str = ""
    node_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    broadcast: Any = None  # Callable for WebSocket broadcasting
    models: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)


class NodeProperty:
    def __init__(self, key: str, type_: type = str, default: Any = None):
        self.key = key
        self.type_ = type_
        self.default = default

    def __get__(self, obj: Any, _objtype: Any = None) -> Any:
        if obj is None:
            return self
        value = obj.node_properties.get(self.key, self.default)
        if value is None:
            return self.default
        try:
            return self.type_(value)
        except (ValueError, TypeError):
            return self.default


F = TypeVar("F", bound=Callable[..., Any])


def require_inputs(*required_keys: str) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(
            self: "BaseNode",
            inputs: dict[str, Any],
            context: "ExecutionContext",
        ) -> dict[str, Any]:
            valid, error_output = self.validate_required_inputs(
                inputs, *required_keys
            )
            if not valid:
                return error_output
            result: dict[str, Any] = await func(self, inputs, context)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


class BaseNode(ABC):
    NODE_TYPE: ClassVar[str] = ""
    DISPLAY_NAME: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = ""
    DESCRIPTION: ClassVar[str] = ""
    HIDDEN: ClassVar[bool] = False
    INPUTS: ClassVar[list[Port]] = []
    OUTPUTS: ClassVar[list[Port]] = []
    PROPERTIES: ClassVar[dict[str, Any]] = {}
    DYNAMIC_INPUTS: ClassVar[dict[str, Any] | None] = None

    def __init__(self, node_id: str, properties: dict[str, Any] | None = None):
        self.node_id = node_id
        self.node_properties = properties or {}
        self._validate_properties()

    def _validate_properties(self) -> None:
        if not self.PROPERTIES:
            return

        schema_keys = set(self.PROPERTIES.keys())
        actual_keys = set(self.node_properties.keys())

        # Check for unknown properties
        unknown = actual_keys - schema_keys
        if unknown:
            logger.warning(
                f"Node {self.node_id} ({self.NODE_TYPE}): Unknown properties {unknown}. "
                f"Expected properties: {schema_keys}. "
                f"This may indicate a typo or outdated workflow file."
            )

        # Check for type mismatches
        for key, value in self.node_properties.items():
            if key not in self.PROPERTIES:
                continue

            schema = self.PROPERTIES[key]
            expected_type = schema.get("type")

            if expected_type == "array" and not isinstance(value, list):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be array, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "object" and not isinstance(value, dict):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be object, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "string" and not isinstance(value, str):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be string, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "number" and not isinstance(
                value, (int, float)
            ):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be number, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )
            elif expected_type == "boolean" and not isinstance(value, bool):
                logger.error(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Property '{key}' should be boolean, "
                    f"got {type(value).__name__}. Value: {value!r}"
                )

        # Check for missing required properties (no default value in schema)
        for key, schema in self.PROPERTIES.items():
            if "default" not in schema and key not in actual_keys:
                logger.warning(
                    f"Node {self.node_id} ({self.NODE_TYPE}): Missing property '{key}' "
                    f"(no default value defined)"
                )

    @abstractmethod
    async def execute(
        self,
        inputs: dict[str, Any],
        context: ExecutionContext,
    ) -> dict[str, Any]:
        pass

    def get_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {
            "node_type": self.NODE_TYPE,
            "display_name": self.DISPLAY_NAME,
            "category": self.CATEGORY,
            "description": self.DESCRIPTION,
            "inputs": [
                {
                    "name": p.name,
                    "port_type": p.port_type.value,
                    "required": p.required,
                    "description": p.description,
                }
                for p in self.INPUTS
            ],
            "outputs": [
                {
                    "name": p.name,
                    "port_type": p.port_type.value,
                    "description": p.description,
                }
                for p in self.OUTPUTS
            ],
            "properties": self.PROPERTIES,
        }
        if self.DYNAMIC_INPUTS:
            schema["dynamic_inputs"] = self.DYNAMIC_INPUTS
        return schema

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        errors = []
        for port in self.INPUTS:
            if port.required and port.name not in inputs:
                errors.append(f"Missing required input: {port.name}")
        return errors

    def validate_required_inputs(
        self, inputs: dict[str, Any], *required_keys: str
    ) -> tuple[bool, dict[str, Any]]:
        missing = []
        for key in required_keys:
            value = inputs.get(key)
            if not value:  # Covers None, {}, [], "", 0, False
                missing.append(key)

        if missing:
            logger.warning(
                f"Node {self.node_id} ({self.NODE_TYPE}) missing required inputs: {missing}"
            )
            return False, self._get_empty_output()

        return True, {}

    def _get_empty_output(self) -> dict[str, Any]:
        mapping_ports = {
            PortType.MODELS,
            PortType.RESPONSES,
            PortType.SCORES,
        }
        return {
            port.name: (
                {}
                if port.port_type in mapping_ports
                else [] if port.port_type == PortType.RANKINGS else ""
            )
            for port in self.OUTPUTS
        }

    async def broadcast_result(
        self, event_type: str, data: dict[str, Any], context: ExecutionContext
    ) -> None:
        if context.broadcast:
            await context.broadcast(
                {
                    "type": event_type,
                    "node_id": self.node_id,
                    "data": data,
                }
            )

    async def ensure_models_or_empty(
        self, models_input: Any, single: bool = False
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        from arbitrium_core.adapters.llm.factory import (
            ensure_model_instances,
            ensure_single_model_instance,
        )

        if single:
            model = await ensure_single_model_instance(models_input, "model")
            if not model:
                return {}, self._get_empty_output()
            return {"model": model}, None
        else:
            models = await ensure_model_instances(models_input)
            if not models:
                return {}, self._get_empty_output()
            return models, None


from arbitrium_core.support.constants import MAX_MULTI_INPUTS


def build_group_outputs(
    groups: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        f"group_{i + 1}": groups[i] if i < len(groups) else {}
        for i in range(MAX_MULTI_INPUTS)
    }


def merge_indexed_dicts(
    inputs: dict[str, Any],
    prefix: str,
    separator: str = "_",
) -> dict[str, Any]:
    """Merge dict-type indexed inputs into single dict.

    Collects inputs like 'input_1', 'input_2', etc. and merges them.
    Used by MergeNode and similar flow control nodes.
    """
    result: dict[str, Any] = {}
    for i in range(1, MAX_MULTI_INPUTS + 1):
        key = f"{prefix}{separator}{i}" if separator else f"{prefix}{i}"
        value = inputs.get(key, {})
        if value:
            result.update(value)
    return result


def merge_indexed_lists(
    inputs: dict[str, Any],
    prefix: str,
    separator: str = "_",
) -> list[Any]:
    """Merge list-type indexed inputs into single list.

    Collects inputs like 'insights1', 'insights2', etc. and extends them.
    Used by AccumulateInsightsNode and similar aggregation nodes.
    """
    result: list[Any] = []
    for i in range(1, MAX_MULTI_INPUTS + 1):
        key = f"{prefix}{separator}{i}" if separator else f"{prefix}{i}"
        value = inputs.get(key, [])
        if value:
            result.extend(value)
    return result


def partition_models(
    models: dict[str, Any],
    scores: dict[str, float],
    check_fn: Callable[[float], bool],
) -> tuple[dict[str, Any], dict[str, Any]]:
    passed = {k: v for k, v in models.items() if check_fn(scores.get(k, 0))}
    failed = {
        k: v for k, v in models.items() if not check_fn(scores.get(k, 0))
    }
    return passed, failed


def format_responses(responses: dict[str, str], separator: str = "===") -> str:
    return "\n\n".join(
        f"{separator} {name} {separator}\n{text}"
        for name, text in responses.items()
    )


def build_evaluation_prompt(
    question: str,
    responses: dict[str, str],
    criteria: str = "",
) -> str:
    formatted = format_responses(responses)
    parts = [
        f"Question: {question}",
        "Evaluate each response on a scale of 1-10:",
        formatted,
    ]
    if criteria:
        parts.append(criteria)
    parts.append(
        "Provide a score for each model in the format:\nMODEL_NAME: SCORE/10"
    )
    return "\n\n".join(parts)


async def safe_generate(model: Any, prompt: str) -> tuple[str, bool]:
    try:
        response = await model.generate(prompt)
        if response.is_error():
            logger.warning(
                f"Model generation returned error response: {response}"
            )
            return "", False
        return response.content, True
    except Exception as e:
        logger.error(f"Exception during model generation: {e}", exc_info=True)
        return "", False


async def parallel_generate(
    models: dict[str, Any],
    prompt_fn: Callable[[str, Any], str],
) -> dict[str, str]:
    async def generate_one(key: str, model: Any) -> tuple[str, str]:
        prompt = prompt_fn(key, model)
        content, _ = await safe_generate(model, prompt)
        return key, content

    tasks = [generate_one(k, m) for k, m in models.items()]
    results = await asyncio.gather(*tasks)
    return {k: v for k, v in results if v}


def rank_by_scores(
    models: dict[str, Any],
    scores: dict[str, float],
    reverse: bool = True,
) -> list[str]:
    return sorted(
        models.keys(), key=lambda k: scores.get(k, 0), reverse=reverse
    )


def split_by_rank(
    models: dict[str, Any],
    ranked_keys: list[str],
    keep_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kept = {k: models[k] for k in ranked_keys[:keep_count]}
    removed = {k: models[k] for k in ranked_keys[keep_count:]}
    return kept, removed
