"""
Base module for LLM model interactions in Arbitrium Framework.
"""

import asyncio
import hashlib
import json
import random
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import litellm

from ..logging import get_contextual_logger
from ..utils.constants import ERROR_PATTERNS
from ..utils.exceptions import ModelResponseError

# Module-level logger for consistent logging
_logger = None


def _get_module_logger() -> Any:
    """Get or create module-level logger."""
    global _logger  # noqa: PLW0603
    if _logger is None:
        _logger = get_contextual_logger("arbitrium.models")
    return _logger


# Retryable error types
_RETRYABLE_ERROR_TYPES = {
    "rate_limit",
    "timeout",
    "connection",
    "service",
    "overloaded",
}

# Permission-related error patterns (not retryable)
_PERMISSION_ERROR_PATTERNS = [
    "permission_denied",
    "service_disabled",
    "api has not been used",
]


def _is_retryable_error_type(error_type: str) -> bool:
    """Check if an error type is retryable."""
    return error_type in _RETRYABLE_ERROR_TYPES


def _check_exception_type(response: Exception) -> tuple[bool, str] | None:
    """Check exception type for known non-retryable errors."""
    exception_name = type(response).__name__.lower()

    if "notfounderror" in exception_name:
        return False, "not_found"

    if "authenticationerror" in exception_name:
        return False, "authentication"

    return None


def _check_permission_errors(error_msg: str) -> bool:
    """Check if error message indicates a permission error."""
    return any(p in error_msg for p in _PERMISSION_ERROR_PATTERNS)


def _match_error_patterns(error_msg: str) -> tuple[bool, str] | None:
    """Match error message against known patterns."""
    for error_type, patterns in ERROR_PATTERNS.items():
        if any(p in error_msg for p in patterns):
            return True, error_type
    return None


def analyze_error_response(response: Any) -> tuple[bool, str]:
    """Analyzes an error response to determine if it's retryable and what type it is."""
    # Extract error message
    error_msg = ""

    # Check if response has explicit error_type attribute
    if hasattr(response, "error"):
        error_msg = str(response.error).lower()
        error_type = getattr(response, "error_type", None)
        if error_type:
            return _is_retryable_error_type(error_type), error_type

    # Handle exception types
    if isinstance(response, Exception):
        error_msg = str(response).lower()
        exception_result = _check_exception_type(response)
        if exception_result:
            return exception_result

    # Check for permission errors
    if _check_permission_errors(error_msg):
        return False, "permission_denied"

    # Match against error patterns
    pattern_result = _match_error_patterns(error_msg)
    if pattern_result:
        return pattern_result

    return False, "general"


# Backoff multipliers for different error types and providers
_BACKOFF_MULTIPLIERS = {
    "rate_limit": {"anthropic": 2.5, "default": 2.0},
    "overloaded": {"anthropic": 3.0, "default": 2.5},
    "timeout": 1.5,
    "connection": 1.8,
    "service": 2.0,
    "general": 1.5,
}


def _get_backoff_multiplier(error_type: str, provider: str) -> float:
    """Get backoff multiplier for error type and provider."""
    provider = provider.lower() if provider else "default"
    multiplier_value = _BACKOFF_MULTIPLIERS.get(
        error_type, _BACKOFF_MULTIPLIERS["general"]
    )

    if isinstance(multiplier_value, dict):
        provider_mult = multiplier_value.get(
            provider, multiplier_value["default"]
        )
        return float(provider_mult) if provider_mult is not None else 1.5

    if isinstance(multiplier_value, (int, float)):
        return float(multiplier_value)

    return 1.5  # Default fallback


def _get_jitter_range(error_type: str, provider: str) -> float:
    """Get jitter range for error type and provider."""
    if error_type in ["rate_limit", "overloaded"] and provider == "anthropic":
        return 0.05
    return 0.1


def _calculate_jittered_delay(
    current_delay: float,
    max_delay: float,
    multiplier: float,
    jitter_range: float,
) -> float:
    """Calculate delay with jitter applied."""
    jitter_factor = 1.0 + random.uniform(-jitter_range, jitter_range)
    return min(current_delay * multiplier, max_delay) * jitter_factor


def _check_timeout_remaining(
    start_time: float,
    total_timeout: float,
    logger: Any | None,
) -> float | None:
    """Check if there's time remaining for retry. Returns remaining time or None."""
    remaining_time = total_timeout - (time.monotonic() - start_time)
    if remaining_time <= 0:
        if logger:
            logger.error("No time left for retry. Stopping retries.")
        return None
    return remaining_time


def _check_min_delay(
    actual_delay: float,
    initial_delay: float,
    error_type: str,
    logger: Any | None,
) -> bool:
    """Check if delay meets minimum threshold. Returns True if sufficient."""
    min_delay_factor = (
        0.5 if error_type in ["rate_limit", "overloaded"] else 0.25
    )
    if actual_delay < initial_delay * min_delay_factor:
        if logger:
            logger.error(
                "Not enough time for proper retry delay. Stopping retries."
            )
        return False
    return True


# Helper for retry delay calculation
async def _calculate_retry_delay(
    current_delay: float,
    start_time: float,
    total_timeout: float,
    initial_delay: float,
    max_delay: float,
    logger: Any | None = None,
    error_type: str = "general",
    provider: str = "default",
) -> float | None:
    """Calculates the delay for the next retry attempt with jitter."""
    multiplier = _get_backoff_multiplier(error_type, provider)
    jitter_range = _get_jitter_range(error_type, provider)
    jittered_delay = _calculate_jittered_delay(
        current_delay, max_delay, multiplier, jitter_range
    )

    # Check timeout
    remaining_time = _check_timeout_remaining(
        start_time, total_timeout, logger
    )
    if remaining_time is None:
        return None

    # Check minimum delay
    actual_delay = min(jittered_delay, remaining_time)
    if not _check_min_delay(actual_delay, initial_delay, error_type, logger):
        return None

    # Sleep and return next delay
    await asyncio.sleep(actual_delay)
    return min(current_delay * multiplier, max_delay)


class ResponseCache:
    """Caches LLM responses to avoid redundant API calls and reduce costs.

    Uses SQLite to store responses keyed by model name, prompt, temperature,
    and max_tokens. This dramatically reduces costs during development,
    testing, and when re-running tournaments with similar questions.

    Args:
        db_path: Path to SQLite database file (default: arbitrium_cache.db)
        enabled: Whether caching is enabled (default: True)

    Example:
        ```python
        cache = ResponseCache("cache.db")

        # Check cache before API call
        cached = cache.get("gpt-4o", prompt, 0.7, 2048)
        if cached:
            response, cost = cached
            return ModelResponse(response, cost=cost)

        # ... make API call ...

        # Save to cache
        cache.set("gpt-4o", prompt, 0.7, 2048, response.content, response.cost)
        ```
    """

    def __init__(
        self, db_path: str | Path = "arbitrium_cache.db", enabled: bool = True
    ) -> None:
        """Initialize response cache with SQLite database."""
        self.enabled = enabled
        if not self.enabled:
            self.conn = None
            return

        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_table()

    def _create_table(self) -> None:
        """Create the responses table if it doesn't exist."""
        if not self.conn:
            return

        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS responses (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                cost REAL NOT NULL,
                timestamp INTEGER NOT NULL
            )
            """
        )
        self.conn.commit()

    def _hash_key(
        self, model_name: str, prompt: str, temperature: float, max_tokens: int
    ) -> str:
        """Generate a cache key from request parameters."""
        key_data = {
            "model": model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()

    def get(
        self, model_name: str, prompt: str, temperature: float, max_tokens: int
    ) -> tuple[str, float] | None:
        """Check cache for a matching response.

        Args:
            model_name: Name of the LLM model
            prompt: The prompt text
            temperature: Temperature parameter
            max_tokens: Max tokens parameter

        Returns:
            Tuple of (response_text, cost) if found, None otherwise
        """
        if not self.enabled or not self.conn:
            return None

        key = self._hash_key(model_name, prompt, temperature, max_tokens)
        cursor = self.conn.execute(
            "SELECT response, cost FROM responses WHERE cache_key = ?", (key,)
        )
        result = cursor.fetchone()
        if result:
            return (result[0], result[1])
        return None

    def set(
        self,
        model_name: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        response: str,
        cost: float,
    ) -> None:
        """Save a response to the cache.

        Args:
            model_name: Name of the LLM model
            prompt: The prompt text
            temperature: Temperature parameter
            max_tokens: Max tokens parameter
            response: The response text to cache
            cost: Cost of the API call in USD
        """
        if not self.enabled or not self.conn:
            return

        key = self._hash_key(model_name, prompt, temperature, max_tokens)
        timestamp = int(time.time())
        self.conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
            (key, response, cost, timestamp),
        )
        self.conn.commit()

    def clear(self) -> None:
        """Clear all cached responses."""
        if not self.enabled or not self.conn:
            return

        self.conn.execute("DELETE FROM responses")
        self.conn.commit()

    def stats(self) -> dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics (total_entries, etc.)
        """
        if not self.enabled or not self.conn:
            return {"total_entries": 0}

        cursor = self.conn.execute("SELECT COUNT(*) FROM responses")
        count = cursor.fetchone()[0]
        return {"total_entries": count}

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self) -> None:
        """Ensure connection is closed on deletion."""
        self.close()


class ModelResponse:
    """Represents a response from an LLM model."""

    def __init__(
        self,
        content: str,
        error: str | None = None,
        error_type: str | None = None,
        provider: str | None = None,
        cost: float = 0.0,
    ):
        """
        Initialize a model response.

        Args:
            content: The text content of the response
            error: Error message if the request failed, None otherwise
            error_type: Type of error for better handling (e.g., "rate_limit", "timeout")
            provider: The provider that generated this response/error
            cost: Cost of the API call in USD
        """
        self.content = content
        self.error = error
        self.error_type = error_type
        self.provider = provider
        self.cost = cost
        self.is_successful = error is None

    @classmethod
    def create_success(
        cls, content: str, cost: float = 0.0
    ) -> "ModelResponse":
        """Create a successful response."""
        return cls(content=content, cost=cost)

    @classmethod
    def create_error(
        cls,
        error_message: str,
        error_type: str | None = None,
        provider: str | None = None,
    ) -> "ModelResponse":
        """Create an error response."""
        return cls(
            content=f"Error: {error_message}",
            error=error_message,
            error_type=error_type,
            provider=provider,
        )

    def is_error(self) -> bool:
        """Check if this response represents an error."""
        return self.error is not None


class BaseModel(ABC):
    """Abstract base class for LLM models."""

    def __init__(
        self,
        model_key: str,
        model_name: str,
        display_name: str,
        provider: str,
        max_tokens: int,
        temperature: float,
        context_window: int | None = None,
        use_llm_compression: bool = True,
        compression_model: str | None = None,
    ):
        """
        Initialize a model.

        Args:
            model_key: Identifier used in config (e.g., 'gpt')
            model_name: Actual model identifier for the API (e.g., 'gpt-4.1')
            display_name: Human-readable name for logs and UI
            provider: Provider name (e.g., 'openai')
            max_tokens: Maximum tokens for the completion/response
            temperature: Temperature setting for response generation
            context_window: Total context window size
            use_llm_compression: Whether to use LLM for prompt compression
            compression_model: Model to use for compression (None = auto-select highest context)
        """
        self.model_key = model_key
        self.model_name = model_name
        self.display_name = display_name
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_llm_compression = use_llm_compression
        self.compression_model = compression_model
        if context_window is None:
            error_message = f"context_window is required for model {model_name}. Please specify context_window in the model configuration."
            raise ValueError(error_message)
        self.context_window = context_window

    @property
    def full_display_name(self) -> str:
        """Returns display name with model name in parentheses for complete identification."""
        return f"{self.display_name} ({self.model_name})"

    @abstractmethod
    async def generate(self, prompt: str) -> ModelResponse:
        """
        Generate a response to the given prompt.

        Args:
            prompt: The prompt to send to the model

        Returns:
            A ModelResponse containing the response text or error
        """


class LiteLLMModel(BaseModel):
    """Implementation of BaseModel using LiteLLM for API access."""

    def __init__(
        self,
        model_key: str,
        model_name: str,
        display_name: str,
        provider: str,
        temperature: float,
        max_tokens: int = 1024,
        context_window: (
            int | None
        ) = None,  # Kept for compatibility with BaseModel
        reasoning: bool = False,
        reasoning_effort: str | None = None,
        model_config: dict[str, Any] | None = None,
        use_llm_compression: bool = True,
        compression_model: str | None = None,
        system_prompt: str | None = None,
    ):
        """Initialize a LiteLLM-backed model."""
        super().__init__(
            model_key=model_key,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,  # Passed to parent but not used in this implementation
            use_llm_compression=use_llm_compression,
            compression_model=compression_model,
        )
        self.reasoning = reasoning
        self.reasoning_effort = reasoning_effort  # "low", "medium", "high"
        self.system_prompt = (
            system_prompt  # Optional system prompt for role-playing
        )

        # Check if this model requires temperature=1.0
        # Models that require this should have force_temp_one: true in their YAML config
        # Example: o4-mini, o3-2025-04-16, etc.
        self.requires_temp_one = (
            model_config is not None
            and hasattr(model_config, "get")
            and model_config.get("force_temp_one", False)
        )

        # Note: LiteLLM logging is centrally disabled in utils/structured_log.py

    def _try_extract_openai_format(
        self, response: Any, logger: Any
    ) -> str | None:
        """Try to extract content from OpenAI/LiteLLM format."""
        if not hasattr(response, "choices") or not response.choices:
            return None
        try:
            content: str = response.choices[0].message.content
            if content and content.strip():
                return content
        except (AttributeError, IndexError) as e:
            logger.debug(f"Failed to extract from choices object: {e}")
        return None

    def _try_extract_dict_format(
        self, response: Any, logger: Any
    ) -> str | None:
        """Try to extract content from dict format."""
        if not isinstance(response, dict):
            return None

        # Try OpenAI dict format
        if "choices" in response:
            try:
                content: str = response["choices"][0]["message"]["content"]
                if content:
                    return content
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Failed to extract from dict choices: {e}")

        # Try Gemini dict format
        if "candidates" in response:
            try:
                content_val: str | None = (
                    response.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text")
                )
                if content_val:
                    return content_val
            except (KeyError, IndexError, TypeError) as e:
                logger.debug(f"Failed to extract from candidates: {e}")

        return None

    def _try_extract_gemini_format(
        self, response: Any, logger: Any
    ) -> str | None:
        """Try to extract content from Gemini object format."""
        if not hasattr(response, "candidates") or not response.candidates:
            return None
        try:
            content: str = response.candidates[0].content.parts[0].text
            if content:
                return content
        except (AttributeError, IndexError) as e:
            logger.debug(f"Failed to extract from candidates object: {e}")
        return None

    def _try_extract_common_attrs(self, response: Any) -> str | None:
        """Try to extract content from common attribute names."""
        for attr in [
            "content",
            "text",
            "completion",
            "answer",
            "response",
            "output",
            "result",
        ]:
            if hasattr(response, attr):
                value = getattr(response, attr)
                if isinstance(value, str) and value.strip():
                    return value
        return None

    def _extract_response_content(self, response: Any) -> str | None:
        """Extracts content from the LLM response with multiple fallback strategies."""
        logger = _get_module_logger()

        # Try different extraction strategies in order
        strategies: list[Any] = [
            lambda: self._try_extract_openai_format(response, logger),
            lambda: self._try_extract_dict_format(response, logger),
            lambda: self._try_extract_gemini_format(response, logger),
            lambda: self._try_extract_common_attrs(response),
        ]

        for strategy in strategies:
            content: str | None = strategy()
            if content:
                return content

        # Last resort: if response is already a string
        if isinstance(response, str) and response.strip():
            logger.warning("Response was already a string, using directly")
            return response

        return None

    def _extract_response_cost(self, response: Any) -> float:
        """Extracts cost from the LLM response."""
        logger = _get_module_logger()

        # LiteLLM adds cost information to the response object
        if hasattr(response, "_hidden_params") and hasattr(
            response._hidden_params, "response_cost"
        ):
            cost: float = response._hidden_params.response_cost
            logger.info(
                f"💰 Cost extracted from _hidden_params.response_cost: ${cost:.4f}"
            )
            return cost
        if hasattr(response, "response_cost"):
            cost_val: float = response.response_cost
            logger.info(
                f"💰 Cost extracted from response_cost: ${cost_val:.4f}"
            )
            return cost_val

        # Enhanced fallback - try to extract from LiteLLM usage calculation
        if hasattr(response, "usage") and response.usage:
            try:
                # Try LiteLLM's completion_cost function if available
                import litellm

                if hasattr(litellm, "completion_cost"):
                    cost_calc: float = litellm.completion_cost(
                        completion_response=response
                    )
                    if cost_calc and cost_calc > 0:
                        logger.info(
                            f"💰 Cost calculated via litellm.completion_cost: ${cost_calc:.4f}"
                        )
                        return cost_calc
            except Exception as e:
                logger.debug(
                    f"Failed to calculate cost via litellm.completion_cost: {e}"
                )

        logger.debug("💰 No cost information found in response, returning 0.0")
        return 0.0

    def _handle_prompt_size_validation(self, prompt: str) -> str | None:
        """Handle prompt size validation - DISABLED to preserve full prompts.

        Users pay for the full context, so we don't truncate anything.
        Large language models can handle large contexts.
        """
        # Always return the full prompt without any truncation or compression
        return prompt

    def _clean_response_content(self, content: str) -> str:
        """Clean response content - preserve markdown formatting.

        User pays for full content including markdown formatting.
        Markdown is useful for readability and structure.
        """
        # Only strip whitespace, preserve all markdown formatting
        return content.strip()

    def _try_extract_content_from_response(
        self, response: Any, cost: float, logger: Any
    ) -> ModelResponse | None:
        """Try to extract content from response, return ModelResponse if successful."""
        content = self._extract_response_content(response)

        if content and content.strip():
            cleaned_content = self._clean_response_content(content)
            return ModelResponse.create_success(cleaned_content, cost=cost)

        # Try fallback
        logger.warning(
            f"{self.display_name} returned a response but content extraction failed. Using str() fallback."
        )
        logger.debug(
            f"Response object type: {type(response)}, has choices: {hasattr(response, 'choices')}"
        )

        fallback_content = str(response)
        if fallback_content and len(fallback_content.strip()) > 10:
            return ModelResponse.create_success(
                fallback_content.strip(), cost=cost
            )

        return None

    def _validate_prompt(
        self, prompt: str
    ) -> tuple[str, ModelResponse | None]:
        """Validate and process prompt. Returns (processed_prompt, error_response)."""
        if not prompt or not prompt.strip():
            return "", ModelResponse.create_error("Empty prompt provided")

        validated_prompt = self._handle_prompt_size_validation(prompt)
        if validated_prompt is None:
            method = (
                "LLM compression" if self.use_llm_compression else "truncation"
            )
            return "", ModelResponse.create_error(
                f"Prompt too large even after {method}"
            )

        return validated_prompt, None

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        """Build messages array with optional system prompt."""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_completion_params(
        self, messages: list[dict[str, str]], logger: Any
    ) -> dict[str, Any]:
        """Build parameters for litellm.acompletion."""
        temperature = (
            1.0 if self.requires_temp_one else float(self.temperature)
        )

        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }

        if self.reasoning_effort:
            params["reasoning_effort"] = self.reasoning_effort
            logger.debug(
                f"Using reasoning_effort={self.reasoning_effort} for {self.display_name}"
            )

        return params

    async def _execute_completion(
        self, params: dict[str, Any], logger: Any
    ) -> ModelResponse:
        """Execute API call and process response."""
        from ..utils.constants import DEFAULT_MODEL_TIMEOUT

        response = await asyncio.wait_for(
            litellm.acompletion(**params), timeout=DEFAULT_MODEL_TIMEOUT
        )
        cost = self._extract_response_cost(response)

        # Try to extract and clean content
        model_response = self._try_extract_content_from_response(
            response, cost, logger
        )
        if not model_response:
            raise ModelResponseError(
                f"Model {self.display_name} returned empty or unusable response",
                model_key=self.model_key,
            )

        # Log the response in structured format
        logger.log_response(
            model_response.content,
            model=self.display_name,
            model_key=self.model_key,
            cost=model_response.cost,
        )
        return model_response

    def _handle_exception(self, exc: Exception, logger: Any) -> ModelResponse:
        """Handle exceptions and return error response."""
        error_type, error_message = self._classify_exception(exc)

        # Log based on exception type
        if isinstance(
            exc,
            (
                litellm.exceptions.RateLimitError,
                litellm.exceptions.Timeout,
                litellm.exceptions.AuthenticationError,
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.InternalServerError,
                ModelResponseError,
            ),
        ):
            logger.warning(
                f"{error_message}. model={self.model_name}, provider={self.provider}"
            )
        else:
            logger.error(
                f"API error with {self.display_name}: {error_message}",
                exc_info=True,
            )

        return ModelResponse.create_error(
            error_message, error_type=error_type, provider=self.provider
        )

    async def generate(self, prompt: str) -> ModelResponse:
        """Generates a response to the given prompt using LiteLLM."""
        logger = _get_module_logger()

        # Validate prompt
        validated_prompt, error = self._validate_prompt(prompt)
        if error:
            return error

        # Log the prompt in structured format
        logger.log_prompt(
            validated_prompt, model=self.display_name, model_key=self.model_key
        )

        # Build request parameters
        messages = self._build_messages(validated_prompt)
        params = self._build_completion_params(messages, logger)

        # Execute request
        try:
            return await self._execute_completion(params, logger)
        except Exception as e:
            return self._handle_exception(e, logger)

    def _classify_exception(self, exc: Exception) -> tuple[str, str]:
        """Classifies an exception into an error type and message."""
        if isinstance(exc, litellm.exceptions.RateLimitError):
            return (
                "rate_limit",
                f"Rate limit exceeded with {self.display_name}: {exc}",
            )
        if isinstance(exc, (litellm.exceptions.Timeout, asyncio.TimeoutError)):
            return (
                "timeout",
                f"Request timed out for {self.display_name}: {exc}",
            )
        if isinstance(exc, litellm.exceptions.AuthenticationError):
            return (
                "authentication",
                f"Authentication failed with {self.display_name}: {exc}",
            )
        if isinstance(exc, litellm.exceptions.ServiceUnavailableError):
            return (
                "service_unavailable",
                f"Service unavailable for {self.display_name}: {exc}",
            )
        if isinstance(exc, litellm.exceptions.InternalServerError):
            return (
                "overloaded" if "overload" in str(exc).lower() else "service"
            ), f"Server error with {self.display_name}: {exc}"
        if isinstance(exc, ModelResponseError):
            return "model_response_error", str(exc)
        return (
            "general",
            f"Unexpected API error with {self.display_name}: {exc}",
        )

    @classmethod
    def _convert_info_to_dict(
        cls, info: Any, model_name: str, logger: Any
    ) -> dict[str, Any]:
        """Convert model info object to dictionary."""
        if isinstance(info, dict):
            return info
        if hasattr(info, "model_dump"):
            result = info.model_dump()
            if isinstance(result, dict):
                return result
        if hasattr(info, "dict"):
            result = info.dict()
            if isinstance(result, dict):
                return result
        try:
            result = dict(info)
            if isinstance(result, dict):
                return result
        except (TypeError, ValueError):
            pass
        logger.debug(f"Could not convert model info to dict for {model_name}")
        return {}

    @classmethod
    def _get_model_info_from_litellm(cls, model_name: str) -> dict[str, Any]:
        """
        Attempts to get model information from LiteLLM.

        Args:
            model_name: The model name to query

        Returns:
            Dictionary with model info, or empty dict if not found
        """
        logger = _get_module_logger()
        try:
            info = litellm.get_model_info(model_name)
            logger.debug(
                f"Retrieved model info from LiteLLM for {model_name}: {info}"
            )
            return cls._convert_info_to_dict(info, model_name, logger)
        except Exception as e:
            logger.debug(
                f"Could not retrieve model info from LiteLLM for {model_name}: {e}"
            )
            return {}

    @classmethod
    def _validate_required_fields(
        cls, model_key: str, model_config: dict[str, Any]
    ) -> None:
        """Validate required configuration fields."""
        required_fields = ["model_name", "provider"]
        for field in required_fields:
            if field not in model_config:
                raise ValueError(
                    f"Required field '{field}' missing in model configuration for {model_key}"
                )

    @classmethod
    def _auto_detect_context_window(
        cls,
        model_key: str,
        model_config: dict[str, Any],
        litellm_info: dict[str, Any],
        logger: Any,
    ) -> None:
        """Auto-detect and set context_window if not provided."""
        if (
            "context_window" not in model_config
            or model_config["context_window"] is None
        ):
            context_window = litellm_info.get("max_input_tokens")
            if context_window:
                logger.info(
                    f"Auto-detected context_window={context_window} for {model_key} from LiteLLM"
                )
                model_config["context_window"] = context_window
            else:
                raise ValueError(
                    f"context_window not provided for {model_key} and could not be auto-detected from LiteLLM. "
                    f"Please specify context_window in config or ensure model is supported by LiteLLM."
                )

    @classmethod
    def _auto_detect_max_tokens(
        cls,
        model_key: str,
        model_config: dict[str, Any],
        litellm_info: dict[str, Any],
        logger: Any,
    ) -> None:
        """Auto-detect and set max_tokens if not provided, capped at 25% of context."""
        if (
            "max_tokens" not in model_config
            or model_config["max_tokens"] is None
        ):
            max_output_tokens = litellm_info.get(
                "max_output_tokens"
            ) or litellm_info.get("max_tokens")
            if max_output_tokens:
                context_win = model_config.get("context_window", 128000)
                safe_max_tokens = min(
                    max_output_tokens, int(context_win * 0.25)
                )
                logger.info(
                    f"Auto-detected max_tokens={safe_max_tokens} for {model_key} (from LiteLLM: {max_output_tokens}, capped at 25% of context)"
                )
                model_config["max_tokens"] = safe_max_tokens
            else:
                raise ValueError(
                    f"max_tokens not provided for {model_key} and could not be auto-detected from LiteLLM. "
                    f"Please specify max_tokens in config or ensure model is supported by LiteLLM."
                )

    @classmethod
    def _validate_temperature(
        cls, model_key: str, model_config: dict[str, Any]
    ) -> None:
        """Validate that temperature is provided in config."""
        if "temperature" not in model_config:
            raise ValueError(
                f"temperature is required in model configuration for {model_key}"
            )

    @classmethod
    def _validate_and_get_reasoning_effort(
        cls, model_key: str, model_config: dict[str, Any], logger: Any
    ) -> str | None:
        """Validate and return reasoning_effort setting."""
        reasoning_effort = model_config.get("reasoning_effort")
        if reasoning_effort:
            supported_efforts = ["low", "medium", "high"]
            if reasoning_effort not in supported_efforts:
                logger.warning(
                    f"Invalid reasoning_effort '{reasoning_effort}' for {model_key}. Must be one of {supported_efforts}"
                )
                return None
            logger.info(
                f"Using reasoning_effort={reasoning_effort} for {model_key}"
            )
        return reasoning_effort

    @classmethod
    def _get_compression_settings(
        cls, model_config: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Get LLM compression settings from config.

        Returns:
            Tuple of (use_llm_compression, compression_model).
            compression_model is None if auto-selection should be used.
        """
        use_llm_compression = model_config.get("llm_compression", True)
        compression_model = model_config.get("compression_model", None)
        return use_llm_compression, compression_model

    @classmethod
    def _get_and_log_system_prompt(
        cls, model_key: str, model_config: dict[str, Any], logger: Any
    ) -> str | None:
        """Get system prompt from config and log if present."""
        system_prompt = model_config.get("system_prompt")
        if system_prompt:
            logger.info(
                f"Using system_prompt for {model_key}: {system_prompt[:100]}..."
            )
        return system_prompt

    @classmethod
    def from_config(
        cls, model_key: str, model_config: dict[str, Any]
    ) -> "LiteLLMModel":
        """
        Create a model instance from configuration.

        Auto-detects max_tokens and context_window from LiteLLM if not provided in config.

        Args:
            model_key: The key used for this model in the config
            model_config: The model configuration dictionary

        Returns:
            A new LiteLLMModel instance
        """
        logger = _get_module_logger()

        # Validate required fields
        cls._validate_required_fields(model_key, model_config)

        # Get model info from LiteLLM for auto-detection
        model_name = model_config["model_name"]
        litellm_info = cls._get_model_info_from_litellm(model_name)

        # Auto-detect missing configuration
        cls._auto_detect_context_window(
            model_key, model_config, litellm_info, logger
        )
        cls._auto_detect_max_tokens(
            model_key, model_config, litellm_info, logger
        )
        cls._validate_temperature(model_key, model_config)

        # Get optional settings
        reasoning_effort = cls._validate_and_get_reasoning_effort(
            model_key, model_config, logger
        )
        use_llm_compression, compression_model = cls._get_compression_settings(
            model_config
        )
        system_prompt = cls._get_and_log_system_prompt(
            model_key, model_config, logger
        )

        # Create and return model instance
        return cls(
            model_key=model_key,
            model_name=model_config["model_name"],
            display_name=model_config.get("display_name")
            or model_config["model_name"],
            provider=model_config["provider"],
            max_tokens=model_config["max_tokens"],
            temperature=float(model_config["temperature"]),
            context_window=model_config["context_window"],
            reasoning=model_config.get("reasoning", False),
            reasoning_effort=reasoning_effort,
            model_config=model_config,
            use_llm_compression=use_llm_compression,
            compression_model=compression_model,
            system_prompt=system_prompt,
        )


def _check_timeout_exceeded(
    start_time: float, total_timeout: int, logger: Any | None
) -> bool:
    """Check if total timeout has been exceeded."""
    if time.monotonic() - start_time > total_timeout:
        if logger:
            logger.error(
                f"Total timeout ({total_timeout}s) exceeded. Stopping retries."
            )
        return True
    return False


async def _handle_retry_response(
    response: ModelResponse,
    attempt: int,
    max_attempts: int,
    current_delay: float,
    start_time: float,
    total_timeout: int,
    initial_delay_val: float,
    max_delay_val: float,
    logger: Any | None,
    provider: str,
) -> float | None:
    """Handle a retry response and return next delay, or None to stop retrying."""
    should_retry, error_type = analyze_error_response(response)
    if not should_retry or attempt >= max_attempts:
        return None

    if logger:
        logger.warning(
            f"Attempt {attempt}/{max_attempts} failed for {provider}. Retrying... Error: {response.error}"
        )

    next_delay = await _calculate_retry_delay(
        current_delay,
        start_time,
        total_timeout,
        initial_delay_val,
        max_delay_val,
        logger,
        error_type,
        provider,
    )
    return next_delay


async def _handle_retry_exception(
    exception: Exception,
    attempt: int,
    max_attempts: int,
    current_delay: float,
    start_time: float,
    total_timeout: int,
    initial_delay_val: float,
    max_delay_val: float,
    logger: Any | None,
    provider: str,
) -> float | None:
    """Handle a retry exception and return next delay, or None to stop retrying."""
    if attempt >= max_attempts:
        return None

    if logger:
        logger.warning(
            f"Attempt {attempt}/{max_attempts} failed for {provider} with exception. Retrying... Error: {exception}"
        )

    _, error_type = analyze_error_response(exception)
    next_delay = await _calculate_retry_delay(
        current_delay,
        start_time,
        total_timeout,
        initial_delay_val,
        max_delay_val,
        logger,
        error_type,
        provider,
    )
    return next_delay


async def run_with_retry(
    model: BaseModel,
    prompt: str,
    max_attempts: int = 5,
    initial_delay: int | None = None,
    max_delay: int | None = None,
    total_timeout: int = 900,  # 15 minutes total timeout (reduced from 30 min)
    logger: Any | None = None,
) -> ModelResponse:
    """Runs a model generation with retry logic."""
    from ..utils.constants import PROVIDER_RETRY_DELAYS as provider_delays

    provider = model.provider.lower() if model.provider else "default"
    provider_config = provider_delays.get(provider, provider_delays["default"])
    initial_delay_val = (
        initial_delay
        if initial_delay is not None
        else provider_config["initial"]
    )
    max_delay_val = (
        max_delay if max_delay is not None else provider_config["max"]
    )

    current_delay: float = float(initial_delay_val)
    start_time = time.monotonic()

    for attempt in range(1, max_attempts + 1):
        if _check_timeout_exceeded(start_time, total_timeout, logger):
            return ModelResponse.create_error(
                f"Exceeded total timeout of {total_timeout}s"
            )

        try:
            response = await model.generate(prompt)
            if not response.is_error():
                return response

            next_delay = await _handle_retry_response(
                response,
                attempt,
                max_attempts,
                current_delay,
                start_time,
                total_timeout,
                initial_delay_val,
                max_delay_val,
                logger,
                provider,
            )
            if next_delay is None:
                return response
            current_delay = next_delay

        except Exception as e:
            next_delay = await _handle_retry_exception(
                e,
                attempt,
                max_attempts,
                current_delay,
                start_time,
                total_timeout,
                initial_delay_val,
                max_delay_val,
                logger,
                provider,
            )
            if next_delay is None:
                _, error_type = analyze_error_response(e)
                return ModelResponse.create_error(
                    str(e), error_type=error_type, provider=provider
                )
            current_delay = next_delay

    return ModelResponse.create_error(
        "Max attempts reached without a successful response."
    )
