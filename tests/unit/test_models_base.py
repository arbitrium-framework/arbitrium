"""
Unit tests for model base classes and utilities.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from arbitrium.models.base import (
    BaseModel,
    LiteLLMModel,
    ModelResponse,
    _calculate_retry_delay,
    analyze_error_response,
    run_with_retry,
)


class TestModelResponse:
    """Tests for ModelResponse class."""

    def test_create_success(self) -> None:
        """Test creating a successful response."""
        response = ModelResponse.create_success("Test content", cost=0.01)

        assert response.content == "Test content"
        assert response.error is None
        assert response.is_successful is True
        assert response.cost == 0.01
        assert not response.is_error()

    def test_create_error(self) -> None:
        """Test creating an error response."""
        response = ModelResponse.create_error(
            "Error message", error_type="rate_limit", provider="openai"
        )

        assert "Error message" in response.content
        assert response.error == "Error message"
        assert response.error_type == "rate_limit"
        assert response.provider == "openai"
        assert response.is_successful is False
        assert response.is_error()

    def test_response_initialization(self) -> None:
        """Test ModelResponse initialization."""
        response = ModelResponse(
            content="Test",
            error="Some error",
            error_type="timeout",
            provider="anthropic",
            cost=0.02,
        )

        assert response.content == "Test"
        assert response.error == "Some error"
        assert response.error_type == "timeout"
        assert response.provider == "anthropic"
        assert response.cost == 0.02
        assert not response.is_successful


class TestAnalyzeErrorResponse:
    """Tests for analyze_error_response function."""

    def test_rate_limit_error(self) -> None:
        """Test analyzing rate limit error."""
        error_response = MagicMock()
        error_response.error = "rate limit exceeded"
        error_response.error_type = "rate_limit"

        should_retry, error_type = analyze_error_response(error_response)

        assert should_retry is True
        assert error_type == "rate_limit"

    def test_timeout_error(self) -> None:
        """Test analyzing timeout error."""
        error_response = MagicMock()
        error_response.error = "request timed out"
        error_response.error_type = "timeout"

        should_retry, error_type = analyze_error_response(error_response)

        assert should_retry is True
        assert error_type == "timeout"

    def test_authentication_error_from_exception(self) -> None:
        """Test analyzing authentication error from exception."""

        class AuthenticationError(Exception):
            pass

        error = AuthenticationError("auth failed")

        should_retry, error_type = analyze_error_response(error)

        assert should_retry is False
        assert error_type == "authentication"

    def test_not_found_error(self) -> None:
        """Test analyzing not found error."""

        class NotFoundError(Exception):
            pass

        error = NotFoundError("model not found")

        should_retry, error_type = analyze_error_response(error)

        assert should_retry is False
        assert error_type == "not_found"

    def test_permission_denied_error(self) -> None:
        """Test analyzing permission denied error."""
        error_response = MagicMock()
        error_response.error = "permission_denied: api has not been used"
        del error_response.error_type

        should_retry, error_type = analyze_error_response(error_response)

        assert should_retry is False
        assert error_type == "permission_denied"

    def test_overloaded_error(self) -> None:
        """Test analyzing overloaded error."""
        error_response = MagicMock()
        error_response.error = "service is overloaded"
        error_response.error_type = "overloaded"

        should_retry, error_type = analyze_error_response(error_response)

        assert should_retry is True
        assert error_type == "overloaded"

    def test_general_error(self) -> None:
        """Test analyzing general error."""
        error_response = MagicMock()
        error_response.error = "unknown error occurred"
        del error_response.error_type

        _should_retry, error_type = analyze_error_response(error_response)

        assert error_type == "general"


class TestCalculateRetryDelay:
    """Tests for _calculate_retry_delay function."""

    @pytest.mark.asyncio
    async def test_calculate_delay_rate_limit(self) -> None:
        """Test calculating delay for rate limit error."""
        import time

        start_time = time.monotonic()

        delay = await _calculate_retry_delay(
            current_delay=1.0,
            start_time=start_time,
            total_timeout=60.0,
            initial_delay=1.0,
            max_delay=30.0,
            logger=None,
            error_type="rate_limit",
            provider="openai",
        )

        assert delay is not None
        assert delay > 1.0
        assert delay <= 30.0

    @pytest.mark.asyncio
    async def test_calculate_delay_overloaded_anthropic(self) -> None:
        """Test calculating delay for overloaded error with Anthropic."""
        import time

        start_time = time.monotonic()

        delay = await _calculate_retry_delay(
            current_delay=1.0,
            start_time=start_time,
            total_timeout=60.0,
            initial_delay=1.0,
            max_delay=30.0,
            logger=None,
            error_type="overloaded",
            provider="anthropic",
        )

        assert delay is not None
        assert delay > 1.0

    @pytest.mark.asyncio
    async def test_calculate_delay_timeout_exceeded(self) -> None:
        """Test that delay returns None when timeout exceeded."""
        import time

        start_time = time.monotonic() - 61

        delay = await _calculate_retry_delay(
            current_delay=1.0,
            start_time=start_time,
            total_timeout=60.0,
            initial_delay=1.0,
            max_delay=30.0,
            logger=None,
            error_type="general",
            provider="default",
        )

        assert delay is None

    @pytest.mark.asyncio
    async def test_calculate_delay_respects_max_delay(self) -> None:
        """Test that calculated delay respects max_delay."""
        import time

        start_time = time.monotonic()

        delay = await _calculate_retry_delay(
            current_delay=25.0,
            start_time=start_time,
            total_timeout=120.0,
            initial_delay=1.0,
            max_delay=30.0,
            logger=None,
            error_type="general",
            provider="default",
        )

        assert delay is not None
        assert delay <= 30.0


class TestBaseModel:
    """Tests for BaseModel class."""

    def test_base_model_initialization(self) -> None:
        """Test BaseModel initialization."""

        class TestModel(BaseModel):
            async def generate(self, prompt: str) -> ModelResponse:
                return ModelResponse.create_success("test")

        model = TestModel(
            model_key="test",
            model_name="test-model",
            display_name="Test Model",
            provider="test",
            max_tokens=100,
            temperature=0.7,
            context_window=1000,
        )

        assert model.model_key == "test"
        assert model.model_name == "test-model"
        assert model.display_name == "Test Model"
        assert model.provider == "test"
        assert model.max_tokens == 100
        assert model.temperature == 0.7
        assert model.context_window == 1000

    def test_base_model_missing_context_window(self) -> None:
        """Test that BaseModel requires context_window."""

        class TestModel(BaseModel):
            async def generate(self, prompt: str) -> ModelResponse:
                return ModelResponse.create_success("test")

        with pytest.raises(ValueError, match="context_window is required"):
            TestModel(
                model_key="test",
                model_name="test-model",
                display_name="Test Model",
                provider="test",
                max_tokens=100,
                temperature=0.7,
                context_window=None,
            )

    def test_full_display_name(self) -> None:
        """Test full_display_name property."""

        class TestModel(BaseModel):
            async def generate(self, prompt: str) -> ModelResponse:
                return ModelResponse.create_success("test")

        model = TestModel(
            model_key="test",
            model_name="test-model-v1",
            display_name="Test Model",
            provider="test",
            max_tokens=100,
            temperature=0.7,
            context_window=1000,
        )

        assert model.full_display_name == "Test Model (test-model-v1)"


class TestLiteLLMModel:
    """Tests for LiteLLMModel class."""

    def test_litellm_model_initialization(self) -> None:
        """Test LiteLLMModel initialization."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        assert model.model_key == "gpt"
        assert model.model_name == "gpt-4"
        assert model.reasoning is False
        assert model.reasoning_effort is None

    def test_litellm_model_with_reasoning(self) -> None:
        """Test LiteLLMModel with reasoning enabled."""
        model = LiteLLMModel(
            model_key="o1",
            model_name="o1-preview",
            display_name="O1 Preview",
            provider="openai",
            temperature=1.0,
            max_tokens=100,
            context_window=8000,
            reasoning=True,
            reasoning_effort="high",
        )

        assert model.reasoning is True
        assert model.reasoning_effort == "high"

    def test_litellm_model_with_system_prompt(self) -> None:
        """Test LiteLLMModel with system prompt."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
            system_prompt="You are a helpful assistant.",
        )

        assert model.system_prompt == "You are a helpful assistant."

    def test_extract_openai_format(self) -> None:
        """Test extracting content from OpenAI format."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test content"
        logger = MagicMock()

        content = model._try_extract_openai_format(mock_response, logger)

        assert content == "Test content"

    def test_extract_openai_format_no_choices(self) -> None:
        """Test extracting from OpenAI format with no choices."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        mock_response = MagicMock()
        mock_response.choices = []
        logger = MagicMock()

        content = model._try_extract_openai_format(mock_response, logger)

        assert content is None

    def test_extract_dict_format_openai(self) -> None:
        """Test extracting content from dict format (OpenAI style)."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        response_dict = {
            "choices": [{"message": {"content": "Test content from dict"}}]
        }
        logger = MagicMock()

        content = model._try_extract_dict_format(response_dict, logger)

        assert content == "Test content from dict"

    def test_extract_dict_format_gemini(self) -> None:
        """Test extracting content from dict format (Gemini style)."""
        model = LiteLLMModel(
            model_key="gemini",
            model_name="gemini-pro",
            display_name="Gemini Pro",
            provider="google",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        response_dict = {
            "candidates": [
                {"content": {"parts": [{"text": "Gemini response"}]}}
            ]
        }
        logger = MagicMock()

        content = model._try_extract_dict_format(response_dict, logger)

        assert content == "Gemini response"

    def test_extract_common_attrs(self) -> None:
        """Test extracting content from common attributes."""
        model = LiteLLMModel(
            model_key="test",
            model_name="test-model",
            display_name="Test",
            provider="test",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        mock_response = MagicMock()
        mock_response.content = "Content from attribute"

        content = model._try_extract_common_attrs(mock_response)

        assert content == "Content from attribute"

    def test_extract_response_cost_from_hidden_params(self) -> None:
        """Test extracting cost from _hidden_params."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        mock_response = MagicMock()
        mock_response._hidden_params.response_cost = 0.05

        cost = model._extract_response_cost(mock_response)

        assert cost == 0.05

    def test_extract_response_cost_from_response_cost_attr(self) -> None:
        """Test extracting cost from response_cost attribute."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        mock_response = MagicMock()
        del mock_response._hidden_params
        mock_response.response_cost = 0.03

        cost = model._extract_response_cost(mock_response)

        assert cost == 0.03

    def test_extract_response_cost_default(self) -> None:
        """Test extracting cost when none available."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        mock_response = MagicMock()
        del mock_response._hidden_params
        del mock_response.response_cost
        del mock_response.usage

        cost = model._extract_response_cost(mock_response)

        assert cost == 0.0

    def test_handle_prompt_size_validation(self) -> None:
        """Test prompt size validation (currently returns prompt as-is)."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        prompt = "Test prompt"
        result = model._handle_prompt_size_validation(prompt)

        assert result == prompt

    def test_clean_response_content(self) -> None:
        """Test cleaning response content."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        content = "  \n  Test content with whitespace  \n  "
        cleaned = model._clean_response_content(content)

        assert cleaned == "Test content with whitespace"

    @pytest.mark.asyncio
    async def test_generate_empty_prompt(self) -> None:
        """Test generating with empty prompt."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        response = await model.generate("")

        assert response.is_error()
        assert "Empty prompt" in response.error

    @pytest.mark.asyncio
    async def test_classify_exception_rate_limit(self) -> None:
        """Test classifying rate limit exception."""
        import litellm

        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        exc = litellm.exceptions.RateLimitError(
            message="Rate limit exceeded", llm_provider="openai", model="gpt-4"
        )
        error_type, error_message = model._classify_exception(exc)

        assert error_type == "rate_limit"
        assert "Rate limit" in error_message

    @pytest.mark.asyncio
    async def test_classify_exception_timeout(self) -> None:
        """Test classifying timeout exception."""
        model = LiteLLMModel(
            model_key="gpt",
            model_name="gpt-4",
            display_name="GPT-4",
            provider="openai",
            temperature=0.7,
            max_tokens=100,
            context_window=8000,
        )

        exc = asyncio.TimeoutError("Request timed out")
        error_type, error_message = model._classify_exception(exc)

        assert error_type == "timeout"
        assert "timed out" in error_message

    def test_from_config_valid(self) -> None:
        """Test creating model from valid config."""
        config = {
            "model_name": "gpt-4",
            "provider": "openai",
            "temperature": 0.7,
            "max_tokens": 1000,
            "context_window": 8000,
        }

        model = LiteLLMModel.from_config("gpt", config)

        assert model.model_key == "gpt"
        assert model.model_name == "gpt-4"
        assert model.provider == "openai"

    def test_from_config_missing_required_field(self) -> None:
        """Test creating model with missing required field."""
        config = {
            "model_name": "gpt-4",
            # missing provider
            "temperature": 0.7,
        }

        with pytest.raises(ValueError, match="Required field"):
            LiteLLMModel.from_config("gpt", config)


class TestRunWithRetry:
    """Tests for run_with_retry function."""

    @pytest.mark.asyncio
    async def test_run_with_retry_success_first_attempt(self) -> None:
        """Test successful response on first attempt."""
        mock_model = MagicMock()
        mock_model.provider = "openai"
        mock_model.generate = AsyncMock(
            return_value=ModelResponse.create_success("Success")
        )

        response = await run_with_retry(
            mock_model, "test prompt", max_attempts=3
        )

        assert response.is_successful
        assert response.content == "Success"
        mock_model.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_retry_success_after_retries(self) -> None:
        """Test successful response after retries."""
        mock_model = MagicMock()
        mock_model.provider = "openai"

        # First call fails, second succeeds
        mock_model.generate = AsyncMock(
            side_effect=[
                ModelResponse.create_error(
                    "Rate limit", error_type="rate_limit"
                ),
                ModelResponse.create_success("Success"),
            ]
        )

        response = await run_with_retry(
            mock_model,
            "test prompt",
            max_attempts=3,
            initial_delay=0.1,
            total_timeout=10,
        )

        assert response.is_successful
        assert mock_model.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_run_with_retry_max_attempts_exceeded(self) -> None:
        """Test failure after max attempts."""
        mock_model = MagicMock()
        mock_model.provider = "openai"
        mock_model.generate = AsyncMock(
            return_value=ModelResponse.create_error(
                "Rate limit", error_type="rate_limit"
            )
        )

        response = await run_with_retry(
            mock_model,
            "test prompt",
            max_attempts=2,
            initial_delay=0.1,
            total_timeout=10,
        )

        assert response.is_error()
        assert mock_model.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_run_with_retry_non_retryable_error(self) -> None:
        """Test stopping retry on non-retryable error."""
        mock_model = MagicMock()
        mock_model.provider = "openai"
        mock_model.generate = AsyncMock(
            return_value=ModelResponse.create_error(
                "Auth failed", error_type="authentication"
            )
        )

        response = await run_with_retry(
            mock_model,
            "test prompt",
            max_attempts=5,
            initial_delay=0.1,
            total_timeout=10,
        )

        assert response.is_error()
        # Should only attempt once for non-retryable errors
        assert mock_model.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_run_with_retry_exception(self) -> None:
        """Test handling exception during generation."""
        mock_model = MagicMock()
        mock_model.provider = "openai"
        mock_model.generate = AsyncMock(side_effect=Exception("Network error"))

        response = await run_with_retry(
            mock_model,
            "test prompt",
            max_attempts=2,
            initial_delay=0.1,
            total_timeout=10,
        )

        assert response.is_error()
