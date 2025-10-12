"""Integration test fixtures and utilities."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from arbitrium import Arbitrium
from arbitrium.models.base import BaseModel, ModelResponse


class MockModel(BaseModel):
    """Mock model for integration testing."""

    def __init__(
        self,
        model_name: str = "test-model",
        display_name: str = "Test Model",
        provider: str = "mock",
        response_text: str = (
            "This is a comprehensive mock response with sufficient detail "
            "for knowledge bank validation"
        ),
        should_fail: bool = False,
        delay: float = 0.0,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        context_window: int = 4000,
    ):
        """Initialize mock model."""
        # Initialize parent BaseModel with required parameters
        super().__init__(
            model_key=model_name,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
        )

        # Mock-specific attributes
        self._response_text = response_text
        self._should_fail = should_fail
        self._delay = delay
        self._call_count = 0

    async def generate(self, prompt: str) -> ModelResponse:
        """Generate mock response."""
        import asyncio
        import re

        self._call_count += 1

        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._should_fail:
            return ModelResponse.create_error(
                "Mock failure", provider=self.provider
            )

        # Vary responses based on call count for more realistic testing
        if "evaluate" in prompt.lower() or "score" in prompt.lower():
            # If response_text was explicitly set to something different
            # from default, and it looks like apology/refusal,
            # use it instead of auto-generating scores
            default_response = (
                "This is a comprehensive mock response with sufficient "
                "detail for knowledge bank validation"
            )
            if self._response_text != default_response and any(
                keyword in self._response_text.lower()
                for keyword in [
                    "sorry",
                    "cannot",
                    "can't",
                    "unable",
                    "apologize",
                ]
            ):
                # Use the custom response text (likely an apology/refusal)
                response = self._response_text
            else:
                # Extract model names from the prompt to provide adaptive scores
                # Look for patterns like "LLM1:", "Model A:", etc.
                model_names = re.findall(
                    r"(LLM\d+|Model [A-Z]|Response \d+)", prompt
                )
                unique_models = list(
                    dict.fromkeys(model_names)
                )  # Preserve order, remove duplicates

                if unique_models:
                    # Generate scores for the actual models in the prompt
                    scores = []
                    for i, model in enumerate(unique_models):
                        # Vary scores to ensure there's a winner and loser
                        score = (
                            8 - i
                        )  # First model gets 8, second gets 7, etc.
                        scores.append(f"{model}: {score}/10")
                    response = "\n".join(scores)
                else:
                    # Fallback to generic response if no models detected
                    response = "Model A: 8/10\nModel B: 7/10"
        elif "improve" in prompt.lower() or "refine" in prompt.lower():
            response = (
                f"Improved: {self._response_text} (call {self._call_count})"
            )
        elif "feedback" in prompt.lower():
            response = "Feedback: This answer could be improved by adding more detail."
        elif "extract" in prompt.lower() or "insight" in prompt.lower():
            # Provide longer insights to pass 50-character minimum
            response = (
                "- The primary consideration here is the long-term "
                "sustainability of the approach\n"
                "- Historical precedent suggests this strategy has "
                "proven effective in similar contexts\n"
                "- Cost-benefit analysis indicates significant "
                "potential for optimization"
            )
        else:
            # Only add call tracking suffix if response_text is non-empty
            if self._response_text:
                response = f"{self._response_text} (call {self._call_count})"
            else:
                response = self._response_text

        return ModelResponse(
            content=response,
            cost=0.001,
            provider=self.provider,
        )


def _get_base_model_configs() -> dict[str, Any]:
    """Get base model configurations shared across fixtures."""
    return {
        "model_a": {
            "provider": "mock",
            "model_name": "test-a",
            "display_name": "Model A",
            "context_window": 4000,
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "model_b": {
            "provider": "mock",
            "model_name": "test-b",
            "display_name": "Model B",
            "context_window": 4000,
            "max_tokens": 1000,
            "temperature": 0.7,
        },
    }


def _get_base_config() -> dict[str, Any]:
    """Get base configuration shared across fixtures."""
    return {
        "retry": {
            "max_attempts": 2,
            "initial_delay": 0.1,
            "max_delay": 1,
        },
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
            "judge_model": None,
            "llm_compression": False,
        },
        "knowledge_bank": {
            "enabled": False,
        },
        "prompts": {
            "initial": (
                "Please answer the following question clearly and "
                "concisely."
            ),
            "feedback": "Provide constructive feedback on the answer.",
            "improvement": (
                "Improve your answer based on the context provided."
            ),
            "evaluate": "Evaluate the responses and provide scores.",
        },
    }


@pytest.fixture()
def basic_config(tmp_dir: Path) -> dict[str, Any]:
    """Basic configuration for integration tests."""
    config = _get_base_config()
    models = _get_base_model_configs()

    # Add model_c for basic config
    models["model_c"] = {
        "provider": "mock",
        "model_name": "test-c",
        "display_name": "Model C",
        "context_window": 4000,
        "max_tokens": 1000,
        "temperature": 0.7,
    }

    config["models"] = models
    config["features"]["save_reports_to_disk"] = True
    config["features"]["knowledge_bank_model"] = "leader"
    config["knowledge_bank"]["similarity_threshold"] = 0.75
    config["knowledge_bank"]["max_insights"] = 100
    config["improvement_phase"] = {
        "enabled": True,
        "feedback_enabled": False,
        "share_responses": True,
    }
    config["refinement_phase"] = {
        "enabled": True,
        "feedback_enabled": False,
        "share_responses": True,
    }
    config["outputs_dir"] = str(tmp_dir)

    return config


@pytest.fixture()
def kb_enabled_config(basic_config: dict[str, Any]) -> dict[str, Any]:
    """Configuration with knowledge bank enabled."""
    config = basic_config.copy()
    config["knowledge_bank"]["enabled"] = True
    return config


@pytest.fixture()
def minimal_config(tmp_dir: Path) -> dict[str, Any]:
    """Minimal configuration (2 models, phases disabled)."""
    config = _get_base_config()

    # Use only 2 models for minimal config
    config["models"] = _get_base_model_configs()

    # Override prompts with minimal versions
    config["prompts"] = {
        "initial": "Answer the question.",
        "evaluate": "Score the responses.",
    }

    # Disable phases
    config["improvement_phase"] = {"enabled": False}
    config["refinement_phase"] = {"enabled": False}
    config["outputs_dir"] = str(tmp_dir)

    return config


@pytest.fixture()
def mock_models() -> dict[str, MockModel]:
    """Create mock models for testing."""
    return {
        "model_a": MockModel(
            model_name="test-a",
            display_name="Model A",
            response_text="Model A's detailed answer",
        ),
        "model_b": MockModel(
            model_name="test-b",
            display_name="Model B",
            response_text="Model B's comprehensive response",
        ),
        "model_c": MockModel(
            model_name="test-c",
            display_name="Model C",
            response_text="Model C's thorough analysis",
        ),
    }


@pytest.fixture()
def failing_model() -> MockModel:
    """Create a mock model that always fails."""
    return MockModel(
        model_name="failing-model",
        display_name="Failing Model",
        should_fail=True,
    )


@pytest_asyncio.fixture()
async def arbitrium_instance(
    basic_config: dict[str, Any],
    mock_models: dict[str, MockModel],
) -> AsyncGenerator[Arbitrium, None]:
    """Create an Arbitrium instance with mock models.

    This fixture provides a pre-configured Arbitrium instance for integration tests,
    eliminating the need to manually set up the instance in each test.

    Usage:
        @pytest.mark.asyncio
        async def test_something(arbitrium_instance: Arbitrium) -> None:
            result, metrics = await arbitrium_instance.run_tournament("question")
            assert result is not None

    For tests requiring custom configuration or mock models with specific behavior,
    you can either:
    1. Create a custom fixture based on this one
    2. Manually set up the instance (only if necessary)

    This fixture uses:
    - basic_config: Standard configuration with 3 models and all phases enabled
    - mock_models: Three mock models (model_a, model_b, model_c) with default behavior
    """
    # Skip health check to avoid LiteLLM calls
    arbitrium = await Arbitrium.from_settings(
        settings=basic_config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # Replace all models with mocks (both all_models and healthy_models)
    arbitrium._all_models = mock_models  # type: ignore[assignment]
    arbitrium._healthy_models = mock_models  # type: ignore[assignment]

    yield arbitrium


@pytest.fixture()
def complex_question() -> str:
    """Complex test question requiring analysis."""
    return """
    Should a startup with 50 employees migrate from monolithic architecture
    to microservices? Consider technical feasibility, team capacity, and
    business impact.
    """.strip()


@pytest.fixture()
def test_questions() -> list[str]:
    """Multiple test questions for batch testing."""
    return [
        "What is 2+2?",
        "Explain quantum computing in simple terms.",
        "What are the benefits of remote work?",
    ]
