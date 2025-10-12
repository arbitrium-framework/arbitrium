#!/usr/bin/env python3
"""
Network-free smoke test for tournament system.

Uses MockModel from integration conftest to verify tournament logic
without external dependencies.
"""

import sys
from pathlib import Path
from typing import Any

import pytest

from arbitrium.core.comparison import ModelComparison
from arbitrium.core.tournament import EventHandler, HostEnvironment

# Add tests directory to path to import from conftest
tests_dir = Path(__file__).parent.parent
sys.path.insert(0, str(tests_dir))

from tests.integration.conftest import MockModel  # noqa: E402


class FakeEventHandler(EventHandler):
    """Minimal event handler for testing."""

    def publish(self, _event_name: str, _data: dict[str, Any]) -> None:
        """Publish events - no-op for testing."""
        pass

    async def on_tournament_start(
        self, _question: str, _model_keys: list[str]
    ) -> None:
        pass

    async def on_round_start(
        self, _round_number: int, _active_models: list[str]
    ) -> None:
        pass

    async def on_model_response(self, _model_key: str, _response: str) -> None:
        pass

    async def on_evaluation_start(self, _evaluator_key: str) -> None:
        pass

    async def on_scores_received(self, _scores: dict[str, float]) -> None:
        pass

    async def on_eliminations(
        self, _eliminated: list[str], _remaining: list[str]
    ) -> None:
        pass

    async def on_champion_declared(
        self, _champion_key: str, _final_response: str
    ) -> None:
        pass

    async def on_error(self, _error: str) -> None:
        pass


class FakeHost(HostEnvironment):
    """Minimal host for testing."""

    def __init__(self) -> None:
        """Initialize fake host with base_dir for compatibility."""
        self.base_dir = "/tmp/arbitrium_test"

    async def read_file(self, path: str) -> str:
        return ""

    async def write_file(self, path: str, content: str) -> None:
        pass

    def get_secret(self, key: str) -> str | None:
        return None


@pytest.mark.asyncio
async def test_smoke_tournament() -> None:
    """Smoke test: verify tournament completes and returns champion."""
    # Create mock models with deterministic responses
    models = {
        "model_a": MockModel(
            model_name="model_a",
            display_name="Model A",
            response_text="Response A: The answer is 42.",
            temperature=0.0,
            max_tokens=100,
            context_window=1000,
        ),
        "model_b": MockModel(
            model_name="model_b",
            display_name="Model B",
            response_text="Response B: The answer is clearly 42.",
            temperature=0.0,
            max_tokens=100,
            context_window=1000,
        ),
    }

    # Minimal config
    config = {
        "retry": {"max_attempts": 1, "initial_delay": 1, "max_delay": 1},
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
            "judge_model": None,
            "knowledge_bank_model": None,
            "llm_compression": False,
        },
        "knowledge_bank": {"enabled": False},
        "prompts": {
            "initial": "Answer the question.",
            "feedback": "Provide feedback.",
            "improvement": "Improve the answer.",
            "evaluate": "Score from 0-10.",
        },
    }

    comparison = ModelComparison(
        config=config,
        models=models,  # type: ignore
        event_handler=FakeEventHandler(),
        host=FakeHost(),
    )

    # Run tournament
    result = await comparison.run("What is the meaning of life?")

    # Verify we got a non-empty result
    assert result, "Tournament should return a non-empty response"
    assert isinstance(result, str), "Result should be a string"
    assert len(result) > 0, "Result should not be empty"


@pytest.mark.asyncio
async def test_single_model_tournament() -> None:
    """Test tournament with only one model."""
    models = {
        "solo": MockModel(
            model_name="solo",
            display_name="Solo Model",
            response_text="I am the only model.",
            temperature=0.0,
            max_tokens=100,
            context_window=1000,
        ),
    }

    config = {
        "retry": {"max_attempts": 1, "initial_delay": 1, "max_delay": 1},
        "features": {
            "save_reports_to_disk": False,
            "deterministic_mode": True,
            "judge_model": None,
            "knowledge_bank_model": None,
            "llm_compression": False,
        },
        "knowledge_bank": {"enabled": False},
        "prompts": {
            "initial": "Answer.",
            "feedback": "Feedback.",
            "improvement": "Improve.",
            "evaluate": "Score.",
        },
    }

    comparison = ModelComparison(
        config=config,
        models=models,  # type: ignore
        event_handler=FakeEventHandler(),
        host=FakeHost(),
    )

    result = await comparison.run("Test question")

    assert result, "Single model tournament should return response"
    assert (
        "solo" in comparison.active_model_keys
        or len(comparison.active_model_keys) == 1
    )
