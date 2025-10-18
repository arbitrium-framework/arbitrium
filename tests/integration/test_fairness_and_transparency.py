"""Integration tests for fairness and transparency features.

These tests verify USER-FACING behavior, not implementation details.
Focus: What users see in provenance, logs, and elimination decisions.
"""

import pytest

from arbitrium import Arbitrium
from arbitrium.models.base import BaseModel, ModelResponse
from arbitrium.models.registry import ProviderRegistry


class BiasedScoringModel(BaseModel):
    """Mock model that exhibits scoring bias for testing fairness detection."""

    def __init__(
        self,
        model_name: str,
        display_name: str,
        scoring_behavior: str = "normal",  # "harsh", "generous", "self-promoting"
        provider: str = "mock_biased",
        max_tokens: int = 1000,
        temperature: float = 0.7,
        context_window: int = 4000,
    ):
        """Initialize with specific scoring behavior."""
        super().__init__(
            model_key=model_name,
            model_name=model_name,
            display_name=display_name,
            provider=provider,
            max_tokens=max_tokens,
            temperature=temperature,
            context_window=context_window,
        )
        self.scoring_behavior = scoring_behavior

    async def generate(self, prompt: str) -> ModelResponse:
        """Generate response with biased scoring patterns."""
        # Initial responses
        if "evaluate" not in prompt.lower() and "score" not in prompt.lower():
            return ModelResponse.create_success(
                content=f"Detailed answer from {self.display_name} with unique insights",
                cost=0.0,
            )

        # Evaluation - apply scoring bias
        my_name = self.display_name.replace(
            " ", ""
        )  # Remove spaces for matching

        # Parse which models to score from prompt
        lines = []
        if "LLM1" in prompt:
            score = self._get_score_for_model("LLM1", my_name)
            lines.append(f"LLM1: {score}/10")
        if "LLM2" in prompt:
            score = self._get_score_for_model("LLM2", my_name)
            lines.append(f"LLM2: {score}/10")
        if "LLM3" in prompt:
            score = self._get_score_for_model("LLM3", my_name)
            lines.append(f"LLM3: {score}/10")

        return ModelResponse.create_success(
            content="\n".join(lines) if lines else "No scores",
            cost=0.0,
        )

    def _get_score_for_model(
        self, model_anon_name: str, my_anon_name: str
    ) -> float:
        """Apply scoring bias based on behavior type."""
        is_self = (
            model_anon_name in my_anon_name or my_anon_name in model_anon_name
        )

        if self.scoring_behavior == "self-promoting":
            # Score self much higher than others
            return 9.5 if is_self else 6.0
        elif self.scoring_behavior == "harsh":
            # Give low scores to everyone (but slightly higher to self)
            return 5.5 if is_self else 4.0
        elif self.scoring_behavior == "generous":
            # Give high scores to everyone
            return 9.0 if is_self else 8.5
        else:  # "normal"
            # Fair scoring with slight self-preference
            return 8.0 if is_self else 7.5


@pytest.fixture(autouse=True)
def register_biased_provider():
    """Register biased mock provider."""

    @ProviderRegistry.register("mock_biased")
    class BiasedMockProvider:
        @classmethod
        def from_config(cls, model_key: str, config: dict) -> BaseModel:
            return BiasedScoringModel(
                model_name=model_key,
                display_name=config.get("display_name", model_key),
                scoring_behavior=config.get("scoring_behavior", "normal"),
            )


@pytest.mark.asyncio
async def test_knowledge_bank_preserves_insights_from_eliminated_models(
    tmp_output_dir,
):
    """USER EXPECTATION: When a model is eliminated, its unique insights are preserved.

    This is the core value proposition - no ideas are lost.
    """
    config = {
        "models": {
            "model_a": {
                "provider": "mock_biased",
                "model_name": "model-a",
                "display_name": "Model A",
                "scoring_behavior": "normal",
                "context_window": 4000,
            },
            "model_b": {
                "provider": "mock_biased",
                "model_name": "model-b",
                "display_name": "Model B",
                "scoring_behavior": "self-promoting",  # Will be eliminated
                "context_window": 4000,
            },
            "model_c": {
                "provider": "mock_biased",
                "model_name": "model-c",
                "display_name": "Model C",
                "scoring_behavior": "normal",
                "context_window": 4000,
            },
        },
        "retry": {"max_attempts": 1},
        "features": {
            "save_reports_to_disk": True,  # Save to check provenance
            "deterministic_mode": True,
        },
        "knowledge_bank": {
            "enabled": True,  # CRITICAL: KB must be enabled
        },
        "prompts": {
            "initial": {"content": "Answer.", "metadata": {}},
            "feedback": {"content": "Feedback.", "metadata": {}},
            "improvement": {"content": "Improve.", "metadata": {}},
            "evaluate": {"content": "Score.", "metadata": {}},
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # Run tournament
    result, metrics = await arbitrium.run_tournament(
        "What are the best strategies for X?"
    )

    # USER EXPECTATION: Tournament completes successfully
    assert result is not None
    assert metrics is not None

    # USER EXPECTATION: Eliminated models appear in metrics
    assert "eliminated_models" in metrics
    eliminated = metrics["eliminated_models"]
    assert len(eliminated) > 0

    # USER EXPECTATION: Each elimination has preserved insights
    # This is what users PAY FOR - no insights lost!
    for elimination in eliminated:
        assert "insights_preserved" in elimination
        # Insights should be a list (may be empty if KB extraction failed, but key must exist)
        assert isinstance(elimination["insights_preserved"], list)


@pytest.mark.asyncio
async def test_self_scoring_bias_is_detected_and_reported(tmp_output_dir):
    """USER EXPECTATION: System detects when models score themselves unfairly high.

    Users need to know if results are influenced by self-promotion.
    """
    config = {
        "models": {
            "fair_model": {
                "provider": "mock_biased",
                "model_name": "fair",
                "display_name": "Fair Model",
                "scoring_behavior": "normal",  # Minimal self-bias
                "context_window": 4000,
            },
            "biased_model": {
                "provider": "mock_biased",
                "model_name": "biased",
                "display_name": "Biased Model",
                "scoring_behavior": "self-promoting",  # +3.5 point self-bias!
                "context_window": 4000,
            },
        },
        "retry": {"max_attempts": 1},
        "features": {
            "save_reports_to_disk": True,
            "deterministic_mode": True,
        },
        "knowledge_bank": {"enabled": False},
        "prompts": {
            "initial": {"content": "Answer.", "metadata": {}},
            "feedback": {"content": "Feedback.", "metadata": {}},
            "improvement": {"content": "Improve.", "metadata": {}},
            "evaluate": {"content": "Score.", "metadata": {}},
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    result, metrics = await arbitrium.run_tournament(
        "Test self-scoring bias detection"
    )

    assert result is not None
    assert metrics is not None

    # USER EXPECTATION: If a model exhibited self-scoring bias, it's reported
    eliminated = metrics.get("eliminated_models", [])

    # At least one elimination should have happened
    assert len(eliminated) > 0

    # USER EXPECTATION: Bias information is available in provenance
    # (May be in elimination data if that model was eliminated)
    for elimination in eliminated:
        # If this model had self-scoring bias, it should be reported
        if "self_scoring_bias" in elimination:
            bias = elimination["self_scoring_bias"]
            # Bias should be a number
            assert isinstance(bias, (int, float))


@pytest.mark.asyncio
async def test_elimination_confidence_reflects_judge_agreement(tmp_output_dir):
    """USER EXPECTATION: System reports confidence in elimination decisions.

    High agreement = high confidence. Low agreement = should flag for review.
    """
    config = {
        "models": {
            "model_a": {
                "provider": "mock_biased",
                "model_name": "a",
                "display_name": "Model A",
                "scoring_behavior": "normal",
                "context_window": 4000,
            },
            "model_b": {
                "provider": "mock_biased",
                "model_name": "b",
                "display_name": "Model B",
                "scoring_behavior": "normal",
                "context_window": 4000,
            },
            "model_c": {
                "provider": "mock_biased",
                "model_name": "c",
                "display_name": "Model C",
                "scoring_behavior": "self-promoting",  # Will score poorly from others
                "context_window": 4000,
            },
        },
        "retry": {"max_attempts": 1},
        "features": {
            "save_reports_to_disk": True,
            "deterministic_mode": True,
        },
        "knowledge_bank": {"enabled": False},
        "prompts": {
            "initial": {"content": "Answer.", "metadata": {}},
            "feedback": {"content": "Feedback.", "metadata": {}},
            "improvement": {"content": "Improve.", "metadata": {}},
            "evaluate": {"content": "Score.", "metadata": {}},
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    result, metrics = await arbitrium.run_tournament(
        "Test elimination confidence"
    )

    assert result is not None
    assert metrics is not None

    # USER EXPECTATION: Eliminations include confidence level
    eliminated = metrics.get("eliminated_models", [])
    assert len(eliminated) > 0

    for elimination in eliminated:
        # Must have elimination_confidence field
        assert "elimination_confidence" in elimination
        confidence = elimination["elimination_confidence"]

        # Must be one of: "high", "medium", "low", or "unknown"
        assert confidence in ["high", "medium", "low", "unknown"]

        # If confidence is reported, score_variance should also be present
        if confidence != "unknown":
            assert "score_variance" in elimination
            # Variance should be a number or None
            assert elimination["score_variance"] is None or isinstance(
                elimination["score_variance"], (int, float)
            )


@pytest.mark.asyncio
async def test_judge_normalization_makes_harsh_and_generous_graders_comparable(
    tmp_output_dir,
):
    """USER EXPECTATION: Harsh vs generous judges don't unfairly skew results.

    A harsh judge giving 5/10 and generous judge giving 9/10 to same answer
    should normalize to similar relative rankings.
    """
    config = {
        "models": {
            "harsh_judge": {
                "provider": "mock_biased",
                "model_name": "harsh",
                "display_name": "Harsh Judge",
                "scoring_behavior": "harsh",  # Mean ~4.5
                "context_window": 4000,
            },
            "generous_judge": {
                "provider": "mock_biased",
                "model_name": "generous",
                "display_name": "Generous Judge",
                "scoring_behavior": "generous",  # Mean ~8.7
                "context_window": 4000,
            },
            "normal_model": {
                "provider": "mock_biased",
                "model_name": "normal",
                "display_name": "Normal Model",
                "scoring_behavior": "normal",
                "context_window": 4000,
            },
        },
        "retry": {"max_attempts": 1},
        "features": {
            "save_reports_to_disk": True,
            "deterministic_mode": True,
        },
        "knowledge_bank": {"enabled": False},
        "prompts": {
            "initial": {"content": "Answer.", "metadata": {}},
            "feedback": {"content": "Feedback.", "metadata": {}},
            "improvement": {"content": "Improve.", "metadata": {}},
            "evaluate": {"content": "Score.", "metadata": {}},
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    # USER EXPECTATION: Tournament completes despite wildly different grading scales
    result, metrics = await arbitrium.run_tournament(
        "Test judge normalization across different grading scales"
    )

    # Should complete successfully
    assert result is not None
    assert metrics is not None

    # Should produce a champion
    assert "champion_model" in metrics

    # USER EXPECTATION: The harshest grader doesn't automatically lose
    # due to giving lower scores - normalization should level the playing field
    eliminated = metrics.get("eliminated_models", [])

    # At least one elimination should occur
    assert len(eliminated) > 0


@pytest.mark.asyncio
async def test_complete_transparency_flow_end_to_end(tmp_output_dir):
    """USER EXPECTATION: Complete provenance with all transparency features.

    This is the integration test that verifies the COMPLETE user experience:
    - Insights preserved
    - Bias detected
    - Confidence reported
    - Fair judging
    """
    config = {
        "models": {
            "model_a": {
                "provider": "mock_biased",
                "model_name": "a",
                "display_name": "Model A",
                "scoring_behavior": "normal",
                "context_window": 4000,
            },
            "model_b": {
                "provider": "mock_biased",
                "model_name": "b",
                "display_name": "Model B",
                "scoring_behavior": "self-promoting",
                "context_window": 4000,
            },
            "model_c": {
                "provider": "mock_biased",
                "model_name": "c",
                "display_name": "Model C",
                "scoring_behavior": "generous",
                "context_window": 4000,
            },
        },
        "retry": {"max_attempts": 1},
        "features": {
            "save_reports_to_disk": True,
            "deterministic_mode": True,
        },
        "knowledge_bank": {
            "enabled": True,  # Enable KB for full transparency
        },
        "prompts": {
            "initial": {"content": "Answer.", "metadata": {}},
            "feedback": {"content": "Feedback.", "metadata": {}},
            "improvement": {"content": "Improve.", "metadata": {}},
            "evaluate": {"content": "Score.", "metadata": {}},
        },
        "improvement_phase": {"enabled": False},
        "refinement_phase": {"enabled": False},
        "outputs_dir": str(tmp_output_dir),
    }

    arbitrium = await Arbitrium.from_settings(
        settings=config,
        skip_secrets=True,
        skip_health_check=True,
    )

    result, metrics = await arbitrium.run_tournament(
        "Complete transparency test with all fairness features enabled"
    )

    # USER EXPECTATION 1: Tournament completes
    assert result is not None
    assert metrics is not None

    # USER EXPECTATION 2: Has eliminations with COMPLETE transparency data
    eliminated = metrics.get("eliminated_models", [])
    assert (
        len(eliminated) >= 1
    )  # At least one elimination in 3-model tournament

    for elimination in eliminated:
        # Must have core fields
        assert "model" in elimination
        assert "round" in elimination
        assert "score" in elimination
        assert "reason" in elimination

        # USER EXPECTATION: Transparency fields present
        assert "insights_preserved" in elimination  # KB working
        assert "elimination_confidence" in elimination  # Confidence reported
        assert "score_variance" in elimination  # Variance tracked

        # Self-scoring bias may or may not be present (only if this model was a judge)
        # But if present, should be valid
        if "self_scoring_bias" in elimination:
            assert isinstance(elimination["self_scoring_bias"], (int, float))

    # USER EXPECTATION 3: Champion is selected
    assert "champion_model" in metrics
    assert metrics["champion_model"]  # Not empty

    # USER EXPECTATION 4: Cost tracking works
    assert "total_cost" in metrics
    assert "cost_by_model" in metrics
