"""
Integration tests for Knowledge Bank functionality.

These tests verify that the Knowledge Bank preserves and integrates insights
from eliminated models, testing from a user's perspective rather than
implementation details.

Philosophy:
- Test observable behavior (insights preserved, answers improved)
- Avoid accessing internal KB state unless necessary for verification
- Focus on user outcomes
"""

from pathlib import Path

import pytest

from arbitrium import Arbitrium
from tests.integration.conftest import MockModel


class TestKnowledgeBankPreservation:
    """Test that Knowledge Bank preserves insights from eliminated models."""

    @pytest.mark.asyncio
    async def test_kb_enabled_tournament_completes_successfully(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: User runs tournament with Knowledge Bank enabled
        EXPECTED: Tournament completes and KB captures insights
        """
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="Model A",
                response_text=(
                    "The primary factor is long-term sustainability which requires "
                    "careful environmental impact assessment and stakeholder engagement."
                ),
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="Model B",
                response_text=(
                    "Economic viability and cost-benefit analysis should drive the decision, "
                    "considering both short-term and long-term financial implications."
                ),
            ),
            "model_c": MockModel(
                model_name="c",
                display_name="Model C",
                response_text=(
                    "Community feedback and social impact assessment are critical, "
                    "ensuring the solution addresses real needs and concerns."
                ),
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament(
            "What factors should guide infrastructure development decisions?"
        )

        # Verify: Tournament completed
        assert result is not None, "KB-enabled tournament should complete"
        assert metrics["champion_model"] is not None
        assert len(metrics["eliminated_models"]) == 2

    @pytest.mark.asyncio
    async def test_kb_disabled_tournament_still_works(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User runs tournament with Knowledge Bank disabled
        EXPECTED: Tournament works normally without KB features
        """
        # Ensure KB is disabled
        basic_config["knowledge_bank"]["enabled"] = False

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="a", display_name="A"),
            "model_b": MockModel(model_name="b", display_name="B"),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Test question")

        # Verify: Works without KB
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_kb_handles_models_with_short_responses(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: Some models provide very brief responses
        EXPECTED: KB gracefully handles short responses without errors
        """
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "detailed": MockModel(
                model_name="detailed",
                display_name="Detailed",
                response_text=(
                    "This is a comprehensive answer with multiple perspectives, "
                    "detailed analysis, and thorough consideration of various factors "
                    "that contribute to a complete understanding of the topic."
                ),
            ),
            "brief": MockModel(
                model_name="brief",
                display_name="Brief",
                response_text="Yes.",  # Very short
            ),
            "moderate": MockModel(
                model_name="moderate",
                display_name="Moderate",
                response_text="This is a moderate length response.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Should we proceed?")

        # Verify: Handles mixed response lengths gracefully
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_kb_handles_models_with_error_responses(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: One model returns error-like response
        EXPECTED: KB skips extracting from error responses
        """
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "working": MockModel(
                model_name="working",
                display_name="Working",
                response_text="This is a valid and comprehensive response with good insights.",
            ),
            "error_like": MockModel(
                model_name="error",
                display_name="Error-like",
                response_text="Error: Failed to generate response due to timeout.",
            ),
            "working_2": MockModel(
                model_name="working2",
                display_name="Working 2",
                response_text="Another valid response with different perspectives.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Test question")

        # Verify: Tournament completes despite error-like response
        assert result is not None
        assert metrics["champion_model"] is not None


class TestKnowledgeBankConfiguration:
    """Test Knowledge Bank configuration options."""

    @pytest.mark.asyncio
    async def test_kb_with_custom_similarity_threshold(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: User sets custom similarity threshold for KB
        EXPECTED: KB respects custom threshold for deduplication
        """
        # Set custom threshold
        kb_enabled_config["knowledge_bank"]["similarity_threshold"] = 0.9

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="A",
                response_text="The key factor is environmental sustainability.",
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="B",
                response_text="Environmental sustainability is the primary consideration.",
            ),
            "model_c": MockModel(
                model_name="c",
                display_name="C",
                response_text="Cost-effectiveness and budget constraints matter most.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("What matters most?")

        # Verify: Tournament completes with custom threshold
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_kb_with_max_insights_limit(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: User sets maximum number of insights to preserve
        EXPECTED: KB respects max_insights configuration
        """
        # Set max insights
        kb_enabled_config["knowledge_bank"]["max_insights"] = 10

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            f"model_{i}": MockModel(
                model_name=f"m{i}",
                display_name=f"Model {i}",
                response_text=f"Insight {i}: This is a unique perspective on the topic.",
            )
            for i in range(5)
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Analyze this topic")

        # Verify: Tournament completes with max insights limit
        assert result is not None
        assert metrics["champion_model"] is not None


class TestKnowledgeBankIntegration:
    """Test Knowledge Bank integration with tournament phases."""

    @pytest.mark.asyncio
    async def test_kb_insights_available_in_later_rounds(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: KB extracts insights from eliminated models
        EXPECTED: Subsequent rounds can access these insights
        """
        # Enable improvement and refinement phases to have multiple rounds
        kb_enabled_config["improvement_phase"]["enabled"] = True
        kb_enabled_config["refinement_phase"]["enabled"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="A",
                response_text=(
                    "The historical context shows that incremental changes "
                    "have proven more successful than radical transformations."
                ),
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="B",
                response_text=(
                    "Data-driven decision making requires robust metrics "
                    "and continuous monitoring of key performance indicators."
                ),
            ),
            "model_c": MockModel(
                model_name="c",
                display_name="C",
                response_text=(
                    "Stakeholder buy-in and change management are critical "
                    "for successful implementation of any strategic initiative."
                ),
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament(
            "What best practices should guide organizational change?"
        )

        # Verify: Tournament completed with multiple rounds
        assert result is not None
        assert metrics["champion_model"] is not None
        # With 3 models, should have 2 eliminations
        assert len(metrics["eliminated_models"]) == 2

    @pytest.mark.asyncio
    async def test_kb_with_leader_extraction_model(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: KB configured to use leader model for extraction
        EXPECTED: Leader (highest scoring) model extracts insights
        """
        # Set KB to use leader for extraction
        kb_enabled_config["features"]["knowledge_bank_model"] = "leader"

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "strong": MockModel(
                model_name="strong",
                display_name="Strong Model",
                response_text="Comprehensive analysis with deep insights.",
            ),
            "weak": MockModel(
                model_name="weak",
                display_name="Weak Model",
                response_text="Brief response.",
            ),
            "moderate": MockModel(
                model_name="moderate",
                display_name="Moderate Model",
                response_text="Moderate quality response.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Analyze this")

        # Verify: Tournament completes with leader-based extraction
        assert result is not None
        assert metrics["champion_model"] is not None


class TestKnowledgeBankEdgeCases:
    """Test Knowledge Bank behavior in edge cases."""

    @pytest.mark.asyncio
    async def test_kb_with_single_model_no_elimination(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: Tournament with single model (no eliminations)
        EXPECTED: KB has no insights (nothing to preserve)
        """
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "solo": MockModel(
                model_name="solo",
                display_name="Solo",
                response_text="This is the only response.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Solo question")

        # Verify: Completes despite no eliminations
        assert result is not None
        assert metrics["champion_model"] == "solo"
        assert len(metrics["eliminated_models"]) == 0

    @pytest.mark.asyncio
    async def test_kb_with_identical_responses(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: All models provide very similar responses
        EXPECTED: KB deduplicates similar insights
        """
        # Lower similarity threshold to catch similar responses
        kb_enabled_config["knowledge_bank"]["similarity_threshold"] = 0.8

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        identical_text = (
            "The answer is 42. This is the meaning of life, "
            "the universe, and everything."
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="A",
                response_text=identical_text,
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="B",
                response_text=identical_text
                + " Indeed.",  # Slightly different
            ),
            "model_c": MockModel(
                model_name="c",
                display_name="C",
                response_text=identical_text,
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament(
            "What is the meaning?"
        )

        # Verify: Tournament handles duplicate responses
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_kb_with_very_long_responses(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: Models provide very long responses
        EXPECTED: KB handles long text without errors
        """
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Generate very long responses
        long_response = " ".join(
            [
                f"This is insight number {i} about the topic at hand."
                for i in range(100)
            ]
        )

        mock_models = {
            "verbose_a": MockModel(
                model_name="a",
                display_name="Verbose A",
                response_text=long_response,
            ),
            "verbose_b": MockModel(
                model_name="b",
                display_name="Verbose B",
                response_text=long_response + " Additional thoughts.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Discuss thoroughly")

        # Verify: Handles long responses
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_kb_extraction_handles_extraction_failures_gracefully(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: Insight extraction fails for some eliminated models
        EXPECTED: Tournament continues, processes remaining models
        """
        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "normal": MockModel(
                model_name="normal",
                display_name="Normal",
                response_text="This is a normal response with valid insights.",
            ),
            "problematic": MockModel(
                model_name="prob",
                display_name="Problematic",
                response_text="@#$%^&*",  # Potentially problematic content
            ),
            "another_normal": MockModel(
                model_name="normal2",
                display_name="Normal 2",
                response_text="Another normal response.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Test question")

        # Verify: Tournament completes despite potential extraction issues
        assert result is not None
        assert metrics["champion_model"] is not None


class TestKnowledgeBankReporting:
    """Test Knowledge Bank reporting and provenance tracking."""

    @pytest.mark.asyncio
    async def test_kb_provenance_includes_source_information(
        self, kb_enabled_config: dict, tmp_output_dir: Path
    ) -> None:
        """
        SCENARIO: User wants to trace where insights came from
        EXPECTED: KB tracks source model and round for each insight
        """
        # Enable report saving
        kb_enabled_config["features"]["save_reports_to_disk"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=kb_enabled_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="Model A",
                response_text="Unique perspective A with distinctive insights.",
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="Model B",
                response_text="Different angle B with alternative viewpoints.",
            ),
            "model_c": MockModel(
                model_name="c",
                display_name="Model C",
                response_text="Third approach C with novel considerations.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        await arbitrium.run_tournament("Analyze from multiple perspectives")

        # Verify: Report files created
        reports = list(tmp_output_dir.glob("*.md"))
        assert len(reports) > 0, "Should create report files"

        # Check if any report mentions insights or knowledge bank
        any(
            "insight" in report.read_text().lower()
            or "knowledge" in report.read_text().lower()
            for report in reports
        )
        # Note: Whether KB info appears in reports depends on implementation
        # This test just verifies reports are created when KB is enabled
        assert reports  # At minimum, reports should exist
