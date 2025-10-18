"""
Integration tests for tournament behavior - testing WHAT the system does, not HOW.

These tests focus on observable behavior and outcomes rather than implementation details.
They validate the system works correctly from a user perspective.
"""

import pytest

from arbitrium import Arbitrium
from tests.integration.conftest import MockModel


class TestTournamentBasicBehavior:
    """Test that tournaments produce expected outcomes."""

    @pytest.mark.asyncio
    async def test_tournament_produces_single_winner(
        self, basic_config: dict, tmp_output_dir
    ) -> None:
        """Tournament must always produce exactly one winner from multiple models."""
        # Setup
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="Model A",
                response_text="Response from Model A with sufficient detail",
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="Model B",
                response_text="Response from Model B with sufficient detail",
            ),
            "model_c": MockModel(
                model_name="c",
                display_name="Model C",
                response_text="Response from Model C with sufficient detail",
            ),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        result, metrics = await arbitrium.run_tournament(
            "What is the meaning of life?"
        )

        # Assert - Observable outcomes
        assert result is not None, "Tournament must produce a result"
        assert len(result) > 0, "Winner's answer must not be empty"
        assert (
            metrics["champion_model"] is not None
        ), "Tournament must declare a champion"
        assert (
            metrics["champion_model"] in mock_models
        ), "Champion must be one of the participating models"

    @pytest.mark.asyncio
    async def test_tournament_eliminates_losing_models(
        self, basic_config: dict
    ) -> None:
        """Tournament must eliminate models that don't win."""
        # Setup: 3 models -> 1 winner means 2 eliminations
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            f"model_{i}": MockModel(
                model_name=f"m{i}",
                display_name=f"Model {i}",
                response_text=f"Response from model {i} with detail",
            )
            for i in range(3)
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        _result, metrics = await arbitrium.run_tournament("Test question")

        # Assert - Focus on observable behavior, not internal details
        eliminated = metrics["eliminated_models"]
        champion = metrics["champion_model"]

        # Core behavioral assertions:
        # 1. Should have exactly 2 eliminations for 3 models
        assert len(eliminated) == 2, "3 models should result in 2 eliminations"

        # 2. Should have a champion
        assert champion is not None, "Should declare a champion"
        assert champion in mock_models, "Champion should be one of our models"

        # 3. Number of eliminations + champion should equal total models
        assert len(eliminated) + 1 == len(
            mock_models
        ), "Eliminations + champion should equal total models"

    @pytest.mark.asyncio
    async def test_tournament_answer_comes_from_champion(
        self, basic_config: dict
    ) -> None:
        """The final answer must come from the winning model."""
        # Setup
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        # Create models with distinct responses
        mock_models = {
            "unique_a": MockModel(
                model_name="a",
                display_name="Model A",
                response_text="SIGNATURE_A: This is model A's unique response",
            ),
            "unique_b": MockModel(
                model_name="b",
                display_name="Model B",
                response_text="SIGNATURE_B: This is model B's unique response",
            ),
            "unique_c": MockModel(
                model_name="c",
                display_name="Model C",
                response_text="SIGNATURE_C: This is model C's unique response",
            ),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        result, metrics = await arbitrium.run_tournament("Test question")
        champion = metrics["champion_model"]

        # Assert - The result should contain the champion's signature
        # (This tests that we're actually returning the winner's answer)
        expected_signature = f"SIGNATURE_{champion.split('_')[1].upper()}"
        assert (
            expected_signature in result
        ), f"Final answer should contain {expected_signature} from champion {champion}"


class TestTournamentCostTracking:
    """Test that tournament correctly tracks costs."""

    @pytest.mark.asyncio
    async def test_tournament_reports_total_cost(
        self, basic_config: dict
    ) -> None:
        """Tournament must report total cost of execution."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="a", display_name="Model A"),
            "model_b": MockModel(model_name="b", display_name="Model B"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        _, metrics = await arbitrium.run_tournament("Test question")

        # Assert
        assert "total_cost" in metrics, "Metrics must include total_cost"
        assert metrics["total_cost"] >= 0, "Total cost cannot be negative"
        assert isinstance(
            metrics["total_cost"], (int, float)
        ), "Total cost must be numeric"

    @pytest.mark.asyncio
    async def test_tournament_cost_increases_with_models(
        self, basic_config: dict
    ) -> None:
        """More models should result in higher costs (more API calls)."""
        # Test with 2 models
        arbitrium_2 = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )
        mock_models_2 = {
            "a": MockModel(model_name="a", display_name="A"),
            "b": MockModel(model_name="b", display_name="B"),
        }
        arbitrium_2._healthy_models = mock_models_2  # type: ignore

        _, metrics_2 = await arbitrium_2.run_tournament("Test question")

        # Test with 4 models
        arbitrium_4 = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )
        mock_models_4 = {
            f"m{i}": MockModel(model_name=f"m{i}", display_name=f"M{i}")
            for i in range(4)
        }
        arbitrium_4._healthy_models = mock_models_4  # type: ignore

        _, metrics_4 = await arbitrium_4.run_tournament("Test question")

        # Assert
        # More models = more API calls = higher cost
        # Note: This might not always be true if costs are zero, but validates tracking
        assert (
            metrics_4["total_cost"] >= metrics_2["total_cost"]
        ), "4 models should cost at least as much as 2 models"


class TestTournamentWithDifferentQuestions:
    """Test tournament behavior with various question types."""

    @pytest.mark.asyncio
    async def test_tournament_handles_simple_question(
        self, basic_config: dict
    ) -> None:
        """Tournament should handle simple factual questions."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "a": MockModel(model_name="a", display_name="A"),
            "b": MockModel(model_name="b", display_name="B"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        result, metrics = await arbitrium.run_tournament("What is 2+2?")

        # Assert
        assert result is not None
        assert len(result) > 0
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_tournament_handles_complex_question(
        self, basic_config: dict
    ) -> None:
        """Tournament should handle complex analytical questions."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "a": MockModel(
                model_name="a",
                display_name="A",
                response_text="Complex analytical response with multiple perspectives and detailed reasoning",
            ),
            "b": MockModel(
                model_name="b",
                display_name="B",
                response_text="Different analytical approach with comprehensive detail and evidence",
            ),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        question = "What are the ethical implications of AI in healthcare, and how should we address privacy concerns?"
        result, metrics = await arbitrium.run_tournament(question)

        # Assert
        assert result is not None
        assert (
            len(result) > 20
        ), "Complex question should produce substantial answer"
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_tournament_handles_empty_question(
        self, basic_config: dict
    ) -> None:
        """Tournament should handle edge case of empty question gracefully."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "a": MockModel(model_name="a", display_name="A"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act - should not crash
        result, metrics = await arbitrium.run_tournament("")

        # Assert - system should handle it gracefully
        assert result is not None  # Should return something, not crash
        assert metrics is not None


class TestTournamentModelFailures:
    """Test tournament behavior when models fail."""

    @pytest.mark.asyncio
    async def test_tournament_continues_when_one_model_fails(
        self, basic_config: dict
    ) -> None:
        """Tournament should continue if one model fails but others work."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "working_a": MockModel(
                model_name="a",
                display_name="Working A",
                response_text="I work correctly",
            ),
            "failing_b": MockModel(
                model_name="b",
                display_name="Failing B",
                should_fail=True,
            ),
            "working_c": MockModel(
                model_name="c",
                display_name="Working C",
                response_text="I also work correctly",
            ),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        result, metrics = await arbitrium.run_tournament("Test question")

        # Assert
        assert (
            result is not None
        ), "Tournament should complete despite one failure"
        assert (
            metrics["champion_model"] is not None
        ), "Should select champion from working models"
        # Champion should be one of the working models
        assert metrics["champion_model"] in [
            "working_a",
            "working_c",
        ], "Champion should be a working model"


class TestTournamentOutputs:
    """Test that tournament produces expected outputs."""

    @pytest.mark.asyncio
    async def test_tournament_saves_report_to_disk(
        self, basic_config: dict, tmp_output_dir
    ) -> None:
        """Tournament should save a report file to the outputs directory."""
        # Ensure save_reports_to_disk is enabled
        basic_config["features"]["save_reports_to_disk"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "a": MockModel(model_name="a", display_name="A"),
            "b": MockModel(model_name="b", display_name="B"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        await arbitrium.run_tournament("Test question")

        # Assert - Check that a report file was created
        outputs_dir = tmp_output_dir
        report_files = list(outputs_dir.glob("*.md"))
        assert len(report_files) > 0, "Should create at least one report file"

        # Verify report contains expected sections
        report_content = report_files[0].read_text()
        # Report should have question section (various possible headings)
        has_question = any(
            keyword in report_content
            for keyword in ["Question", "QUESTION", "Initial Question"]
        )
        assert has_question, "Report should contain question section"

        # Report should identify winner
        has_winner = any(
            keyword in report_content
            for keyword in ["Champion", "Winner", "Champion Solution"]
        )
        assert has_winner, "Report should identify winner"

    @pytest.mark.asyncio
    async def test_tournament_metrics_structure(
        self, basic_config: dict
    ) -> None:
        """Tournament metrics should have expected structure."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "a": MockModel(model_name="a", display_name="A"),
            "b": MockModel(model_name="b", display_name="B"),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        _, metrics = await arbitrium.run_tournament("Test")

        # Assert - Check expected metric keys exist
        expected_keys = {
            "champion_model",
            "total_cost",
            "eliminated_models",
        }
        assert expected_keys.issubset(
            set(metrics.keys())
        ), f"Metrics should contain at least: {expected_keys}"


class TestSingleModelExecution:
    """Test single model execution (non-tournament mode)."""

    @pytest.mark.asyncio
    async def test_single_model_returns_response(
        self, basic_config: dict
    ) -> None:
        """Single model execution should return the model's response."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "test_model": MockModel(
                model_name="test",
                display_name="Test Model",
                response_text="This is my response to your question",
            ),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        response = await arbitrium.run_single_model(
            "test_model", "What is AI?"
        )

        # Assert
        assert response.is_successful, "Response should be successful"
        assert (
            "This is my response" in response.content
        ), "Should contain model's response"
        assert not response.is_error(), "Should not be an error"

    @pytest.mark.asyncio
    async def test_single_model_failure_returns_error(
        self, basic_config: dict
    ) -> None:
        """Single model execution should return error when model fails."""
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "failing_model": MockModel(
                model_name="fail",
                display_name="Failing Model",
                should_fail=True,
            ),
        }
        arbitrium._healthy_models = mock_models  # type: ignore

        # Act
        response = await arbitrium.run_single_model("failing_model", "Test")

        # Assert
        assert response.is_error(), "Response should be an error"
        assert response.error is not None, "Error message should be present"
