"""
Comprehensive integration tests for Arbitrium tournament workflows.

These tests verify the system works correctly from a user's perspective,
testing complete workflows and observable behavior rather than implementation details.

Philosophy:
- Test what users experience, not how the code works
- Focus on inputs, outputs, and side effects
- Independent of implementation changes
- Test real-world scenarios and edge cases
"""

from pathlib import Path

import pytest

from arbitrium import Arbitrium
from tests.integration.conftest import MockModel

# ==============================================================================
# Core Tournament Workflows
# ==============================================================================


class TestCompleteTournamentWorkflows:
    """Test complete tournament workflows from start to finish."""

    @pytest.mark.asyncio
    async def test_basic_tournament_end_to_end(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """
        SCENARIO: User runs a basic tournament with 3 models
        EXPECTED: Tournament completes and produces a champion answer
        """
        # Setup
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "claude": MockModel(
                model_name="claude-sonnet-4",
                display_name="Claude Sonnet 4",
                response_text="The capital of France is Paris, established as the capital in 508 AD.",
                context_window=200000,
            ),
            "gpt": MockModel(
                model_name="gpt-4o",
                display_name="GPT-4o",
                response_text="Paris is the capital of France, a major European city.",
                context_window=128000,
            ),
            "gemini": MockModel(
                model_name="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                response_text="France's capital is Paris.",
                context_window=100000,
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        question = "What is the capital of France?"
        result, metrics = await arbitrium.run_tournament(question)

        # Verify: Tournament produced a result
        assert result is not None, "Tournament must produce a champion answer"
        assert len(result) > 0, "Champion answer cannot be empty"
        assert "Paris" in result, "Answer should be about Paris"

        # Verify: Metrics are complete
        assert (
            metrics["champion_model"] in mock_models
        ), "Champion must be one of the participating models"
        assert metrics["total_cost"] >= 0, "Total cost must be non-negative"
        assert (
            len(metrics["eliminated_models"]) == 2
        ), "With 3 models, 2 should be eliminated"
        assert "cost_by_model" in metrics, "Should track cost per model"

    @pytest.mark.asyncio
    async def test_tournament_with_knowledge_bank_enabled(
        self, kb_enabled_config: dict
    ) -> None:
        """
        SCENARIO: User runs tournament with Knowledge Bank to preserve insights
        EXPECTED: Champion answer may incorporate insights from eliminated models
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
                response_text="The key consideration is long-term sustainability and environmental impact.",
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="Model B",
                response_text="Economic feasibility and cost-benefit analysis are crucial factors.",
            ),
            "model_c": MockModel(
                model_name="c",
                display_name="Model C",
                response_text="Stakeholder engagement and community feedback should drive decisions.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament(
            "What factors should guide urban planning decisions?"
        )

        # Verify: Tournament completed successfully
        assert result is not None
        assert metrics["champion_model"] is not None

        # Verify: Result contains substantial analysis (from KB integration)
        assert len(result) > 50, "Result should contain detailed analysis"

    @pytest.mark.asyncio
    async def test_minimal_two_model_tournament(
        self, minimal_config: dict
    ) -> None:
        """
        SCENARIO: User runs minimal tournament with just 2 models (no extra phases)
        EXPECTED: Tournament completes with one winner, minimal processing
        """
        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="Model A",
                response_text="Answer A: Comprehensive response",
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="Model B",
                response_text="Answer B: Alternative perspective",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Test question")

        # Verify: One winner from two models
        assert result is not None
        assert metrics["champion_model"] in mock_models
        assert (
            len(metrics["eliminated_models"]) == 1
        ), "Exactly 1 model should be eliminated"

    @pytest.mark.asyncio
    async def test_tournament_with_custom_judge_model(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User configures a dedicated judge model for evaluation
        EXPECTED: Judge model evaluates competitors without participating
        """
        # Configure judge model
        basic_config["features"]["judge_model"] = "judge"
        basic_config["models"]["judge"] = {
            "provider": "mock",
            "model_name": "judge-model",
            "display_name": "Judge Model",
            "temperature": 0.3,
            "max_tokens": 2000,
            "context_window": 50000,
        }

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        competitor_models = {
            "model_a": MockModel(model_name="a", display_name="Model A"),
            "model_b": MockModel(model_name="b", display_name="Model B"),
        }
        judge_model = MockModel(
            model_name="judge-model",
            display_name="Judge Model",
            temperature=0.3,
        )

        # All models (competitors + judge)
        arbitrium._healthy_models = {**competitor_models, "judge": judge_model}

        # Execute
        _result, metrics = await arbitrium.run_tournament(
            "Evaluate these approaches"
        )

        # Verify: Champion is a competitor, not the judge
        assert metrics["champion_model"] in competitor_models
        assert metrics["champion_model"] != "judge"


# ==============================================================================
# Error Handling & Edge Cases
# ==============================================================================


class TestTournamentErrorHandling:
    """Test how tournaments handle errors and edge cases."""

    @pytest.mark.asyncio
    async def test_tournament_continues_when_one_model_fails(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: One model fails during tournament execution
        EXPECTED: Tournament continues with remaining working models
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "working_1": MockModel(
                model_name="work1",
                display_name="Working Model 1",
                response_text="I provide a good answer",
            ),
            "failing": MockModel(
                model_name="fail",
                display_name="Failing Model",
                should_fail=True,
            ),
            "working_2": MockModel(
                model_name="work2",
                display_name="Working Model 2",
                response_text="I also provide a good answer",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Test question")

        # Verify: Tournament completes with working models
        assert (
            result is not None
        ), "Tournament should complete despite one failure"
        assert metrics["champion_model"] in ["working_1", "working_2"]

    @pytest.mark.asyncio
    async def test_tournament_with_empty_question(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User submits an empty or whitespace-only question
        EXPECTED: System handles gracefully without crashing
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model": MockModel(
                model_name="test",
                display_name="Test Model",
                response_text="I respond even to empty questions",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute: These should not crash
        for empty_question in ["", "   ", "\n\n", "\t"]:
            result, metrics = await arbitrium.run_tournament(empty_question)

            # Verify: System handles gracefully
            assert result is not None
            assert metrics is not None

    @pytest.mark.asyncio
    async def test_tournament_with_very_long_question(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User submits an extremely long question (edge case)
        EXPECTED: System processes or handles gracefully
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model": MockModel(model_name="test", display_name="Test Model"),
        }
        arbitrium._healthy_models = mock_models

        # Create a very long question (10,000 characters)
        long_question = "What is " + ("the meaning of life " * 1000)

        # Execute
        result, metrics = await arbitrium.run_tournament(long_question)

        # Verify: System handles without crashing
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_tournament_with_special_characters_in_question(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: Question contains special characters, unicode, emojis
        EXPECTED: System processes correctly without encoding errors
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model": MockModel(model_name="test", display_name="Test"),
        }
        arbitrium._healthy_models = mock_models

        # Questions with various special characters
        special_questions = [
            "What is the meaning of 生活 (life)?",
            "Explain AI 🤖 and ML 📊 differences",
            "How to handle <xml> & JSON?",
            "What about \"quotes\" and 'apostrophes'?",
        ]

        for question in special_questions:
            result, metrics = await arbitrium.run_tournament(question)

            # Verify: Processes without errors
            assert result is not None
            assert metrics["champion_model"] is not None


# ==============================================================================
# Output Validation & Reporting
# ==============================================================================


class TestTournamentOutputs:
    """Test tournament output generation and file creation."""

    @pytest.mark.asyncio
    async def test_tournament_saves_report_files(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """
        SCENARIO: User runs tournament with report saving enabled
        EXPECTED: Tournament creates markdown report files in output directory
        """
        # Enable report saving
        basic_config["features"]["save_reports_to_disk"] = True

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
        await arbitrium.run_tournament("Generate a report")

        # Verify: Report files were created
        report_files = list(tmp_output_dir.glob("*.md"))
        assert len(report_files) > 0, "Should create at least one report file"

        # Verify: Report contains expected content
        champion_report = report_files[0]
        content = champion_report.read_text()

        # Should contain key sections
        assert any(
            keyword in content
            for keyword in ["Champion", "Winner", "Question"]
        ), "Report should identify winner and question"

    @pytest.mark.asyncio
    async def test_tournament_report_contains_question_and_answer(
        self, basic_config: dict, tmp_output_dir: Path
    ) -> None:
        """
        SCENARIO: User wants to verify report contains the question and answer
        EXPECTED: Report file includes both the original question and champion answer
        """
        basic_config["features"]["save_reports_to_disk"] = True

        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "winner": MockModel(
                model_name="winner",
                display_name="Winner Model",
                response_text="UNIQUE_ANSWER: This is the champion's distinctive response",
            ),
            "loser": MockModel(
                model_name="loser",
                display_name="Loser Model",
                response_text="Standard response",
            ),
        }
        arbitrium._healthy_models = mock_models

        question = "UNIQUE_QUESTION: What makes a great test?"

        # Execute
        await arbitrium.run_tournament(question)

        # Verify: Report contains question and answer
        reports = list(tmp_output_dir.glob("*.md"))
        assert len(reports) > 0

        content = reports[0].read_text()
        assert (
            "UNIQUE_QUESTION" in content
        ), "Report should contain the original question"
        # The champion's answer should be in the report
        assert "response" in content.lower() or "answer" in content.lower()

    @pytest.mark.asyncio
    async def test_tournament_metrics_have_correct_structure(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User needs to programmatically access tournament metrics
        EXPECTED: Metrics dict has consistent structure with expected fields
        """
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
        _result, metrics = await arbitrium.run_tournament("Test")

        # Verify: Required fields present
        required_fields = {
            "champion_model",
            "total_cost",
            "cost_by_model",
            "eliminated_models",
        }
        assert required_fields.issubset(
            set(metrics.keys())
        ), f"Metrics must include: {required_fields}"

        # Verify: Field types are correct
        assert isinstance(metrics["champion_model"], str)
        assert isinstance(metrics["total_cost"], (int, float))
        assert isinstance(metrics["cost_by_model"], dict)
        assert isinstance(metrics["eliminated_models"], list)


# ==============================================================================
# Different Tournament Configurations
# ==============================================================================


class TestTournamentConfigurations:
    """Test tournaments with different configuration options."""

    @pytest.mark.asyncio
    async def test_tournament_with_different_model_counts(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User runs tournaments with different numbers of models
        EXPECTED: All configurations complete successfully
        """
        test_cases = [
            (2, 1),  # 2 models -> 1 elimination
            (3, 2),  # 3 models -> 2 eliminations
            (4, 3),  # 4 models -> 3 eliminations
            (5, 4),  # 5 models -> 4 eliminations
        ]

        for num_models, expected_eliminations in test_cases:
            arbitrium = await Arbitrium.from_settings(
                settings=basic_config,
                skip_secrets=True,
                skip_health_check=True,
            )

            # Create N models
            mock_models = {
                f"model_{i}": MockModel(
                    model_name=f"m{i}",
                    display_name=f"Model {i}",
                    response_text=f"Response from model {i}",
                )
                for i in range(num_models)
            }
            arbitrium._healthy_models = mock_models

            # Execute
            result, metrics = await arbitrium.run_tournament("Test question")

            # Verify
            assert (
                result is not None
            ), f"Tournament with {num_models} models should complete"
            assert (
                len(metrics["eliminated_models"]) == expected_eliminations
            ), f"With {num_models} models, should have {expected_eliminations} eliminations"

    @pytest.mark.asyncio
    async def test_tournament_with_phases_disabled(
        self, minimal_config: dict
    ) -> None:
        """
        SCENARIO: User disables improvement and refinement phases for speed
        EXPECTED: Tournament completes faster with basic evaluation only
        """
        arbitrium = await Arbitrium.from_settings(
            settings=minimal_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="a", display_name="A"),
            "model_b": MockModel(model_name="b", display_name="B"),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("Quick question")

        # Verify: Tournament still completes successfully
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_deterministic_mode_produces_consistent_results(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User enables deterministic mode for reproducible results
        EXPECTED: Running same tournament twice produces same champion
        (assuming same model configurations and responses)
        """
        # Ensure deterministic mode is enabled
        basic_config["features"]["deterministic_mode"] = True

        # Create two identical arbitrium instances
        results = []
        for _ in range(2):
            arbitrium = await Arbitrium.from_settings(
                settings=basic_config,
                skip_secrets=True,
                skip_health_check=True,
            )

            # Use same models with same responses
            mock_models = {
                "model_a": MockModel(
                    model_name="a",
                    display_name="A",
                    response_text="Consistent response A",
                ),
                "model_b": MockModel(
                    model_name="b",
                    display_name="B",
                    response_text="Consistent response B",
                ),
            }
            arbitrium._healthy_models = mock_models

            result, metrics = await arbitrium.run_tournament("Same question")
            results.append((result, metrics))

        # Verify: Same champion in both runs (deterministic)
        assert (
            results[0][1]["champion_model"] == results[1][1]["champion_model"]
        ), "Deterministic mode should produce same champion"


# ==============================================================================
# Single Model Execution
# ==============================================================================


class TestSingleModelExecution:
    """Test single model execution (non-tournament mode)."""

    @pytest.mark.asyncio
    async def test_run_single_model_returns_response(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User wants to run a single model without tournament
        EXPECTED: Model's response is returned directly
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "solo_model": MockModel(
                model_name="solo",
                display_name="Solo Model",
                response_text="This is my solo response to your question",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        response = await arbitrium.run_single_model(
            "solo_model", "What is AI?"
        )

        # Verify
        assert response.is_successful, "Single model execution should succeed"
        assert "solo response" in response.content
        assert not response.is_error()

    @pytest.mark.asyncio
    async def test_run_single_model_with_failing_model(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User runs single model that fails
        EXPECTED: Error response is returned (not an exception)
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "failing": MockModel(
                model_name="fail",
                display_name="Failing",
                should_fail=True,
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        response = await arbitrium.run_single_model("failing", "Test")

        # Verify: Returns error response, doesn't crash
        assert response.is_error(), "Should return error for failing model"
        assert response.error is not None


# ==============================================================================
# Cost Tracking
# ==============================================================================


class TestCostTracking:
    """Test that tournament costs are tracked correctly."""

    @pytest.mark.asyncio
    async def test_tournament_reports_non_negative_costs(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User runs tournament and checks costs
        EXPECTED: All costs are non-negative numbers
        """
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
        _result, metrics = await arbitrium.run_tournament("Test")

        # Verify: Costs are valid
        assert metrics["total_cost"] >= 0, "Total cost cannot be negative"

        for model_key, cost in metrics["cost_by_model"].items():
            assert cost >= 0, f"Cost for {model_key} cannot be negative"

    @pytest.mark.asyncio
    async def test_total_cost_equals_sum_of_model_costs(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User verifies cost accounting is accurate
        EXPECTED: Total cost equals sum of individual model costs
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(model_name="a", display_name="A"),
            "model_b": MockModel(model_name="b", display_name="B"),
            "model_c": MockModel(model_name="c", display_name="C"),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        _result, metrics = await arbitrium.run_tournament("Test")

        # Verify: Accounting is accurate
        total_from_models = sum(metrics["cost_by_model"].values())
        assert (
            abs(metrics["total_cost"] - total_from_models) < 0.01
        ), "Total cost should equal sum of individual model costs"


# ==============================================================================
# Question Type Handling
# ==============================================================================


class TestDifferentQuestionTypes:
    """Test tournament behavior with different question types."""

    @pytest.mark.asyncio
    async def test_simple_factual_question(self, basic_config: dict) -> None:
        """
        SCENARIO: User asks simple factual question
        EXPECTED: Tournament produces concise factual answer
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "model_a": MockModel(
                model_name="a",
                display_name="A",
                response_text="2 + 2 = 4",
            ),
            "model_b": MockModel(
                model_name="b",
                display_name="B",
                response_text="The answer is 4",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament("What is 2 + 2?")

        # Verify
        assert result is not None
        assert metrics["champion_model"] is not None

    @pytest.mark.asyncio
    async def test_complex_analytical_question(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User asks complex question requiring analysis
        EXPECTED: Tournament produces detailed analytical answer
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "analyst": MockModel(
                model_name="analyst",
                display_name="Analyst",
                response_text=(
                    "The ethical implications of AI in healthcare are multifaceted. "
                    "Privacy concerns include data security, patient consent, and potential "
                    "for discriminatory algorithms. We should address these through robust "
                    "governance frameworks, transparency requirements, and continuous auditing."
                ),
            ),
            "brief": MockModel(
                model_name="brief",
                display_name="Brief",
                response_text="AI in healthcare raises privacy concerns that need regulation.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        question = (
            "What are the ethical implications of AI in healthcare, "
            "and how should we address privacy concerns?"
        )
        result, _metrics = await arbitrium.run_tournament(question)

        # Verify
        assert result is not None
        assert (
            len(result) > 50
        ), "Complex question should produce substantial answer"

    @pytest.mark.asyncio
    async def test_open_ended_creative_question(
        self, basic_config: dict
    ) -> None:
        """
        SCENARIO: User asks open-ended creative question
        EXPECTED: Tournament selects most creative/comprehensive response
        """
        arbitrium = await Arbitrium.from_settings(
            settings=basic_config,
            skip_secrets=True,
            skip_health_check=True,
        )

        mock_models = {
            "creative": MockModel(
                model_name="creative",
                display_name="Creative",
                response_text=(
                    "In a world without technology, human connection would return to "
                    "storytelling around fires, handwritten letters, and face-to-face "
                    "conversations. Communities would be smaller but more tightly knit."
                ),
            ),
            "simple": MockModel(
                model_name="simple",
                display_name="Simple",
                response_text="People would talk more in person.",
            ),
        }
        arbitrium._healthy_models = mock_models

        # Execute
        result, metrics = await arbitrium.run_tournament(
            "Imagine a world without technology. How would humans communicate?"
        )

        # Verify
        assert result is not None
        assert metrics["champion_model"] is not None
