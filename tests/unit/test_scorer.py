"""
Unit tests for score extraction and normalization.
"""

from unittest.mock import patch

from arbitrium.core.scorer import ScoreExtractor


class TestScoreExtractor:
    """Tests for ScoreExtractor class."""

    def test_extractor_initialization(self) -> None:
        """Test extractor initializes correctly."""
        extractor = ScoreExtractor()
        assert extractor.logger is not None

    def test_extract_numeric_score_from_int(self) -> None:
        """Test extracting numeric score from integer."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score(8)
        assert score == 8.0

    def test_extract_numeric_score_from_float(self) -> None:
        """Test extracting numeric score from float."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score(7.5)
        assert score == 7.5

    def test_extract_numeric_score_from_string(self) -> None:
        """Test extracting numeric score from string."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score("8.5")
        assert score == 8.5

    def test_extract_numeric_score_from_fraction(self) -> None:
        """Test extracting numeric score from fraction format."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score("8/10")
        assert score == 8.0

    def test_extract_numeric_score_from_list(self) -> None:
        """Test extracting numeric score from list."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score([9.0, 8.0])
        assert score == 9.0

    def test_extract_numeric_score_from_empty_list(self) -> None:
        """Test extracting from empty list returns None."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score([])
        assert score is None

    def test_extract_numeric_score_invalid(self) -> None:
        """Test extracting from invalid value returns None."""
        extractor = ScoreExtractor()
        score = extractor._extract_numeric_score("invalid")
        assert score is None

    def test_normalize_score_valid(self) -> None:
        """Test normalizing a valid score."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(8.0, "test_evaluator")
        assert score == 8.0

    def test_normalize_score_too_high(self) -> None:
        """Test normalizing score above 10 but within 10.5."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(10.2, "test_evaluator")
        assert score is not None
        assert score <= 10.0

    def test_normalize_score_way_too_high(self) -> None:
        """Test rejecting score way above valid range."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(15.0, "test_evaluator")
        assert score is None

    def test_normalize_score_too_low(self) -> None:
        """Test rejecting score below valid range."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(0.3, "test_evaluator")
        assert score is None

    def test_normalize_score_fractional_normalizes_up(self) -> None:
        """Test normalizing fractional score (0-1 range) to 1-10."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(0.8, "test_evaluator")
        assert score is not None
        assert score == 8.0

    def test_normalize_score_at_boundary(self) -> None:
        """Test normalizing score at boundary (exactly 10)."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(10.0, "test_evaluator")
        assert score == 10.0

    def test_match_model_name_exact(self) -> None:
        """Test exact model name matching."""
        extractor = ScoreExtractor()
        model_names = ["model_a", "model_b"]
        result = extractor._match_model_name("model_a", model_names)
        assert result == "model_a"

    def test_match_model_name_fuzzy(self) -> None:
        """Test fuzzy model name matching."""
        extractor = ScoreExtractor()
        model_names = ["gpt-4", "claude-3"]
        result = extractor._match_model_name("gpt", model_names)
        assert result == "gpt-4"

    def test_match_model_name_no_match(self) -> None:
        """Test no match returns None."""
        extractor = ScoreExtractor()
        model_names = ["model_a", "model_b"]
        result = extractor._match_model_name("model_c", model_names)
        assert result is None

    def test_extract_scores_using_pattern_matching(self) -> None:
        """Test extracting scores using pattern matching."""
        extractor = ScoreExtractor()
        evaluation_text = """
        model_a: 8/10
        model_b: 9/10
        """
        model_names = ["model_a", "model_b"]

        scores = extractor._extract_scores_using_pattern_matching(
            evaluation_text, model_names
        )

        assert "model_a" in scores
        assert "model_b" in scores
        assert scores["model_a"] == 8.0
        assert scores["model_b"] == 9.0

    def test_extract_scores_with_alternative_names(self) -> None:
        """Test extracting scores using alternative names (LLM1, LLM2)."""
        extractor = ScoreExtractor()
        evaluation_text = """
        LLM1: 7/10
        LLM2: 8/10
        """
        model_names = ["model_a", "model_b"]

        scores = extractor._extract_scores_with_alternative_names(
            evaluation_text, model_names
        )

        assert len(scores) == 2

    def test_extract_scores_with_response_alternative_names(self) -> None:
        """Test extracting scores using Response 1, Response 2 format."""
        extractor = ScoreExtractor()
        evaluation_text = """
        Response 1: 7/10
        Response 2: 8/10
        """
        model_names = ["model_a", "model_b"]

        scores = extractor._extract_scores_with_alternative_names(
            evaluation_text, model_names
        )

        assert len(scores) == 2

    def test_extract_scores_complete(self) -> None:
        """Test complete score extraction with fallback to alternatives."""
        extractor = ScoreExtractor()
        evaluation_text = """
        model_a: 8/10
        LLM2: 9/10
        """
        model_names = ["model_a", "model_b"]

        scores = extractor.extract_scores(evaluation_text, model_names)

        assert "model_a" in scores
        # Should fall back to alternative names for model_b
        assert len(scores) >= 1

    def test_extract_scores_from_evaluation_with_apology(self) -> None:
        """Test that apology responses return empty dict."""
        extractor = ScoreExtractor()
        evaluation_text = "I apologize, but I cannot evaluate these responses."
        model_names = ["model_a", "model_b"]

        with patch(
            "arbitrium.core.scorer.detect_apology_or_refusal",
            return_value=True,
        ):
            scores = extractor.extract_scores_from_evaluation(
                evaluation_text, model_names, "test_evaluator"
            )

        assert scores == {}

    def test_extract_scores_from_evaluation_missing_models(self) -> None:
        """Test that incomplete evaluations return empty dict."""
        extractor = ScoreExtractor()
        evaluation_text = """
        model_a: 8/10
        """
        model_names = ["model_a", "model_b", "model_c"]

        scores = extractor.extract_scores_from_evaluation(
            evaluation_text, model_names, "test_evaluator"
        )

        # Should return empty dict when not all models have scores
        assert scores == {}

    def test_extract_scores_from_evaluation_complete(self) -> None:
        """Test complete evaluation extraction."""
        extractor = ScoreExtractor()
        evaluation_text = """
        model_a: 8/10
        model_b: 9/10
        """
        model_names = ["model_a", "model_b"]

        with patch(
            "arbitrium.core.scorer.detect_apology_or_refusal",
            return_value=False,
        ):
            scores = extractor.extract_scores_from_evaluation(
                evaluation_text, model_names, "test_evaluator"
            )

        assert len(scores) == 2
        assert "model_a" in scores
        assert "model_b" in scores

    def test_try_extract_fractional_score_valid(self) -> None:
        """Test extracting fractional score."""
        import re

        extractor = ScoreExtractor()
        pattern = r"(\d+)/(\d+)"
        match = re.search(pattern, "8/10")

        if match:
            score = extractor._try_extract_fractional_score(match, "model_a")
            assert score == 8.0

    def test_try_extract_fractional_score_zero_denominator(self) -> None:
        """Test extracting fractional score with zero denominator."""
        import re

        extractor = ScoreExtractor()
        pattern = r"(\d+)/(\d+)"
        match = re.search(pattern, "8/0")

        if match:
            score = extractor._try_extract_fractional_score(match, "model_a")
            assert score is None

    def test_extract_simple_score(self) -> None:
        """Test extracting simple score from match."""
        import re

        extractor = ScoreExtractor()
        pattern = r"(\d+\.?\d*)"
        match = re.search(pattern, "8.5")

        if match:
            score = extractor._extract_simple_score(match, "model_a")
            assert score == 8.5

    def test_extract_simple_score_invalid(self) -> None:
        """Test extracting simple score with invalid match."""
        import re

        extractor = ScoreExtractor()
        pattern = r"([a-z]+)"
        match = re.search(pattern, "invalid")

        if match:
            score = extractor._extract_simple_score(match, "model_a")
            assert score is None

    def test_extract_score_for_model_found(self) -> None:
        """Test extracting score for specific model."""
        extractor = ScoreExtractor()
        evaluation_text = "model_a scored 8/10"
        patterns = [r"{model_name}.*?(\d+)/(\d+)"]

        score = extractor._extract_score_for_model(
            evaluation_text, "model_a", patterns
        )

        assert score == 8.0

    def test_extract_score_for_model_not_found(self) -> None:
        """Test extracting score when model not in text."""
        extractor = ScoreExtractor()
        evaluation_text = "model_a scored 8/10"
        patterns = [r"{model_name}.*?(\d+)/(\d+)"]

        score = extractor._extract_score_for_model(
            evaluation_text, "model_b", patterns
        )

        assert score is None

    def test_try_extract_score_from_match_fractional(self) -> None:
        """Test extracting score from match with fractional format."""
        import re

        extractor = ScoreExtractor()
        pattern = r"(\d+)/(\d+)"
        match = re.search(pattern, "7/10")

        if match:
            score = extractor._try_extract_score_from_match(match, "model_a")
            assert score == 7.0

    def test_try_extract_score_from_match_simple(self) -> None:
        """Test extracting score from match with simple format."""
        import re

        extractor = ScoreExtractor()
        pattern = r"(\d+\.?\d*)"
        match = re.search(pattern, "8.5")

        if match:
            score = extractor._try_extract_score_from_match(match, "model_a")
            assert score == 8.5

    def test_extract_scores_no_scores_found(self) -> None:
        """Test extracting scores when none are found."""
        extractor = ScoreExtractor()
        evaluation_text = "These models performed well."
        model_names = ["model_a", "model_b"]

        scores = extractor.extract_scores(evaluation_text, model_names)

        assert len(scores) == 0

    def test_extract_scores_partial_match(self) -> None:
        """Test extracting scores with partial matches."""
        extractor = ScoreExtractor()
        evaluation_text = """
        model_a: 8/10
        Something else here
        """
        model_names = ["model_a", "model_b"]

        scores = extractor.extract_scores(evaluation_text, model_names)

        assert "model_a" in scores
        # model_b might be found via alternative names or not at all

    def test_normalize_score_edge_case_0_5(self) -> None:
        """Test normalizing score at lower boundary."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(0.5, "test_evaluator")
        assert score is not None

    def test_normalize_score_edge_case_10_5(self) -> None:
        """Test normalizing score at upper boundary."""
        extractor = ScoreExtractor()
        score = extractor.normalize_score(10.5, "test_evaluator")
        assert score is not None
