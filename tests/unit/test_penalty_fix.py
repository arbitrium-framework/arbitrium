#!/usr/bin/env python3
"""Test the new penalty logic - invalid evaluations should be discarded."""

import logging

from arbitrium.core.comparison import ScoreExtractor
from arbitrium.logging import get_contextual_logger, setup_logging

# Setup logger
setup_logging(level=logging.INFO, enable_file_logging=False)
logger = get_contextual_logger("test")

# Create score extractor instance
extractor = ScoreExtractor(logger)

print("=" * 80)
print("TEST 1: Valid evaluation (all models scored)")
print("=" * 80)

evaluation_text_valid = """
LLM1: 9/10
LLM2: 7/10
LLM3: 8/10
"""

scores = extractor.extract_scores_from_evaluation(evaluation_text_valid, ["LLM1", "LLM2", "LLM3"])
print(f"Result: {scores}")
print("Expected: {'LLM1': 9.0, 'LLM2': 7.0, 'LLM3': 8.0}")
assert scores == {"LLM1": 9.0, "LLM2": 7.0, "LLM3": 8.0}, f"FAILED: {scores}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 2: Invalid evaluation (missing one model)")
print("=" * 80)

evaluation_text_invalid = """
LLM1: 9/10
LLM3: 8/10
"""

scores = extractor.extract_scores_from_evaluation(evaluation_text_invalid, ["LLM1", "LLM2", "LLM3"])
print(f"Result: {scores}")
print("Expected: {}")
assert scores == {}, f"FAILED: Expected empty dict, got {scores}"
print("✅ PASSED\n")

print("=" * 80)
print("TEST 3: Invalid evaluation (missing multiple models)")
print("=" * 80)

evaluation_text_invalid2 = """
LLM1: 9/10
"""

scores = extractor.extract_scores_from_evaluation(evaluation_text_invalid2, ["LLM1", "LLM2", "LLM3"])
print(f"Result: {scores}")
print("Expected: {}")
assert scores == {}, f"FAILED: Expected empty dict, got {scores}"
print("✅ PASSED\n")

print("=" * 80)
print("ALL TESTS PASSED! 🎉")
print("=" * 80)
print("\nSummary:")
print("- Valid evaluations return all scores")
print("- Invalid evaluations (missing any model) return empty dict")
print("- This prevents unfair penalties when evaluator fails")
