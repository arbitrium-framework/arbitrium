"""Prompt management for Arbitrium Framework.

This package contains all prompt-related functionality:
- Formatter: Dynamic delimiter generation and section wrapping
- Builder: Constructs prompts for different tournament phases
- Templates: Template constants for prompts
"""

from .builder import PromptBuilder
from .formatter import DelimiterStyle, PromptFormatter
from .templates import (
    EVALUATION_PROMPT_TEMPLATE,
    FEEDBACK_PROMPT_TEMPLATE,
    IMPROVEMENT_PROMPT_TEMPLATE,
    INITIAL_PROMPT_TEMPLATE,
    LOG_EVALUATOR_RESPONSE,
    LOG_LEVEL_CRITICAL,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    TEXT_COMPRESSION_INSTRUCTION,
)

__all__ = [
    # Templates
    "EVALUATION_PROMPT_TEMPLATE",
    "FEEDBACK_PROMPT_TEMPLATE",
    "IMPROVEMENT_PROMPT_TEMPLATE",
    "INITIAL_PROMPT_TEMPLATE",
    "LOG_EVALUATOR_RESPONSE",
    # Log levels
    "LOG_LEVEL_CRITICAL",
    "LOG_LEVEL_DEBUG",
    "LOG_LEVEL_ERROR",
    "LOG_LEVEL_INFO",
    "LOG_LEVEL_WARNING",
    "TEXT_COMPRESSION_INSTRUCTION",
    # Types
    "DelimiterStyle",
    # Main classes
    "PromptBuilder",
    "PromptFormatter",
]
