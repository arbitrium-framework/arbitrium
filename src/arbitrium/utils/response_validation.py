"""Shared utilities for validating model responses.

This module contains common validation functions used across the framework
to detect invalid, malformed, or problematic responses from language models.
"""


def detect_apology_or_refusal(response_text: str) -> bool:
    """Detect if response is an apology or refusal instead of proper content.

    Models sometimes refuse to perform tasks or apologize instead of providing
    the requested content. This function detects such responses.

    Args:
        response_text: The text to check

    Returns:
        True if the response appears to be an apology/refusal, False otherwise

    Examples:
        >>> detect_apology_or_refusal("I cannot help with that request")
        True
        >>> detect_apology_or_refusal("Here is the analysis you requested...")
        False
    """
    if not response_text:
        return False

    # Convert to lowercase for case-insensitive matching
    text_lower = response_text.lower().strip()

    # Common refusal/apology patterns
    refusal_patterns = [
        "i cannot",
        "i can't",
        "i'm sorry",
        "i am sorry",
        "i apologize",
        "sorry, i",
        "sorry but",
        "i'm unable",
        "i am unable",
        "i don't have",
        "i do not have",
        "as an ai",
        "i'm an ai",
        "i am an ai",
    ]

    # Check first 200 chars for refusal patterns
    # (refusals typically appear at the beginning)
    text_start = text_lower[:200]
    return any(pattern in text_start for pattern in refusal_patterns)
