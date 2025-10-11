"""Helper functions for tournament execution."""

import re
from typing import Any


def indent_text(text: str, indent: str = "    ") -> str:
    """
    Add indentation to each line of text for better log readability.

    Args:
        text: Text to indent
        indent: Indentation string (default: 4 spaces)

    Returns:
        Indented text with empty lines removed
    """
    lines = [indent + line for line in text.splitlines() if line.strip()]
    return "\n" + "\n".join(lines)


def strip_meta_commentary(text: str, logger: Any | None = None) -> str:
    """
    Remove meta-commentary from model responses.

    Models often add unwanted prefixes like:
    - "Sure, here is an improved answer:"
    - "Hello! I am here to help you..."
    - "Here's my improved response:"

    This function detects and strips these patterns while preserving actual content.

    Args:
        text: The text to clean
        logger: Optional logger for debug messages

    Returns:
        Cleaned text with meta-commentary removed
    """
    if not text or not text.strip():
        return text

    original_text = text
    lines = text.split("\n")

    # Patterns that indicate meta-commentary (case-insensitive)
    meta_patterns = [
        # Common prefixes - including standalone acknowledgments
        r"^\s*(?:sure|okay|alright|certainly|absolutely)(?:[,!\s]+|$)",
        r"^\s*(?:here\s+is|here\'s)\s+",
        r"^\s*(?:improved|refined|enhanced|better)\s+(?:answer|response|version)",
        r"^\s*(?:let me|i will|i\'ll)\s+(?:provide|give|present)",
        r"^\s*(?:my\s+)?(?:improved|refined|enhanced)\s+(?:answer|response)",
        # Greetings
        r"^\s*(?:hello|hi|hey|greetings)!?\s*,?\s*",
        r"^\s*(?:i am|i\'m)\s+(?:here to )?help",
        # Meta phrases at the start
        r"^\s*(?:as requested|as you asked)",
        r"^\s*(?:below is|following is)",
    ]

    # Compile patterns
    compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in meta_patterns]

    # Track removed lines
    removed_lines = []

    # Process lines from the beginning, removing meta-commentary
    clean_lines = []
    started_content = False

    for line in lines:
        stripped_line = line.strip()

        # Skip empty lines at the beginning
        if not started_content and not stripped_line:
            continue

        # Check if this line is meta-commentary
        is_meta = False
        if not started_content and stripped_line:
            for pattern in compiled_patterns:
                if pattern.match(stripped_line):
                    is_meta = True
                    removed_lines.append(stripped_line)
                    break

        # If not meta-commentary, it's actual content
        if not is_meta:
            started_content = True
            clean_lines.append(line)

    # Join the clean lines
    cleaned_text = "\n".join(clean_lines).strip()

    # Log if we removed meta-commentary
    if removed_lines and logger:
        removed_preview = removed_lines[:3]
        extra = f" (and {len(removed_lines) - 3} more)" if len(removed_lines) > 3 else ""
        logger.debug(f"Stripped meta-commentary. Removed: {removed_preview}{extra}")

    # Return original if we accidentally removed everything
    if not cleaned_text and original_text.strip():
        if logger:
            logger.warning("Meta-commentary filter removed all content, returning original")
        return original_text

    return cleaned_text
