"""
JSON extraction utilities for Arbitrium Framework.

This module provides functions to extract JSON from text responses,
handling various formats including code blocks and raw JSON.
"""

import json
import re


def extract_json_from_text(text: str) -> dict[str, object] | list[object] | None:
    """
    Extract JSON from text, handling both code blocks and raw JSON.

    Args:
        text: Text that may contain JSON

    Returns:
        Parsed JSON object (dict or list), or None if extraction fails
    """
    if not text or not isinstance(text, str):
        return None

    # Try to find JSON code block first
    json_block_match = re.search(r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL)
    if json_block_match:
        json_text = json_block_match.group(1)
    else:
        # Try to find bare JSON (starting with { or [)
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            json_text = stripped
        else:
            # No JSON found
            return None

    # Try to parse the extracted text
    try:
        result: dict[str, object] | list[object] = json.loads(json_text)
        return result
    except json.JSONDecodeError:
        # Check if it's a set-like syntax (e.g., {"item1", "item2"} instead of ["item1", "item2"])
        # Some models (like gemma-2b) incorrectly use set syntax with curly braces for lists
        # This is invalid JSON but we can fix it by converting to proper list syntax
        if json_text.strip().startswith("{") and ":" not in json_text[:50]:
            # Likely a set-like structure without key:value pairs
            # Try to convert to list by replacing { with [ and } with ]
            try:
                list_text = json_text.replace("{", "[", 1)
                # Replace the last } with ]
                list_text = "]".join(list_text.rsplit("}", 1))
                result_list: dict[str, object] | list[object] = json.loads(list_text)
                return result_list
            except json.JSONDecodeError:
                pass
        return None
