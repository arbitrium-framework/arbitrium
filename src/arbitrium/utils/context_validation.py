"""Context window validation utilities for Arbitrium Framework.

This module provides functions to validate prompt sizes against model context windows
before making API calls, preventing context overflow errors.
"""

from typing import Any

import litellm

from .constants import (
    DEFAULT_CHARS_PER_TOKEN,
    DEFAULT_CONTEXT_SAFETY_MARGIN,
    DEFAULT_PRESERVE_END_CHARS,
    DEFAULT_PRESERVE_START_CHARS,
)


def _process_inline_token(token: Any, plain_parts: list[str]) -> None:
    """Process an inline token and add to plain parts."""
    if token.children:
        for child in token.children:
            if child.type == "text":
                plain_parts.append(child.content)
            elif child.type == "code_inline":
                plain_parts.append(child.content)
            elif child.type in ["link_open", "link_close"]:
                continue
    else:
        plain_parts.append(token.content)


def _process_code_block(token: Any, plain_parts: list[str]) -> None:
    """Process a code block token and add to plain parts."""
    plain_parts.append("\n")
    plain_parts.append(token.content)
    plain_parts.append("\n")


def _process_structural_token(token: Any, plain_parts: list[str]) -> None:
    """Process structural tokens (headings, paragraphs, lists, breaks)."""
    if token.type in ["heading_open", "paragraph_open", "bullet_list_open", "ordered_list_open"]:
        plain_parts.append("\n")
    elif token.type in ["heading_close", "paragraph_close", "bullet_list_close", "ordered_list_close"]:
        plain_parts.append("\n")
    elif token.type in ["softbreak", "hardbreak"]:
        plain_parts.append("\n")


def _extract_text_from_tokens(tokens: Any, plain_parts: list[str]) -> None:
    """Recursively extract text from markdown tokens."""
    for token in tokens:
        if token.type == "inline":
            _process_inline_token(token, plain_parts)
        elif token.type in ["code_block", "fence"]:
            _process_code_block(token, plain_parts)
        else:
            _process_structural_token(token, plain_parts)

        # Recursively process children
        if token.children:
            _extract_text_from_tokens(token.children, plain_parts)


def markdown_to_plain_text(text: str) -> str:
    """Convert markdown to plain text to reduce token count.

    Removes all markdown formatting (bold, italic, headers, links, etc.)
    while preserving the actual content, including formulas and code.

    Args:
        text: The markdown text to convert

    Returns:
        Plain text version with formatting removed
    """
    import logging
    import re

    from markdown_it import MarkdownIt

    # Disable markdown_it debug logging
    logging.getLogger("markdown_it").setLevel(logging.WARNING)

    # Parse markdown and render to tokens
    md = MarkdownIt()
    tokens = md.parse(text)

    # Extract plain text from tokens
    plain_parts: list[str] = []
    _extract_text_from_tokens(tokens, plain_parts)
    result = "".join(plain_parts)

    # Clean up excessive whitespace
    result = re.sub(r"\n\n+", "\n\n", result)  # Max 2 newlines
    result = re.sub(r" +", " ", result)  # Collapse spaces

    return result.strip()


def estimate_token_count(text: str, model_name: str) -> int:
    """Estimate the number of tokens in text for a given model.

    Args:
        text: The text to count tokens for
        model_name: The model name for tokenizer selection

    Returns:
        Estimated token count

    Raises:
        ValueError: If tokenization fails
    """
    count: int = litellm.token_counter(model=model_name, text=text)
    return count


def validate_prompt_size(
    prompt: str,
    model_name: str,
    context_window: int,
    max_tokens: int = 4000,
    safety_margin: float = DEFAULT_CONTEXT_SAFETY_MARGIN,
) -> tuple[bool, int, str]:
    """Validate that a prompt will fit within the model's context window.

    Args:
        prompt: The prompt text to validate
        model_name: The model name for tokenizer selection
        context_window: The model's context window size
        max_tokens: Maximum tokens reserved for response
        safety_margin: Safety margin as fraction of context window (0.1 = 10%)

    Returns:
        Tuple of (is_valid, token_count, message)
    """
    token_count = estimate_token_count(prompt, model_name)

    # Calculate available space
    safety_tokens = int(context_window * safety_margin)
    available_tokens = context_window - max_tokens - safety_tokens

    if token_count <= available_tokens:
        return True, token_count, f"Prompt fits within context window ({token_count}/{available_tokens} tokens)"
    else:
        excess_tokens = token_count - available_tokens
        return (
            False,
            token_count,
            (f"Prompt exceeds context window by {excess_tokens} tokens ({token_count}/{available_tokens} available)"),
        )


def _try_markdown_removal(prompt: str, model_name: str, target_tokens: int, current_tokens: int, logger: Any) -> tuple[str, int]:
    """Try to reduce tokens by removing markdown formatting."""
    try:
        cleaned = markdown_to_plain_text(prompt)
        cleaned_tokens = estimate_token_count(cleaned, model_name)

        if logger:
            logger.info(f"Markdown removal: {current_tokens} → {cleaned_tokens} tokens ({int(cleaned_tokens / current_tokens * 100)}%)")

        return cleaned, cleaned_tokens
    except Exception as e:
        if logger:
            logger.warning(f"Markdown removal failed: {e}, skipping to LLM compression")
        return prompt, current_tokens


def _compress_with_llm_once(current_text: str, current_tokens: int, compression_model: str, model_name: str, logger: Any) -> tuple[str, int]:
    """Perform a single LLM compression iteration."""
    from arbitrium.core.prompt_templates import TEXT_COMPRESSION_INSTRUCTION

    compression_instruction = TEXT_COMPRESSION_INSTRUCTION.format(text=current_text)

    response = litellm.completion(
        model=compression_model,
        messages=[{"role": "user", "content": compression_instruction}],
        temperature=0.1,
        max_tokens=min(int(current_tokens * 0.9), 4096),
    )

    compressed = response.choices[0].message.content.strip()
    compressed_tokens = estimate_token_count(compressed, model_name)

    if logger:
        logger.info(f"Compression: {current_tokens} → {compressed_tokens} tokens ({int(compressed_tokens / current_tokens * 100)}%)")

    return compressed, compressed_tokens


def _iterative_llm_compression(cleaned: str, cleaned_tokens: int, target_tokens: int, model_name: str, compression_model: str, logger: Any) -> str:
    """Perform iterative LLM compression until target is met."""
    current_text = cleaned
    current_tokens = cleaned_tokens
    iteration = 0
    max_iterations = 5

    while current_tokens > target_tokens and iteration < max_iterations:
        iteration += 1

        try:
            current_text, current_tokens = _compress_with_llm_once(current_text, current_tokens, compression_model, model_name, logger)

            # Check if we made progress
            if current_tokens >= cleaned_tokens * 0.95:
                if logger:
                    logger.warning(f"Iteration {iteration} made minimal progress, falling back to truncation")
                return truncate_prompt_intelligently(cleaned, model_name, target_tokens)

        except Exception as e:
            if logger:
                logger.warning(f"Compression iteration {iteration} failed: {e}, falling back to truncation")
            return truncate_prompt_intelligently(cleaned, model_name, target_tokens)

    # Check final result
    if current_tokens <= target_tokens:
        if logger:
            logger.info(f"LLM compression complete: {cleaned_tokens} → {current_tokens} tokens in {iteration} iterations")
        return current_text
    else:
        if logger:
            logger.warning(f"Max iterations reached ({iteration}), falling back to truncation")
        return truncate_prompt_intelligently(cleaned, model_name, target_tokens)


def compress_prompt_with_llm(
    prompt: str,
    model_name: str,
    target_tokens: int,
    compression_model: str = "ollama/qwen:1.8b",
    logger: Any | None = None,
) -> str:
    """Compress a prompt using punctuation removal and iterative LLM compression.

    Uses a multi-stage approach:
    1. First removes extra punctuation (quick win)
    2. Then iteratively compresses by 20% using LLM until target is met
    3. Falls back to truncation if needed

    Args:
        prompt: The prompt to compress
        model_name: Target model name for token counting
        target_tokens: Target token count after compression
        compression_model: LLM model to use for compression (should be fast and cheap)
        logger: Optional logger instance

    Returns:
        Compressed prompt
    """
    current_tokens = estimate_token_count(prompt, model_name)

    if current_tokens <= target_tokens:
        return prompt

    # Stage 1: Remove markdown formatting (quick token reduction)
    cleaned, cleaned_tokens = _try_markdown_removal(prompt, model_name, target_tokens, current_tokens, logger)

    if cleaned_tokens <= target_tokens:
        return cleaned

    # Stage 2: Iterative 20% compression until target met
    return _iterative_llm_compression(cleaned, cleaned_tokens, target_tokens, model_name, compression_model, logger)


def truncate_prompt_intelligently(
    prompt: str,
    model_name: str,
    target_tokens: int,
    preserve_start: int = DEFAULT_PRESERVE_START_CHARS,
    preserve_end: int = DEFAULT_PRESERVE_END_CHARS,
) -> str:
    """Intelligently truncate a prompt to fit within token limits.

    Preserves the beginning and end of the prompt, truncating the middle.
    This maintains context about the task while keeping recent information.

    Args:
        prompt: The prompt to truncate
        model_name: Model name for tokenizer selection
        target_tokens: Target token count after truncation
        preserve_start: Characters to preserve from start
        preserve_end: Characters to preserve from end

    Returns:
        Truncated prompt
    """
    # Handle edge case of empty prompt
    if not prompt or not prompt.strip():
        return prompt

    current_tokens = estimate_token_count(prompt, model_name)

    if current_tokens <= target_tokens:
        return prompt

    # If the prompt is very short, just truncate from the end
    if len(prompt) < preserve_start + preserve_end:
        # Character-based truncation with token estimation
        chars_per_token = len(prompt) / current_tokens if current_tokens > 0 else DEFAULT_CHARS_PER_TOKEN
        target_chars = int(target_tokens * chars_per_token)
        return prompt[:target_chars] + "...[truncated]"

    # Preserve start and end, truncate middle
    start_part = prompt[:preserve_start]
    end_part = prompt[-preserve_end:]

    # Estimate tokens for preserved parts
    start_tokens = estimate_token_count(start_part, model_name)
    end_tokens = estimate_token_count(end_part, model_name)
    truncation_marker_tokens = estimate_token_count("\n\n...[content truncated for length]...\n\n", model_name)

    available_tokens = target_tokens - start_tokens - end_tokens - truncation_marker_tokens

    if available_tokens > 0:
        # Calculate how much of the middle we can keep
        middle_part = prompt[preserve_start:-preserve_end]
        middle_tokens = estimate_token_count(middle_part, model_name)

        if middle_tokens <= available_tokens:
            return prompt  # Actually fits

        # Truncate middle part
        chars_per_token = len(middle_part) / middle_tokens if middle_tokens > 0 else DEFAULT_CHARS_PER_TOKEN
        target_middle_chars = int(available_tokens * chars_per_token)
        truncated_middle = middle_part[:target_middle_chars]

        return start_part + "\n\n...[content truncated for length]...\n\n" + truncated_middle + end_part
    else:
        # Even start + end is too much, just use start
        truncation_marker = "\n\n...[heavily truncated]...\n\n"
        available_for_start = target_tokens - estimate_token_count(truncation_marker, model_name)

        if available_for_start > 0:
            chars_per_token = len(start_part) / start_tokens if start_tokens > 0 else DEFAULT_CHARS_PER_TOKEN
            target_start_chars = int(available_for_start * chars_per_token)
            return prompt[:target_start_chars] + truncation_marker
        else:
            return "...[prompt too long to process]..."
