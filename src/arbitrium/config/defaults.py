"""Default configuration values for Arbitrium Framework.

All defaults are defined as Python dictionaries to ensure they are properly
included in the package distribution. This avoids path resolution issues
with YAML files when running from repository vs installed package.
"""

from typing import Any


def select_model_with_highest_context(models: dict[str, Any]) -> str | None:
    """Select the model with the highest context window from available models.

    Works with both model configs (dict[str, dict]) and model instances (dict[str, BaseModel]).

    Args:
        models: Dictionary of model configurations or BaseModel instances

    Returns:
        Model key with highest context window, or None if no models available
    """
    if not models:
        return None

    best_model_key = None
    best_context = -1

    for model_key, model_obj in models.items():
        # Handle both dict configs and BaseModel instances
        if isinstance(model_obj, dict):
            context_window = model_obj.get("context_window", 0)
        elif hasattr(model_obj, "context_window"):
            context_window = getattr(model_obj, "context_window", 0)
        else:
            context_window = 0

        if context_window and context_window > best_context:
            best_context = context_window
            best_model_key = model_key

    return best_model_key


# Default model definitions
MODELS: dict[str, dict[str, Any]] = {
    # ============================================================================
    # Small models (2-4B parameters, 2-4GB RAM)
    # ============================================================================
    "gemma-2b": {  # 2B parameters, ~2GB RAM
        "provider": "ollama",
        "model_name": "ollama/gemma:2b",
        "display_name": "Gemma 2B",
        "temperature": 0.7,
        "max_tokens": 4096,  # Override auto-detected 2048 (25% of 8192)
    },
    "phi": {  # 2.7B parameters, ~3GB RAM
        "provider": "ollama",
        "model_name": "ollama/phi",
        "display_name": "Phi-2 2.7B",
        "temperature": 0.7,
        "max_tokens": 1024,  # Override auto-detected 512 (25% of 2048)
    },
    "orca-mini": {  # 3B parameters, ~3GB RAM
        "provider": "ollama",
        "model_name": "ollama/orca-mini:3b",
        "display_name": "Orca Mini 3B",
        "temperature": 0.7,
        "max_tokens": 1024,  # Override auto-detected 512 (25% of 2048)
    },
    # ============================================================================
    # Medium-small models (3.8-4B parameters, 4GB RAM)
    # ============================================================================
    "phi3": {  # 3.8B parameters, ~4GB RAM, context 4096
        "provider": "ollama",
        "model_name": "ollama/phi3",
        "display_name": "Phi-3 Mini 3.8B",
        "temperature": 0.7,
    },
    "phi4-mini": {  # 3.8B parameters, ~4GB RAM
        "provider": "ollama",
        "model_name": "ollama/phi4-mini",
        "display_name": "Phi-4 3.8b",
        "temperature": 0.7,
    },
    "gemma3-4b": {  # 4B parameters, ~4GB RAM
        "provider": "ollama",
        "model_name": "ollama/gemma3:4b",
        "display_name": "Gemma 3 4B",
        "temperature": 0.7,
    },
    "qwen3-4b": {  # 4B parameters, ~4GB RAM, context 8192
        "provider": "ollama",
        "model_name": "ollama/qwen3:4b",
        "display_name": "Qwen 3 4B",
        "temperature": 0.7,
    },
    # ============================================================================
    # Standard-small models (7-12B parameters, 8-12GB RAM)
    # ============================================================================
    "mistral-1": {  # 7B parameters, ~8GB RAM
        "provider": "ollama",
        "model_name": "ollama/mistral:7b-instruct-v0.2-q4_K_M",
        "display_name": "Mistral 7B",
        "temperature": 0.7,
    },
    "llama3": {  # 8B parameters, ~8GB RAM
        "provider": "ollama",
        "model_name": "ollama/llama3:8b-instruct-q4_K_M",
        "display_name": "Llama 3 8B",
        "temperature": 0.7,
    },
    "qwen3-8b": {  # 8B parameters, ~8GB RAM
        "provider": "ollama",
        "model_name": "ollama/qwen3:8b",
        "display_name": "Qwen 3 8B",
        "temperature": 0.7,
    },
    "gemma3-12b": {  # 12B parameters, ~12GB RAM
        "provider": "ollama",
        "model_name": "ollama/gemma3:12b",
        "display_name": "Gemma 3 12B",
        "temperature": 0.7,
    },
    # ============================================================================
    # Medium models (14-30B parameters, 16-32GB RAM)
    # ============================================================================
    "phi4": {  # 14B parameters, ~16GB RAM
        "provider": "ollama",
        "model_name": "ollama/phi4",
        "display_name": "Phi-4 14b",
        "temperature": 0.7,
    },
    "gpt-oss-20b": {  # 20B parameters, ~24GB RAM
        "provider": "ollama",
        "model_name": "ollama/gpt-oss:20b-cloud",
        "display_name": "GPT-OSS 20B",
        "temperature": 0.7,
    },
    "gemma3-27b": {  # 27B parameters, ~32GB RAM
        "provider": "ollama",
        "model_name": "ollama/gemma3:27b",
        "display_name": "Gemma 3 27B",
        "temperature": 0.7,
    },
    "qwen3-30b": {  # 30B parameters, ~32GB RAM
        "provider": "ollama",
        "model_name": "ollama/qwen3:30b",
        "display_name": "Qwen 3 30B",
        "temperature": 0.7,
    },
    # ============================================================================
    # Huge/Semi-clever models (120B-671B parameters, >128GB RAM)
    # ============================================================================
    "gpt-oss-120b": {  # 120B parameters, ~128GB RAM
        "provider": "ollama",
        "model_name": "ollama/gpt-oss:120b-cloud",
        "display_name": "GPT-OSS 120B",
        "temperature": 0.7,
    },
    "deepseek-v3": {  # 671B parameters, >512GB RAM
        "provider": "ollama",
        "model_name": "ollama/deepseek-v3.1:671b-cloud",
        "display_name": "DeepSeek V3 671B",
        "temperature": 0.7,
        "context_window": 128000,
        "max_tokens": 8192,
    },
    "grok": {  # Parameters not public but constantly scored lower than top tier
        "provider": "xai",
        "model_name": "xai/grok-4-latest",
        "display_name": "Grok 4",
        "temperature": 0.7,
    },
    # ============================================================================
    # Top tier models (Parameters not public - SOTA commercial models)
    # ============================================================================
    "gpt": {  # Parameters not public
        "provider": "openai",
        "model_name": "gpt-5",
        "display_name": "GPT-5",
        "temperature": 1.0,  # GPT-5 only supports temperature=1.0
        "reasoning_effort": "high",
        "max_tokens": 32000,  # Must be > thinking.budget_tokens for extended thinking
    },
    "claude": {  # Parameters not public
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-5-20250929",
        "display_name": "Claude 4.5 Sonnet",
        "temperature": 1.0,
        "reasoning_effort": "high",
        "max_tokens": 32000,  # Must be > thinking.budget_tokens for extended thinking
    },
    "gemini": {  # Parameters not public
        "provider": "vertex_ai",
        "model_name": "vertex_ai/gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "temperature": 0.7,
        "reasoning_effort": "high",
        "max_tokens": 32000,  # Must be > thinking.budget_tokens for extended thinking
    },
}

# Default retry settings
RETRY: dict[str, Any] = {
    "max_attempts": 3,
    "initial_delay": 10,
    "max_delay": 60,
}

# Default feature flags
FEATURES: dict[str, Any] = {
    "save_reports_to_disk": True,
    "deterministic_mode": True,
    "judge_model": None,  # None = use peer evaluation, or specify model key for single judge
    "knowledge_bank_model": "leader",  # "leader" = use current leader, or specify model key
    "llm_compression": False,  # Disabled - user pays for full context
    "compression_model": None,  # None = auto-select model with highest context window
}

# Default prompts (JSON-like structured format)
PROMPTS: dict[str, dict[str, Any]] = {
    "initial": {
        "content": (
            "Analyze the problem from multiple perspectives using evidence-based reasoning and common sense. "
            "Be aware of your inherent biases and actively work to counteract them. "
            "Don't be constrained by current trends or prevailing narratives - trends change, "
            "human priorities shift, and emotional reactions should not limit objective analysis. "
            "Identify the core and outline several distinct, well-reasoned strategies to efficiently SOLVE THE PROBLEM,"
            "grounded in scientific principles and common sense. Avoid unfalsifiable claims, "
            "pseudoscience, and speculative nonsense. Think critically and don't hesitate to "
            "challenge assumptions as the question can be self-restrictive."
            "When proposing tactics or metrics with incomplete evidence, DO NOT drop them; label them explicitly as heuristics. "
            "Use an evidence strength tag for every non-obvious claim: [STRONG]/[MODERATE]/[WEAK]/[ANECDOTAL], "
            "and include a confidence estimate (e.g., 0.2-0.8). "
            "If a precise number lacks a source, convert it to an operational range or test rather than deleting it."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "initial_response",
        },
    },
    "feedback": {
        "content": (
            "Provide feedback that will allow the one to improve. "
            "Identify the most insightful, evidence-based ideas that stand out. "
            "Be aware of confirmation bias and other cognitive biases when evaluating. "
            "Don't favor responses that merely align with popular sentiment - value independent "
            "thinking that challenges current trends when evidence supports it. "
            "Distinguish between verifiable insights and mere speculation. "
            "Note which elements rely on common consensus versus proven methodology."
            "Explicitly list high-utility details (e.g., concrete metrics, micro-behaviors) that are weakly evidenced but useful; "
            "recommend preserving them as labeled heuristics rather than removing them. "
            "Penalize deletion of unique, actionable specifics if they can be retained with uncertainty labels."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "feedback",
        },
    },
    "improvement": {
        "content": (
            "Improve the answer using feedback, grounding it in scientific evidence and "
            "practical reasoning. Recognize and mitigate your own biases in the analysis. "
            "Don't let emotional appeals or fashionable opinions constrain rigorous thinking. "
            "Make the key verifiable insights the central thesis. "
            "Rebuild the entire argument around this main point, removing all generic claims, "
            "unsubstantiated speculation, and secondary details."
            "Do NOT discard high-utility specifics; instead, reframe them with evidence tags and confidence levels. "
            "Translate uncited precise numbers into operational ranges or decision rules. "
            "Append a 'Heuristics Annex' that retains weakly evidenced but useful tactics, clearly labeled. "
            "Produce a short Change Log explaining what was generalized/relocated and why."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "improvement",
        },
    },
    "evaluate": {
        "content": (
            "You are an editor judging analytical depth, scientific rigor and common sense. How insightful "
            "and evidence-based is this answer? Does it rely on proven methodology and sound "
            "reasoning, or does it resort to speculation and unfalsifiable claims? "
            "Be aware that all analysis contains inherent biases - evaluate whether the answer "
            "acknowledges and addresses its own potential biases. Does it demonstrate intellectual "
            "independence by challenging prevailing trends when warranted, or does it merely "
            "echo popular sentiment? Be critical of both originality and factual grounding."
            "Reward explicit evidence tags, confidence reporting, and retention of useful heuristics in an annex. "
            "Down-rank answers that delete actionable details without justification or fail to provide a Change Log. "
            "Check that precise claims without sources were converted to ranges/tests rather than removed."
        ),
        "metadata": {
            "version": "1.0",
            "type": "instruction",
            "phase": "evaluation",
        },
    },
}

# Default Knowledge Bank settings
KNOWLEDGE_BANK: dict[str, Any] = {
    "enabled": True,
    "similarity_threshold": 0.75,  # Cosine similarity threshold for duplicate detection
    "max_insights": 100,  # Maximum insights to keep (LRU eviction)
}

# Default API provider secrets configuration
SECRETS: dict[str, Any] = {
    "providers": {
        "openai": {
            "env_var": "OPENAI_API_KEY",
            "op_path": "op://Personal/OpenAI/api-key",
        },
        "anthropic": {
            "env_var": "ANTHROPIC_API_KEY",
            "op_path": "op://Personal/Anthropic/api-key",
        },
        "vertex_ai": {
            "env_var": "VERTEX_AI_API_KEY",
            "op_path": "op://Personal/VertexAI/api-key",
        },
        "xai": {
            "env_var": "XAI_API_KEY",
            "op_path": "op://Personal/XAI/api-key",
        },
        "google": {
            "env_var": "GOOGLE_API_KEY",
            "op_path": "op://Personal/Google/api-key",
        },
        "cohere": {
            "env_var": "COHERE_API_KEY",
            "op_path": "op://Personal/Cohere/api-key",
        },
        "mistral": {
            "env_var": "MISTRAL_API_KEY",
            "op_path": "op://Personal/Mistral/api-key",
        },
    }
}


def get_defaults() -> dict[str, Any]:
    """
    Get all default configuration values.

    Returns:
        Dictionary containing all default configuration sections.
    """
    return {
        "models": MODELS,
        "retry": RETRY,
        "features": FEATURES,
        "prompts": PROMPTS,
        "knowledge_bank": KNOWLEDGE_BANK,
        "secrets": SECRETS,
    }
