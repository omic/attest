"""LLM provider configurations — shared across consensus, chat, and curator.

All providers use OpenAI-compatible APIs via the ``openai`` package.
"""

from __future__ import annotations

# Provider configurations (all OpenAI-compatible)
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "env_key": "GOOGLE_API_KEY",
        "default_model": "gemini-3.1-flash-lite-preview",
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "env_key": "TOGETHER_API_KEY",
        "default_model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "env_key": "OPENAI_API_KEY",
        "default_model": "gpt-4.1-mini",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-chat",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1",
        "env_key": "GROK_API_KEY",
        "default_model": "grok-4-1-fast-non-reasoning",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "env_key": "OPENROUTER_API_KEY",
        "default_model": "deepseek/deepseek-v3.2",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1/",
        "env_key": "ANTHROPIC_API_KEY",
        "default_model": "claude-haiku-4-5-20251001",
    },
    "glm": {
        "base_url": "https://api.z.ai/api/paas/v4/",
        "env_key": "GLM_API_KEY",
        "default_model": "glm-4-flash",
    },
}

# Recommended fallback chain for text extraction, ordered by benchmark performance.
# groq (fastest) → gemini → together → openai → deepseek → grok
EXTRACTION_FALLBACK_CHAIN = ["groq", "gemini", "together", "openai", "deepseek", "grok"]

# Token pricing per 1M tokens (input, output) in USD.
# Used by prompt_kit tools for cost estimation and waste analysis.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-6": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
    # OpenAI
    "gpt-4.1": (2.0, 8.0),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "o3": (10.0, 40.0),
    "o4-mini": (1.10, 4.40),
    # Google
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-3.1-flash-lite-preview": (0.075, 0.30),
    # Open-source / inference providers
    "deepseek-chat": (0.14, 0.28),
    "grok-4-1-fast-non-reasoning": (3.0, 15.0),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "glm-4-flash": (0.0, 0.0),  # free tier
    # Fallback for unknown models
    "_default": (0.50, 1.50),
}

# Model tier classification for routing recommendations.
# tier 1 = reasoning (expensive), tier 2 = execution (mid), tier 3 = cleanup (cheap)
MODEL_TIERS: dict[str, int] = {
    "claude-opus-4-6": 1,
    "o3": 1,
    "gpt-4.1": 1,
    "gemini-2.5-pro": 1,
    "grok-4-1-fast-non-reasoning": 1,
    "claude-sonnet-4-6": 2,
    "gpt-4.1-mini": 2,
    "o4-mini": 2,
    "deepseek-chat": 2,
    "gemini-2.5-flash": 2,
    "llama-3.3-70b-versatile": 2,
    "claude-haiku-4-5-20251001": 3,
    "gpt-4.1-nano": 3,
    "gemini-3.1-flash-lite-preview": 3,
    "glm-4-flash": 3,
}

TIER_NAMES = {1: "reasoning", 2: "execution", 3: "cleanup"}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost for a model call."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["_default"])
    return (prompt_tokens * pricing[0] + completion_tokens * pricing[1]) / 1_000_000


def get_model_tier(model: str) -> int:
    """Return tier (1-3) for a model, defaulting to 2."""
    return MODEL_TIERS.get(model, 2)


def load_env_file(env_path: str) -> dict[str, str]:
    """Load key=value pairs from a .env file."""
    env_vars: dict[str, str] = {}
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    env_vars[key] = value
    except FileNotFoundError:
        pass
    return env_vars
