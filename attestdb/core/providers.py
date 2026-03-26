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
