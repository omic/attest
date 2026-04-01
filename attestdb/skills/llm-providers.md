# Making LLM API Calls

## Provider Config

9 providers in `attestdb/core/providers.py`. All OpenAI-compatible. Config dict uses specific field names:

```python
from attestdb.core.providers import PROVIDERS

cfg = PROVIDERS["groq"]
# Correct field access:
env_key = cfg.get('env_key', cfg.get('env', ''))
model = cfg.get('default_model', cfg.get('model', ''))
```

**Critical:** The fields are `env_key` and `default_model` — NOT `env` and `model`. Using the wrong names silently returns empty strings, causing all LLM calls to fail without errors.

## Fallback Chain

```python
from attestdb.core.providers import EXTRACTION_FALLBACK_CHAIN
# gemini → together → openai → deepseek → grok
```

TextExtractor walks this chain, using the first provider with an API key set.

## GPT-5 Gotchas

- Use `max_completion_tokens` instead of `max_tokens` — GPT-5 rejects `max_tokens`
- **Drop the `temperature` parameter** entirely for GPT-5 reasoning models
- GPT-5 Nano/Mini are reasoning models — they return empty `content`, not useful for structured extraction

TextExtractor handles both `max_tokens` and `max_completion_tokens` automatically.

## Groq

- Use the stable default `llama-3.3-70b-versatile` (not `llama-4-maverick` which got 403-blocked)
- Don't hardcode model names — they change frequently

## Making Calls

All providers use the OpenAI-compatible `/chat/completions` endpoint. Use `requests` (not `httpx`) to avoid uvicorn deadlocks:

```python
import requests
resp = requests.post(
    base_url.rstrip("/") + "/chat/completions",
    headers={"Authorization": f"Bearer {api_key}"},
    json={"model": model, "messages": messages, "max_tokens": 600},
    timeout=20,  # ALWAYS set a timeout
)
```

## Rules

1. **Always set a timeout** on LLM API calls — no timeout = indefinite blocking
2. **Always use `requests`** for LLM calls in uvicorn context (httpx causes deadlocks)
3. **Always look up config with dual-key fallback**: `cfg.get('env_key', cfg.get('env', ''))`
