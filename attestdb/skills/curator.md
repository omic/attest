# Using the Curator

## What It Does

The Curator triages incoming claims: store, skip, or flag for review. It uses LLM prompts to assess claim quality and relevance, with a heuristic fallback for batch processing.

## Input Sanitization (Security-Critical)

**Always sanitize user input before embedding in LLM prompts:**

```python
clean_text = self._sanitize(raw_input)
prompt = f"Evaluate this claim: {clean_text}"
```

`_sanitize()` strips control characters and prompt injection attempts. Without it, malicious claim text can hijack the LLM prompt.

## Error Handling

Never use bare `except:` blocks — they hide real errors:

```python
# Wrong
try:
    result = self._triage(ci)
except:
    pass

# Correct
try:
    result = self._triage(ci)
except (KeyError, ValueError) as e:
    logger.warning("Triage failed: %s", e, exc_info=True)
```

## Batch Triage Fallback

When the LLM is unavailable or rate-limited, use the heuristic fallback:

```python
result = self._heuristic_triage(ci)
```

This uses rule-based scoring (predicate importance, source reliability, entity degree) instead of LLM evaluation.

## Testing

- Test that `_sanitize()` strips injection attempts
- Test heuristic triage produces valid store/skip/flag decisions
- Test LLM triage with mocked API responses
- Test fallback activates when LLM call fails
