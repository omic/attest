# Adding MCP Tools

## Architecture

The MCP server (`attestdb/mcp_server.py`) exposes 84+ tools that AI coding agents call. Server name is `"brain"` (not `"attest"`). Install: `attest-mcp install` or `attest brain install`.

## Adding a Tool

```python
@server.tool()
async def my_tool(subject: str, detail: str) -> str:
    """One-line description of what this tool does.

    Args:
        subject: The entity or topic name
        detail: Specific detail to record
    """
    # Keep the API simple: 2-3 args, not 10
    db = _get_db()
    result = db.ingest(ClaimInput(...))
    return f"Recorded: {subject}"
```

## Design Rules

1. **3-arg APIs, not 10-arg.** `attest_learned(subject, insight, type)` beats full claim triple construction. Agents won't use complex APIs.
2. **Write detailed tool descriptions.** Terse descriptions cause LLMs to misuse tools. Include expected behavior and common error modes.
3. **Context-aware recall.** Use git status/log to decide what's relevant — don't dump everything. Agents ignore large context dumps.

## Type Conversion

When converting string arguments to tuples for claim construction:

```python
# Wrong — splits "hello" into ('h','e','l','l','o')
pair = tuple(some_string)

# Correct — use _to_pair() helper
def _to_pair(val):
    if isinstance(val, str):
        return (val, val)
    return tuple(val)
```

## Installation Config

- Server name: `"brain"` in all MCP configs
- No `type` field in the config (Claude Code rejects `type: stdio`)
- Check parent directories for stale `.mcp.json` files — they override project config

## Testing

- Test tool returns valid string responses
- Test type conversion handles strings vs tuples
- Test error cases return helpful messages, not tracebacks
