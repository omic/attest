"""HTTP transport wrapper for the Attest MCP server.

Wraps the FastMCP streamable-http app with health endpoint, CORS, and optional
Bearer-token auth.

Usage:
    attest-mcp-http                          # default :8893
    attest-mcp-http --port 9000 --api-key sk-xxx
    ATTEST_API_KEY=sk-xxx attest-mcp-http

Environment variables:
    ATTEST_DB_PATH — database file path (default: "attest.db")
    ATTEST_API_KEY — optional Bearer token for auth
"""

from __future__ import annotations

import atexit
import logging
import os
import secrets
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

try:
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.middleware.cors import CORSMiddleware
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Mount, Route
except ImportError:
    raise ImportError(
        "HTTP transport requires starlette and uvicorn. "
        "Install with: pip install attestdb[mcp]"
    )


class _BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require Bearer token on all routes except /health."""

    def __init__(self, app, api_key: str):
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(self, request: Request, call_next):
        # Log X-Attest-Role if present (don't enforce)
        role = request.headers.get("x-attest-role", "")
        if role:
            logger.info("X-Attest-Role: %s", role)

        # Skip auth for health and extension API endpoints from localhost
        if request.url.path == "/health":
            return await call_next(request)
        if request.url.path.startswith("/api/"):
            client_host = request.client.host if request.client else ""
            if client_host in ("127.0.0.1", "::1", "localhost"):
                return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        token = auth_header[7:]
        if not secrets.compare_digest(token, self._api_key):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)

        return await call_next(request)


def create_http_app(db_path: str = "attest.db", api_key: str | None = None) -> Starlette:
    """Create a Starlette app wrapping the MCP server with health/CORS/auth.

    Args:
        db_path: Path to AttestDB file (or ":memory:").
        api_key: Optional Bearer token. If set, non-health routes require it.

    Returns:
        A Starlette app ready for uvicorn.run().
    """
    from attestdb.infrastructure.attest_db import AttestDB
    from attestdb.mcp_server import configure, mcp

    db = AttestDB(db_path, embedding_dim=None)
    configure(db)
    try:
        from attestdb.intelligence.ai_tools_vocabulary import register_ai_tools_vocabulary
        register_ai_tools_vocabulary(db)
    except ImportError:
        pass  # AI tools vocabulary requires attestdb-intelligence

    # Ensure data is flushed on process exit (Rust store only persists on close())
    atexit.register(db.close)

    @asynccontextmanager
    async def _lifespan(app):
        yield
        db.close()

    # Get FastMCP's streamable-http Starlette app
    mcp_app = mcp.streamable_http_app()

    def _health(request: Request) -> JSONResponse:
        stats = db.stats()
        return JSONResponse({
            "status": "ok",
            "entity_count": stats.get("entity_count", 0),
            "claim_count": stats.get("claim_count", 0),
        })

    async def _ingest_text(request: Request) -> JSONResponse:
        """REST endpoint for Chrome extension and other lightweight clients."""
        import json as _json

        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        text = body.get("text", "")
        source_id = body.get("source_id", "chrome-capture")
        conversation_id = body.get("conversation_id", "")
        provider = body.get("provider", "")
        if not text or not text.strip():
            return JSONResponse({"error": "empty text"}, status_code=400)
        try:
            result = db.ingest_text(
                text,
                source_id=source_id or f"{provider}-capture",
            )
            return JSONResponse({
                "n_valid": getattr(result, "n_valid", 0),
                "raw_count": getattr(result, "raw_count", 0),
                "warnings": getattr(result, "warnings", []),
            })
        except ImportError:
            # No intelligence layer — store as raw chat claim instead
            try:
                claim_id = db.ingest(
                    subject=(conversation_id or f"{provider}:chat", "conversation"),
                    predicate=("has_response", "chat_response"),
                    object=(text[:200], "text"),
                    source_type="chat_capture",
                    source_id=source_id or f"{provider}-capture",
                    payload={"schema_ref": "chat_capture/v1", "data": {
                        "full_text": text, "provider": provider,
                        "conversation_id": conversation_id,
                    }},
                )
                return JSONResponse({"n_valid": 1, "raw_count": 1, "claim_id": claim_id, "warnings": []})
            except Exception as exc:
                return JSONResponse({"error": str(exc)}, status_code=500)
        except Exception as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)

    async def _ingest_batch(request: Request) -> JSONResponse:
        """Batch endpoint — accepts array of messages."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)
        messages = body.get("messages", [])
        if not messages:
            return JSONResponse({"error": "empty messages"}, status_code=400)
        total_valid = 0
        total_raw = 0
        errors = []
        for msg in messages:
            text = msg.get("text", "")
            if not text or not text.strip():
                continue
            source_id = msg.get("source_id", msg.get("provider", "chrome-capture"))
            try:
                result = db.ingest_text(text, source_id=source_id)
                total_valid += getattr(result, "n_valid", 0)
                total_raw += getattr(result, "raw_count", 0)
            except ImportError:
                # Fallback to raw claim
                try:
                    db.ingest(
                        subject=(msg.get("conversation_id", f"{source_id}:chat"), "conversation"),
                        predicate=("has_response", "chat_response"),
                        object=(text[:200], "text"),
                        source_type="chat_capture",
                        source_id=source_id,
                        payload={"schema_ref": "chat_capture/v1", "data": {
                            "full_text": text, "provider": msg.get("provider", ""),
                        }},
                    )
                    total_valid += 1
                    total_raw += 1
                except Exception as exc:
                    errors.append(str(exc))
            except Exception as exc:
                errors.append(str(exc))
        return JSONResponse({
            "n_valid": total_valid, "raw_count": total_raw,
            "message_count": len(messages), "errors": errors,
        })

    async def _synthesize(request: Request) -> JSONResponse:
        """Synthesize consensus from multiple provider responses."""
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"error": "invalid JSON"}, status_code=400)

        question = body.get("question", "")
        responses = body.get("responses", {})  # {provider: text}

        if len(responses) < 2:
            return JSONResponse({"error": "need at least 2 responses"}, status_code=400)

        # Try LLM-powered synthesis via agent_consensus
        try:
            from attestdb.intelligence.consensus import ConsensusEngine
            engine = ConsensusEngine()
            # Use the consensus engine's synthesize method directly
            from attestdb.intelligence.consensus import ProviderResponse
            provider_responses = [
                ProviderResponse(provider=p, model=p, response=t, round_number=1)
                for p, t in responses.items()
            ]
            synthesis = engine._synthesize(question, provider_responses)
            return JSONResponse({
                "consensus": synthesis,
                "method": "llm",
                "providers": list(responses.keys()),
            })
        except (ImportError, Exception) as exc:
            # Fall back to simple comparison
            import logging
            logging.getLogger(__name__).debug("Synthesis fallback: %s", exc)

            # Simple synthesis: find common themes
            providers = list(responses.keys())
            summary_parts = []
            summary_parts.append(f"Responses from {len(providers)} providers ({', '.join(providers)}):")
            for p, text in responses.items():
                # Truncate long responses
                preview = text[:500] + "..." if len(text) > 500 else text
                label = {"chatgpt": "ChatGPT", "claude": "Claude", "gemini": "Gemini"}.get(p, p)
                summary_parts.append(f"\n{label} ({len(text)} chars):\n{preview}")

            return JSONResponse({
                "consensus": "\n".join(summary_parts),
                "method": "summary",
                "providers": providers,
            })

    routes = [
        Route("/health", _health, methods=["GET"]),
        Route("/api/ingest_text", _ingest_text, methods=["POST"]),
        Route("/api/ingest_batch", _ingest_batch, methods=["POST"]),
        Route("/api/synthesize", _synthesize, methods=["POST"]),
        Mount("/mcp", app=mcp_app),
    ]

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        ),
    ]

    if api_key:
        middleware.append(Middleware(_BearerAuthMiddleware, api_key=api_key))

    return Starlette(routes=routes, middleware=middleware, lifespan=_lifespan)


def main():
    """Entry point: attest-mcp-http"""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="attest-mcp-http")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8893, help="Bind port (default: 8893)")
    parser.add_argument("--db", default=None, help="DB path (overrides ATTEST_DB_PATH)")
    parser.add_argument(
        "--api-key", default=None,
        help="Bearer token for auth (overrides ATTEST_API_KEY)",
    )
    args = parser.parse_args()

    db_path = args.db or os.environ.get(
        "ATTEST_DB_PATH",
        os.environ.get("SUBSTRATE_DB_PATH", "attest.db"),
    )
    api_key = args.api_key or os.environ.get("ATTEST_API_KEY")

    app = create_http_app(db_path=db_path, api_key=api_key)
    if not api_key:
        logger.warning(
            "Starting without authentication. "
            "Set ATTEST_API_KEY or --api-key for production use."
        )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
