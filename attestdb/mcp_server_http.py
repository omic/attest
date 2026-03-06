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
    from starlette.responses import JSONResponse, Response
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

        # Skip auth for health endpoint
        if request.url.path == "/health":
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

    routes = [
        Route("/health", _health, methods=["GET"]),
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
    parser.add_argument("--api-key", default=None, help="Bearer token for auth (overrides ATTEST_API_KEY)")
    args = parser.parse_args()

    db_path = args.db or os.environ.get("ATTEST_DB_PATH", os.environ.get("SUBSTRATE_DB_PATH", "attest.db"))
    api_key = args.api_key or os.environ.get("ATTEST_API_KEY")

    app = create_http_app(db_path=db_path, api_key=api_key)
    if not api_key:
        logger.warning("Starting without authentication. Set ATTEST_API_KEY or --api-key for production use.")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
