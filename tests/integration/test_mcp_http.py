"""Integration tests for the MCP HTTP transport wrapper."""

import json

import pytest

from attestdb.mcp_server_http import create_http_app

try:
    from starlette.testclient import TestClient
except ImportError:
    pytest.skip("starlette not installed", allow_module_level=True)


@pytest.fixture()
def http_app(tmp_path):
    return create_http_app(db_path=str(tmp_path / "test"))


@pytest.fixture()
def client(http_app):
    return TestClient(http_app)


@pytest.fixture()
def auth_http_app(tmp_path):
    return create_http_app(db_path=str(tmp_path / "test_auth"), api_key="test-secret-key")


@pytest.fixture()
def auth_client(auth_http_app):
    return TestClient(auth_http_app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "entity_count" in data
        assert "claim_count" in data

    def test_health_correct_shape(self, client):
        r = client.get("/health")
        data = r.json()
        assert isinstance(data["entity_count"], int)
        assert isinstance(data["claim_count"], int)


class TestCORS:
    def test_cors_headers_present(self, client):
        r = client.options(
            "/mcp",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert "access-control-allow-origin" in r.headers


class TestRBAC:
    def test_role_header_logged_not_enforced(self, client):
        """X-Attest-Role is logged but not enforced — request succeeds."""
        r = client.get("/health", headers={"X-Attest-Role": "admin"})
        assert r.status_code == 200


class TestAuth:
    def test_auth_required_when_key_set(self, auth_client):
        r = auth_client.get("/health")
        # Health is exempted from auth
        assert r.status_code == 200

    def test_mcp_requires_auth(self, auth_client):
        r = auth_client.post("/mcp", content=b"{}")
        assert r.status_code == 401

    def test_mcp_with_valid_auth(self, auth_client):
        r = auth_client.post(
            "/mcp",
            content=b"{}",
            headers={"Authorization": "Bearer test-secret-key"},
        )
        # May get 400/422 since the body isn't valid MCP, but not 401
        assert r.status_code != 401

    def test_mcp_with_wrong_auth(self, auth_client):
        r = auth_client.post(
            "/mcp",
            content=b"{}",
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert r.status_code == 401
