"""
Automated tests for AgroVision FastAPI health and readiness routes.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from backend.app.core.config import get_settings
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Verify GET /api/v1/health returns HTTP 200 and healthy status."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "app_name" in data
    assert "version" in data
    assert "environment" in data


def test_health_response_structure():
    """Verify exact schema of the health check response."""
    settings = get_settings()
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == settings.APP_NAME
    assert data["version"] == settings.APP_VERSION
    assert data["environment"] == settings.ENVIRONMENT


def test_health_endpoint_independent_of_database():
    """Verify liveness probe remains HTTP 200 even when database is completely down."""
    with patch("backend.app.api.routes.health.check_db_connectivity", return_value=False):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_readiness_endpoint_when_database_available():
    """Verify GET /api/v1/health/ready returns HTTP 200 when database connectivity succeeds."""
    with patch("backend.app.api.routes.health.check_db_connectivity", return_value=True):
        settings = get_settings()
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ready"
        assert data["app_name"] == settings.APP_NAME
        assert data["version"] == settings.APP_VERSION
        assert data["environment"] == settings.ENVIRONMENT
        assert data["checks"] == {
            "configuration": "ok",
            "database": "ok",
        }


def test_readiness_endpoint_when_database_unavailable():
    """Verify GET /api/v1/health/ready returns HTTP 503 when database is unavailable."""
    with patch("backend.app.api.routes.health.check_db_connectivity", return_value=False):
        settings = get_settings()
        response = client.get("/api/v1/health/ready")
        assert response.status_code == 503

        data = response.json()
        assert data["status"] == "not_ready"
        assert data["app_name"] == settings.APP_NAME
        assert data["version"] == settings.APP_VERSION
        assert data["environment"] == settings.ENVIRONMENT
        assert data["checks"] == {
            "configuration": "ok",
            "database": "unavailable",
        }
        # Ensure no passwords or connection strings are leaked
        assert "postgresql" not in response.text.lower()
        assert "password" not in response.text.lower()


def test_application_metadata():
    """Verify that application metadata matches centralized settings."""
    settings = get_settings()
    assert app.title == settings.APP_NAME
    assert app.version == settings.APP_VERSION
    assert app.debug == settings.DEBUG


def test_openapi_schema_endpoint():
    """Verify OpenAPI JSON schema endpoint is accessible under the configured prefix."""
    settings = get_settings()
    response = client.get(f"{settings.API_V1_PREFIX}/openapi.json")
    assert response.status_code == 200

    schema = response.json()
    assert schema["info"]["title"] == settings.APP_NAME
    assert schema["info"]["version"] == settings.APP_VERSION
    assert f"{settings.API_V1_PREFIX}/health" in schema["paths"]
    assert f"{settings.API_V1_PREFIX}/health/ready" in schema["paths"]


def test_swagger_ui_endpoint():
    """Verify Swagger UI documentation is available at /docs."""
    response = client.get("/docs")
    assert response.status_code == 200
    assert "swagger-ui" in response.text.lower()


def test_redoc_ui_endpoint():
    """Verify ReDoc documentation is available at /redoc."""
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "redoc" in response.text.lower()
