"""
Health check endpoints for AgroVision backend.
Provides liveness and readiness status probes.
"""

from typing import Union
from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import check_db_connectivity

router = APIRouter()


class HealthResponse(BaseModel):
    """Liveness probe response model."""

    status: str = Field(default="healthy", description="Application operational status")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Active runtime environment")


class ReadinessChecks(BaseModel):
    """Subsystem status checks for readiness probe."""

    configuration: str = Field(default="ok", description="Status of centralized settings")
    database: str = Field(..., description="Status of database connectivity ('ok' or 'unavailable')")


class ReadinessResponse(BaseModel):
    """Readiness probe response model."""

    status: str = Field(..., description="Application readiness status ('ready' or 'not_ready')")
    app_name: str = Field(..., description="Application name")
    version: str = Field(..., description="Application version")
    environment: str = Field(..., description="Active runtime environment")
    checks: ReadinessChecks = Field(..., description="Detailed subsystem status checks")


@router.get(
    "",
    response_model=HealthResponse,
    summary="Application Liveness Probe",
    description="Check whether the AgroVision backend is running. Does not depend on external services.",
)
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return application liveness status and version information."""
    return HealthResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        200: {"description": "Application and database ready to accept traffic"},
        503: {"description": "Application not ready (database unavailable)"},
    },
    summary="Application Readiness Probe",
    description="Check whether the AgroVision backend and its database are ready to accept requests.",
)
def get_readiness(
    response: Response,
    settings: Settings = Depends(get_settings),
) -> Union[ReadinessResponse, JSONResponse]:
    """
    Return application readiness status and subsystem checks.
    Evaluates centralized configuration and verifies database connectivity via SELECT 1.
    Returns HTTP 200 if all checks pass, or HTTP 503 if the database is unavailable.
    """
    db_ok = check_db_connectivity()
    db_status = "ok" if db_ok else "unavailable"
    overall_status = "ready" if db_ok else "not_ready"

    payload = ReadinessResponse(
        status=overall_status,
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.ENVIRONMENT,
        checks=ReadinessChecks(
            configuration="ok",
            database=db_status,
        ),
    )

    if not db_ok:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=payload.model_dump(),
        )

    return payload
