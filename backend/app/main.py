"""
Main FastAPI application entrypoint for AgroVision backend.
Initializes application, configures CORS, registers routes, and handles lifecycle.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import api_router
from backend.app.core.config import get_settings

logger = logging.getLogger("agrovision")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager for startup and shutdown events.
    Notice: ML model loading is intentionally deferred from this milestone.
    """
    settings = get_settings()
    logger.info(
        "Starting %s v%s in '%s' environment (debug=%s)",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.ENVIRONMENT,
        settings.DEBUG,
    )
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_application() -> FastAPI:
    """Construct and configure the FastAPI application instance."""
    settings = get_settings()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        description="AgroVision AI-Powered Crop Disease Screening and Advisory System API",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register centralized API router
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    return app


app = create_application()
