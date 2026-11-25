"""
Central API router combining all route modules.
"""

from fastapi import APIRouter
from backend.app.api.routes import health

api_router = APIRouter()

# Register route modules
api_router.include_router(health.router, prefix="/health", tags=["Health"])
