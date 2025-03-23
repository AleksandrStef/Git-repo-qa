"""API router for Vanna AI Repository Q&A System."""

from fastapi import APIRouter
from app.api.endpoints import router as api_router

router = APIRouter()

# Include the API endpoints
router.include_router(api_router, prefix="/api/v1")
