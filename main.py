"""Main entry point for Vanna AI Repository Q&A System."""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router as api_router
from app.core.config import get_settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Vanna AI Repository Q&A System",
    description="An agentic AI solution to answer questions about the Vanna AI repository",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Vanna AI Repository Q&A System API",
        "docs_url": "/docs",
        "api_prefix": get_settings()["api_prefix"]
    }

if __name__ == "__main__":
    logger.info("Starting Vanna AI Repository Q&A System")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
