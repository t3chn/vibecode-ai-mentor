#!/usr/bin/env python3
"""
VibeCode AI Mentor: AI-powered code quality analysis tool.

This module provides the main entry point for running the API server.
"""

import uvicorn

from src.core.config import settings
from src.core.logging import logger


def main() -> None:
    """Run the FastAPI application."""
    logger.info(f"Starting VibeCode AI Mentor API on {settings.api_host}:{settings.api_port}")
    
    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.reload if settings.is_development else False,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
