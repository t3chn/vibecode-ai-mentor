"""Main FastAPI application for VibeCode AI Mentor."""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.core.config import settings
from src.core.logging import logger
from src.db.connection import db_pool

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting VibeCode AI Mentor API")
    await db_pool.init()
    
    yield
    
    # Shutdown
    logger.info("Shutting down VibeCode AI Mentor API")
    await db_pool.close()


# Create FastAPI app
app = FastAPI(
    title="VibeCode AI Mentor",
    description="AI-powered code quality analysis tool",
    version="0.1.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.is_development else ["https://vibecode.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    """Handle errors and unexpected exceptions."""
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error"},
        )


# Request/response logging middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and responses."""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else None,
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Log response
    logger.info(
        f"Response: {request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": process_time,
        }
    )
    
    # Add custom headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "version": "0.1.0",
    }


@app.get("/config")
async def get_config():
    """Get non-sensitive configuration (development only)."""
    if not settings.is_development:
        return {"error": "Not available in production"}
    
    return {
        "environment": settings.environment,
        "api_port": settings.api_port,
        "debug": settings.debug,
        "log_level": settings.log_level,
        "batch_size": settings.batch_size,
        "max_chunk_size": settings.max_chunk_size,
        "min_chunk_size": settings.min_chunk_size,
        "cache_enabled": settings.enable_cache,
        "cache_ttl": settings.cache_ttl,
        "rate_limit": {
            "requests": settings.rate_limit_requests,
            "period": settings.rate_limit_period,
        },
    }


# Import routes
from src.api import routes  # noqa: E402

# Include routers
app.include_router(routes.router)