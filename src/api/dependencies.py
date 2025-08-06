"""FastAPI dependencies for authentication, database, and rate limiting."""

from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from slowapi import Limiter

from src.core.config import settings
from src.core.logging import logger
from src.db.connection import db_pool

# Security scheme for API key authentication
security = HTTPBearer()


async def get_db_session() -> AsyncGenerator:
    """
    Create a database session for the request.
    
    Yields:
        Database session
    """
    async with db_pool.acquire() as connection:
        try:
            yield connection
        except Exception as e:
            logger.error(f"Database session error: {str(e)}")
            raise
        finally:
            # Connection is automatically released back to pool
            pass


async def get_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Validate API key from Authorization header.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Validated API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    api_key = credentials.credentials
    
    # In development, accept any non-empty API key
    if settings.is_development and api_key:
        return api_key
    
    # TODO: Implement actual API key validation
    # For now, check against a simple environment variable
    valid_api_keys = settings.api_keys.split(",") if settings.api_keys else []
    
    if not api_key or api_key not in valid_api_keys:
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return api_key


async def get_optional_api_key(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[str]:
    """
    Get API key if provided (for endpoints that optionally require auth).
    
    Args:
        credentials: Optional HTTP authorization credentials
        
    Returns:
        API key if provided, None otherwise
    """
    if not credentials:
        return None
    
    try:
        return await get_api_key(credentials)
    except HTTPException:
        return None


def get_rate_limiter(request: Request) -> Limiter:
    """
    Get rate limiter instance from app state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Rate limiter instance
    """
    return request.app.state.limiter


async def check_rate_limit(
    request: Request,
    api_key: str = Depends(get_api_key),
) -> None:
    """
    Check rate limit for the current API key.
    
    Args:
        request: FastAPI request object
        api_key: Validated API key
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Use API key as rate limit key instead of IP address
    # This allows per-API-key rate limiting
    
    # TODO: Implement custom rate limit logic based on API key tiers
    # For now, using the default rate limiter with IP-based limiting
    pass


class RateLimitDependency:
    """
    Rate limit dependency with configurable limits per endpoint.
    """
    
    def __init__(self, requests: int = 10, period: int = 60):
        """
        Initialize rate limit dependency.
        
        Args:
            requests: Number of requests allowed
            period: Time period in seconds
        """
        self.requests = requests
        self.period = period
    
    async def __call__(
        self,
        request: Request,
        api_key: str = Depends(get_api_key),
    ) -> None:
        """Check rate limit for the request."""
        # TODO: Implement per-API-key rate limiting with Redis
        # For now, this is a placeholder
        pass


# Pre-configured rate limiters for different endpoints
rate_limit_analyze = RateLimitDependency(
    requests=settings.rate_limit_requests,
    period=settings.rate_limit_period
)

rate_limit_index = RateLimitDependency(
    requests=5,  # Fewer requests for resource-intensive operations
    period=300   # 5 minutes
)

rate_limit_search = RateLimitDependency(
    requests=50,  # More requests for search
    period=60     # 1 minute
)