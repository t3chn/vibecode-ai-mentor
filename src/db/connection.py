"""Database connection management for TiDB."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import aiomysql
from aiomysql import Connection, Pool

from src.core.config import settings
from src.core.logging import logger


class DatabasePool:
    """Manages database connection pooling."""

    def __init__(self) -> None:
        self._pool: Optional[Pool] = None

    async def init(self) -> None:
        """Initialize the connection pool."""
        if self._pool is not None:
            return

        try:
            self._pool = await aiomysql.create_pool(
                host=settings.tidb_host,
                port=settings.tidb_port,
                user=settings.tidb_user,
                password=settings.tidb_password,
                db=settings.tidb_database,
                minsize=1,
                maxsize=10,
                pool_recycle=3600,
                autocommit=True,
            )
            logger.info(
                f"Database pool initialized for {settings.tidb_host}:{settings.tidb_port}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._pool = None
            logger.info("Database pool closed")

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """Acquire a connection from the pool."""
        if self._pool is None:
            await self.init()

        async with self._pool.acquire() as conn:
            yield conn


# Global database pool instance
db_pool = DatabasePool()


async def get_db_connection() -> AsyncGenerator[Connection, None]:
    """Get a database connection for dependency injection."""
    async with db_pool.acquire() as conn:
        yield conn