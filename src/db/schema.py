"""TiDB-specific schema definitions and vector index creation."""

from typing import Optional

from sqlalchemy import Index, MetaData, create_engine, text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.core.config import settings
from src.core.logging import logger
from src.db.models import Base


# TiDB Vector Index DDL Template
VECTOR_INDEX_DDL = """
CREATE VECTOR INDEX IF NOT EXISTS {index_name} 
ON {table_name}({column_name})
USING HNSW
WITH (DISTANCE_METRIC = 'cosine', M = 16, EFCONSTRUCTION = 200);
"""

# Additional indexes for performance
PERFORMANCE_INDEXES = [
    # Composite index for repository queries
    "CREATE INDEX IF NOT EXISTS idx_repo_status_date ON repositories(status, last_indexed_at);",
    
    # Composite index for snippet queries
    "CREATE INDEX IF NOT EXISTS idx_snippet_repo_lang ON code_snippets(repository_id, language);",
    "CREATE INDEX IF NOT EXISTS idx_snippet_path ON code_snippets(file_path(255));",
    
    # Index for analysis results
    "CREATE INDEX IF NOT EXISTS idx_analysis_snippet_date ON analysis_results(snippet_id, created_at);",
    
    # Cache expiration indexes
    "CREATE INDEX IF NOT EXISTS idx_embedding_cache_expire ON embedding_cache(expires_at);",
    "CREATE INDEX IF NOT EXISTS idx_search_cache_expire ON search_cache(expires_at);",
]


class DatabaseManager:
    """Manages database schema and connections."""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or settings.tidb_connection_string
        # Replace pymysql with aiomysql for async
        self.async_connection_string = self.connection_string.replace(
            "mysql+pymysql://", "mysql+aiomysql://"
        )

    async def create_async_engine(self) -> AsyncEngine:
        """Create async SQLAlchemy engine."""
        return create_async_engine(
            self.async_connection_string,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def create_sync_engine(self):
        """Create sync SQLAlchemy engine for migrations."""
        return create_engine(
            self.connection_string,
            echo=settings.debug,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    async def create_tables(self, engine: AsyncEngine) -> None:
        """Create all tables in the database."""
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")

    async def create_vector_indexes(self, engine: AsyncEngine) -> None:
        """Create TiDB vector indexes for similarity search."""
        vector_indexes = [
            {
                "index_name": "idx_code_snippets_embedding",
                "table_name": "code_snippets",
                "column_name": "embedding",
            },
            {
                "index_name": "idx_embedding_cache_vector",
                "table_name": "embedding_cache",
                "column_name": "embedding",
            },
        ]

        async with engine.begin() as conn:
            for idx in vector_indexes:
                try:
                    ddl = VECTOR_INDEX_DDL.format(**idx)
                    await conn.execute(text(ddl))
                    logger.info(f"Created vector index: {idx['index_name']}")
                except Exception as e:
                    logger.error(f"Failed to create vector index {idx['index_name']}: {e}")
                    # Continue with other indexes

    async def create_performance_indexes(self, engine: AsyncEngine) -> None:
        """Create additional performance indexes."""
        async with engine.begin() as conn:
            for idx_ddl in PERFORMANCE_INDEXES:
                try:
                    await conn.execute(text(idx_ddl))
                    logger.info(f"Created performance index")
                except Exception as e:
                    logger.error(f"Failed to create index: {e}")
                    # Continue with other indexes

    async def initialize_database(self) -> None:
        """Initialize database with all tables and indexes."""
        engine = await self.create_async_engine()
        try:
            # Create tables
            await self.create_tables(engine)
            
            # Create vector indexes
            await self.create_vector_indexes(engine)
            
            # Create performance indexes
            await self.create_performance_indexes(engine)
            
            logger.info("Database initialization completed")
        finally:
            await engine.dispose()

    def get_async_session_maker(self, engine: AsyncEngine) -> async_sessionmaker:
        """Get async session maker for database operations."""
        return async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )


# Vector search query templates
VECTOR_SEARCH_QUERY = """
SELECT 
    id,
    repository_id,
    file_path,
    language,
    content,
    start_line,
    end_line,
    complexity_score,
    VEC_COSINE_DISTANCE(embedding, :query_embedding) as similarity
FROM code_snippets
WHERE embedding IS NOT NULL
    {filters}
ORDER BY similarity ASC
LIMIT :limit;
"""

HYBRID_SEARCH_QUERY = """
SELECT 
    cs.id,
    cs.repository_id,
    cs.file_path,
    cs.language,
    cs.content,
    cs.start_line,
    cs.end_line,
    cs.complexity_score,
    VEC_COSINE_DISTANCE(cs.embedding, :query_embedding) as similarity,
    r.name as repository_name,
    r.url as repository_url
FROM code_snippets cs
JOIN repositories r ON cs.repository_id = r.id
WHERE cs.embedding IS NOT NULL
    AND r.status = 'completed'
    {filters}
ORDER BY similarity ASC
LIMIT :limit;
"""


def build_vector_search_filter(
    language: Optional[str] = None,
    repository_id: Optional[str] = None,
    min_complexity: Optional[float] = None,
    max_complexity: Optional[float] = None,
) -> str:
    """Build WHERE clause filters for vector search."""
    filters = []
    
    if language:
        filters.append(f"AND language = :language")
    
    if repository_id:
        filters.append(f"AND repository_id = :repository_id")
    
    if min_complexity is not None:
        filters.append(f"AND complexity_score >= :min_complexity")
    
    if max_complexity is not None:
        filters.append(f"AND complexity_score <= :max_complexity")
    
    return " ".join(filters)


# Database manager singleton
db_manager = DatabaseManager()