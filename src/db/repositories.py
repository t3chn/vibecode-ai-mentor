"""Database repository pattern for common operations."""

from typing import Optional, List
from uuid import UUID
from datetime import datetime, timedelta

from sqlalchemy import select, update, delete, and_, or_, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from src.core.logging import logger
from src.db.models import (
    Repository,
    RepositoryStatus,
    CodeSnippet,
    AnalysisResult,
    EmbeddingCache,
    SearchCache,
)
from src.db.schema import VECTOR_SEARCH_QUERY, HYBRID_SEARCH_QUERY, build_vector_search_filter


class RepositoryRepo:
    """Repository operations for Repository model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, name: str, url: Optional[str] = None) -> Repository:
        """Create a new repository."""
        repo = Repository(name=name, url=url)
        self.session.add(repo)
        await self.session.commit()
        await self.session.refresh(repo)
        return repo

    async def get_by_id(self, repo_id: UUID) -> Optional[Repository]:
        """Get repository by ID."""
        result = await self.session.execute(
            select(Repository).where(Repository.id == repo_id)
        )
        return result.scalar_one_or_none()

    async def get_by_name(self, name: str) -> Optional[Repository]:
        """Get repository by name."""
        result = await self.session.execute(
            select(Repository).where(Repository.name == name)
        )
        return result.scalar_one_or_none()

    async def update_status(
        self, repo_id: UUID, status: RepositoryStatus, total_files: Optional[int] = None
    ) -> None:
        """Update repository status."""
        update_data = {"status": status}
        if status == RepositoryStatus.COMPLETED:
            update_data["last_indexed_at"] = datetime.utcnow()
        if total_files is not None:
            update_data["total_files"] = total_files

        await self.session.execute(
            update(Repository).where(Repository.id == repo_id).values(**update_data)
        )
        await self.session.commit()


class CodeSnippetRepo:
    """Repository operations for CodeSnippet model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_batch(self, snippets: List[dict]) -> List[CodeSnippet]:
        """Create multiple code snippets in batch."""
        snippet_objects = [CodeSnippet(**data) for data in snippets]
        self.session.add_all(snippet_objects)
        await self.session.commit()
        return snippet_objects

    async def get_by_id(self, snippet_id: UUID) -> Optional[CodeSnippet]:
        """Get code snippet by ID with relationships."""
        result = await self.session.execute(
            select(CodeSnippet)
            .options(selectinload(CodeSnippet.repository))
            .where(CodeSnippet.id == snippet_id)
        )
        return result.scalar_one_or_none()

    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        language: Optional[str] = None,
        repository_id: Optional[UUID] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
    ) -> List[dict]:
        """Search for similar code snippets using vector similarity."""
        filters = build_vector_search_filter(
            language=language,
            repository_id=str(repository_id) if repository_id else None,
            min_complexity=min_complexity,
            max_complexity=max_complexity,
        )
        
        query = VECTOR_SEARCH_QUERY.format(filters=filters)
        
        # Prepare parameters
        params = {
            "query_embedding": f"[{','.join(map(str, embedding))}]",
            "limit": limit,
        }
        if language:
            params["language"] = language
        if repository_id:
            params["repository_id"] = str(repository_id)
        if min_complexity is not None:
            params["min_complexity"] = min_complexity
        if max_complexity is not None:
            params["max_complexity"] = max_complexity
        
        result = await self.session.execute(text(query), params)
        return [dict(row) for row in result.mappings()]

    async def update_embedding(self, snippet_id: UUID, embedding: List[float]) -> None:
        """Update code snippet embedding."""
        await self.session.execute(
            update(CodeSnippet)
            .where(CodeSnippet.id == snippet_id)
            .values(embedding=embedding)
        )
        await self.session.commit()


class AnalysisResultRepo:
    """Repository operations for AnalysisResult model."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        snippet_id: UUID,
        recommendations: List[dict],
        similar_patterns: List[dict],
        quality_score: float,
    ) -> AnalysisResult:
        """Create analysis result."""
        result = AnalysisResult(
            snippet_id=snippet_id,
            recommendations=recommendations,
            similar_patterns=similar_patterns,
            quality_score=quality_score,
        )
        self.session.add(result)
        await self.session.commit()
        await self.session.refresh(result)
        return result

    async def get_latest_by_snippet(self, snippet_id: UUID) -> Optional[AnalysisResult]:
        """Get latest analysis result for a snippet."""
        result = await self.session.execute(
            select(AnalysisResult)
            .where(AnalysisResult.snippet_id == snippet_id)
            .order_by(AnalysisResult.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()


class CacheRepo:
    """Repository operations for cache models."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_embedding_cache(
        self, content_hash: str
    ) -> Optional[EmbeddingCache]:
        """Get cached embedding if not expired."""
        result = await self.session.execute(
            select(EmbeddingCache).where(
                and_(
                    EmbeddingCache.content_hash == content_hash,
                    EmbeddingCache.expires_at > datetime.utcnow(),
                )
            )
        )
        return result.scalar_one_or_none()

    async def set_embedding_cache(
        self,
        content_hash: str,
        embedding: List[float],
        model_name: str,
        ttl_seconds: int = 3600,
    ) -> EmbeddingCache:
        """Cache embedding with TTL."""
        cache_entry = EmbeddingCache(
            content_hash=content_hash,
            embedding=embedding,
            model_name=model_name,
            expires_at=datetime.utcnow() + timedelta(seconds=ttl_seconds),
        )
        self.session.add(cache_entry)
        await self.session.commit()
        return cache_entry

    async def cleanup_expired_cache(self) -> int:
        """Remove expired cache entries."""
        # Delete expired embedding cache
        result1 = await self.session.execute(
            delete(EmbeddingCache).where(
                EmbeddingCache.expires_at < datetime.utcnow()
            )
        )
        
        # Delete expired search cache
        result2 = await self.session.execute(
            delete(SearchCache).where(SearchCache.expires_at < datetime.utcnow())
        )
        
        await self.session.commit()
        
        total_deleted = result1.rowcount + result2.rowcount
        if total_deleted > 0:
            logger.info(f"Cleaned up {total_deleted} expired cache entries")
        
        return total_deleted