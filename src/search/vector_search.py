"""Vector search functionality with TiDB integration for VibeCode AI Mentor."""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from uuid import UUID

from sqlalchemy import and_, desc, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.db.models import (
    AnalysisResult,
    CodeSnippet,
    Repository,
    SearchCache,
    EmbeddingCache,
)
from src.db.schema import (
    HYBRID_SEARCH_QUERY,
    VECTOR_SEARCH_QUERY,
    build_vector_search_filter,
)
from src.embeddings.factory import get_embedding_manager

logger = logging.getLogger(__name__)


class CodeMatch:
    """Represents a code similarity match result."""

    def __init__(
        self,
        snippet_id: UUID,
        repository_id: UUID,
        repository_name: str,
        file_path: str,
        language: str,
        content: str,
        start_line: int,
        end_line: int,
        similarity_score: float,
        complexity_score: Optional[float] = None,
        repository_url: Optional[str] = None,
    ):
        self.snippet_id = snippet_id
        self.repository_id = repository_id
        self.repository_name = repository_name
        self.file_path = file_path
        self.language = language
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.similarity_score = similarity_score
        self.complexity_score = complexity_score
        self.repository_url = repository_url

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "snippet_id": str(self.snippet_id),
            "repository_id": str(self.repository_id),
            "repository_name": self.repository_name,
            "file_path": self.file_path,
            "language": self.language,
            "content": self.content,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "similarity_score": self.similarity_score,
            "complexity_score": self.complexity_score,
            "repository_url": self.repository_url,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CodeMatch":
        """Create from dictionary."""
        return cls(
            snippet_id=UUID(data["snippet_id"]),
            repository_id=UUID(data["repository_id"]),
            repository_name=data["repository_name"],
            file_path=data["file_path"],
            language=data["language"],
            content=data["content"],
            start_line=data["start_line"],
            end_line=data["end_line"],
            similarity_score=data["similarity_score"],
            complexity_score=data.get("complexity_score"),
            repository_url=data.get("repository_url"),
        )


class SearchFilters:
    """Encapsulates search filters for better type safety."""

    def __init__(
        self,
        language: Optional[str] = None,
        repository_id: Optional[Union[str, UUID]] = None,
        min_complexity: Optional[float] = None,
        max_complexity: Optional[float] = None,
        file_extension: Optional[str] = None,
        exclude_repositories: Optional[List[Union[str, UUID]]] = None,
    ):
        self.language = language
        self.repository_id = str(repository_id) if repository_id else None
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.file_extension = file_extension
        self.exclude_repositories = (
            [str(r) for r in exclude_repositories] if exclude_repositories else None
        )


class VectorSearchService:
    """TiDB-powered vector search service for code similarity."""

    def __init__(self, session: AsyncSession):
        """Initialize with async database session."""
        self.session = session
        self.settings = get_settings()
        self.embedding_manager = get_embedding_manager()
        self._cache_ttl = timedelta(hours=1)

    async def search_similar_code(
        self,
        query_embedding: List[float],
        filters: Optional[SearchFilters] = None,
        limit: int = 10,
        similarity_threshold: float = 1.0,  # TiDB cosine distance: 0=identical, 2=opposite
        use_hybrid: bool = True,
    ) -> List[CodeMatch]:
        """Search for similar code using vector similarity.
        
        Args:
            query_embedding: Query vector embedding
            filters: Optional search filters
            limit: Maximum number of results
            similarity_threshold: Maximum distance threshold (lower = more similar)
            use_hybrid: Whether to use hybrid search with repository info
            
        Returns:
            List of CodeMatch objects ordered by similarity
        """
        filters = filters or SearchFilters()
        
        # Build query parameters
        params = {
            "query_embedding": json.dumps(query_embedding),
            "limit": limit,
        }
        
        # Build filter clause
        filter_clause = self._build_filter_clause(filters, params)
        
        # Choose query template
        query_template = HYBRID_SEARCH_QUERY if use_hybrid else VECTOR_SEARCH_QUERY
        query = query_template.format(filters=filter_clause)
        
        # Add similarity threshold filter
        if similarity_threshold < 1.0:
            if "WHERE" in query:
                query = query.replace(
                    "ORDER BY similarity",
                    f"HAVING similarity <= {similarity_threshold} ORDER BY similarity"
                )
        
        logger.debug(f"Executing vector search with {len(query_embedding)}D embedding")
        
        try:
            result = await self.session.execute(text(query), params)
            rows = result.fetchall()
            
            matches = []
            for row in rows:
                if use_hybrid:
                    match = CodeMatch(
                        snippet_id=UUID(row.id),
                        repository_id=UUID(row.repository_id),
                        repository_name=row.repository_name,
                        file_path=row.file_path,
                        language=row.language,
                        content=row.content,
                        start_line=row.start_line,
                        end_line=row.end_line,
                        similarity_score=float(row.similarity),
                        complexity_score=row.complexity_score,
                        repository_url=row.repository_url,
                    )
                else:
                    match = CodeMatch(
                        snippet_id=UUID(row.id),
                        repository_id=UUID(row.repository_id),
                        repository_name="",  # Not available in simple query
                        file_path=row.file_path,
                        language=row.language,
                        content=row.content,
                        start_line=row.start_line,
                        end_line=row.end_line,
                        similarity_score=float(row.similarity),
                        complexity_score=row.complexity_score,
                    )
                matches.append(match)
            
            logger.info(f"Found {len(matches)} similar code snippets")
            return matches
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

    async def search_by_text(
        self,
        query_text: str,
        language: str = "python",
        filters: Optional[SearchFilters] = None,
        limit: int = 10,
        use_cache: bool = True,
    ) -> List[CodeMatch]:
        """Search for similar code using text query.
        
        Args:
            query_text: Text to search for
            language: Programming language filter
            filters: Additional search filters
            limit: Maximum number of results
            use_cache: Whether to use cached results
            
        Returns:
            List of CodeMatch objects
        """
        # Create cache key
        cache_key = self._create_cache_key(query_text, language, filters, limit)
        
        # Try cache first
        if use_cache:
            cached_results = await self._get_cached_results(cache_key, language)
            if cached_results:
                logger.debug("Returning cached search results")
                return cached_results
        
        # Generate embedding for query text
        preprocessed_text = self.embedding_manager.preprocess_code(query_text)
        query_embedding = await self.embedding_manager.generate_embedding(preprocessed_text)
        
        # Set language filter if not already set
        if not filters:
            filters = SearchFilters(language=language)
        elif not filters.language:
            filters.language = language
        
        # Perform search
        results = await self.search_similar_code(
            query_embedding=query_embedding,
            filters=filters,
            limit=limit,
        )
        
        # Cache results
        if use_cache and results:
            await self._cache_results(cache_key, language, results)
        
        return results

    async def search_similar_functions(
        self,
        function_code: str,
        threshold: float = 0.8,
        language: Optional[str] = None,
        limit: int = 10,
    ) -> List[CodeMatch]:
        """Search for similar functions with high similarity threshold.
        
        Args:
            function_code: Function code to find similar examples
            threshold: Similarity threshold (0-1, higher = more similar)
            language: Programming language filter
            limit: Maximum number of results
            
        Returns:
            List of similar functions
        """
        # Convert similarity threshold to distance threshold
        distance_threshold = 1.0 - threshold
        
        # Auto-detect language if not provided
        if not language:
            language = self._detect_language(function_code)
        
        # Generate embedding
        preprocessed_code = self.embedding_manager.preprocess_code(function_code)
        query_embedding = await self.embedding_manager.generate_embedding(preprocessed_code)
        
        # Search with strict filters
        filters = SearchFilters(language=language)
        
        return await self.search_similar_code(
            query_embedding=query_embedding,
            filters=filters,
            limit=limit,
            similarity_threshold=distance_threshold,
        )

    async def search_anti_patterns(
        self,
        pattern_type: str,
        threshold: float = 0.7,
        language: str = "python",
        limit: int = 20,
    ) -> List[CodeMatch]:
        """Search for potential anti-patterns in code.
        
        Args:
            pattern_type: Type of anti-pattern to search for
            threshold: Similarity threshold
            language: Programming language
            limit: Maximum number of results
            
        Returns:
            List of potential anti-pattern matches
        """
        # Define anti-pattern templates
        anti_pattern_templates = {
            "god_object": "class LargeClass: def method1(): pass def method2(): pass def method3(): pass",
            "long_method": "def long_function(): if condition: if nested: if deep: pass pass pass",
            "duplicate_code": "if condition: do_something() else: do_something()",
            "magic_numbers": "if value > 42 and status == 3: return 999",
            "nested_loops": "for i in range(10): for j in range(10): for k in range(10): pass",
        }
        
        template = anti_pattern_templates.get(pattern_type)
        if not template:
            logger.warning(f"Unknown anti-pattern type: {pattern_type}")
            return []
        
        # Search for similar patterns
        return await self.search_by_text(
            query_text=template,
            language=language,
            limit=limit,
        )

    async def get_code_recommendations(
        self,
        code_snippet: str,
        language: str = "python",
        limit: int = 5,
    ) -> Tuple[List[CodeMatch], List[Dict]]:
        """Get code recommendations based on similar patterns.
        
        Args:
            code_snippet: Code to analyze
            language: Programming language
            limit: Number of similar examples to find
            
        Returns:
            Tuple of (similar_examples, recommendations)
        """
        # Find similar code
        similar_examples = await self.search_by_text(
            query_text=code_snippet,
            language=language,
            limit=limit,
        )
        
        # Get existing analysis results for similar code
        recommendations = []
        if similar_examples:
            snippet_ids = [match.snippet_id for match in similar_examples[:3]]
            
            result = await self.session.execute(
                text("""
                SELECT ar.recommendations, ar.quality_score, cs.content
                FROM analysis_results ar
                JOIN code_snippets cs ON ar.snippet_id = cs.id
                WHERE ar.snippet_id IN :snippet_ids
                ORDER BY ar.quality_score DESC
                """),
                {"snippet_ids": [str(sid) for sid in snippet_ids]}
            )
            
            for row in result.fetchall():
                if row.recommendations:
                    recommendations.extend(row.recommendations)
        
        return similar_examples, recommendations

    async def find_duplicate_code(
        self,
        repository_id: Optional[UUID] = None,
        similarity_threshold: float = 0.95,
        min_lines: int = 5,
        limit: int = 50,
    ) -> List[Tuple[CodeMatch, CodeMatch]]:
        """Find potential duplicate code within repositories.
        
        Args:
            repository_id: Optional repository to search within
            similarity_threshold: Similarity threshold for duplicates
            min_lines: Minimum lines for considering duplicates
            limit: Maximum pairs to return
            
        Returns:
            List of duplicate code pairs
        """
        distance_threshold = 1.0 - similarity_threshold
        
        # Query for finding similar pairs
        query = """
        SELECT 
            cs1.id as id1, cs1.file_path as path1, cs1.content as content1,
            cs1.start_line as start1, cs1.end_line as end1,
            cs2.id as id2, cs2.file_path as path2, cs2.content as content2,
            cs2.start_line as start2, cs2.end_line as end2,
            VEC_COSINE_DISTANCE(cs1.embedding, cs2.embedding) as similarity,
            cs1.repository_id, cs1.language
        FROM code_snippets cs1
        JOIN code_snippets cs2 ON cs1.id < cs2.id
        WHERE cs1.embedding IS NOT NULL 
            AND cs2.embedding IS NOT NULL
            AND cs1.language = cs2.language
            AND (cs1.end_line - cs1.start_line + 1) >= :min_lines
            AND (cs2.end_line - cs2.start_line + 1) >= :min_lines
            AND VEC_COSINE_DISTANCE(cs1.embedding, cs2.embedding) <= :threshold
            {repo_filter}
        ORDER BY similarity ASC
        LIMIT :limit
        """
        
        params = {
            "min_lines": min_lines,
            "threshold": distance_threshold,
            "limit": limit,
        }
        
        repo_filter = ""
        if repository_id:
            repo_filter = "AND cs1.repository_id = :repo_id AND cs2.repository_id = :repo_id"
            params["repo_id"] = str(repository_id)
        
        final_query = query.format(repo_filter=repo_filter)
        
        result = await self.session.execute(text(final_query), params)
        rows = result.fetchall()
        
        duplicate_pairs = []
        for row in rows:
            match1 = CodeMatch(
                snippet_id=UUID(row.id1),
                repository_id=UUID(row.repository_id),
                repository_name="",
                file_path=row.path1,
                language=row.language,
                content=row.content1,
                start_line=row.start1,
                end_line=row.end1,
                similarity_score=float(row.similarity),
            )
            
            match2 = CodeMatch(
                snippet_id=UUID(row.id2),
                repository_id=UUID(row.repository_id),
                repository_name="",
                file_path=row.path2,
                language=row.language,
                content=row.content2,
                start_line=row.start2,
                end_line=row.end2,
                similarity_score=float(row.similarity),
            )
            
            duplicate_pairs.append((match1, match2))
        
        logger.info(f"Found {len(duplicate_pairs)} potential duplicate pairs")
        return duplicate_pairs

    def _build_filter_clause(self, filters: SearchFilters, params: Dict) -> str:
        """Build SQL WHERE clause from filters."""
        clauses = []
        
        if filters.language:
            clauses.append("AND language = :language")
            params["language"] = filters.language
        
        if filters.repository_id:
            clauses.append("AND repository_id = :repository_id")
            params["repository_id"] = filters.repository_id
        
        if filters.min_complexity is not None:
            clauses.append("AND complexity_score >= :min_complexity")
            params["min_complexity"] = filters.min_complexity
        
        if filters.max_complexity is not None:
            clauses.append("AND complexity_score <= :max_complexity")
            params["max_complexity"] = filters.max_complexity
        
        if filters.file_extension:
            clauses.append("AND file_path LIKE :file_extension")
            params["file_extension"] = f"%.{filters.file_extension}"
        
        if filters.exclude_repositories:
            placeholders = [f":exclude_{i}" for i in range(len(filters.exclude_repositories))]
            clauses.append(f"AND repository_id NOT IN ({','.join(placeholders)})")
            for i, repo_id in enumerate(filters.exclude_repositories):
                params[f"exclude_{i}"] = repo_id
        
        return " ".join(clauses)

    def _create_cache_key(
        self,
        query_text: str,
        language: str,
        filters: Optional[SearchFilters],
        limit: int,
    ) -> str:
        """Create cache key for search results."""
        filter_str = ""
        if filters:
            filter_data = {
                "language": filters.language,
                "repository_id": filters.repository_id,
                "min_complexity": filters.min_complexity,
                "max_complexity": filters.max_complexity,
                "file_extension": filters.file_extension,
                "exclude_repositories": filters.exclude_repositories,
            }
            filter_str = json.dumps(filter_data, sort_keys=True)
        
        key_data = f"{query_text}:{language}:{filter_str}:{limit}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    async def _get_cached_results(
        self, cache_key: str, language: str
    ) -> Optional[List[CodeMatch]]:
        """Get cached search results."""
        try:
            result = await self.session.execute(
                text("""
                SELECT results FROM search_cache 
                WHERE query_hash = :cache_key 
                    AND language = :language 
                    AND expires_at > NOW()
                LIMIT 1
                """),
                {"cache_key": cache_key, "language": language}
            )
            
            row = result.fetchone()
            if row and row.results:
                return [CodeMatch.from_dict(item) for item in row.results]
                
        except Exception as e:
            logger.warning(f"Failed to retrieve cached results: {e}")
        
        return None

    async def _cache_results(
        self, cache_key: str, language: str, results: List[CodeMatch]
    ) -> None:
        """Cache search results."""
        try:
            expires_at = datetime.utcnow() + self._cache_ttl
            serialized_results = [match.to_dict() for match in results]
            
            # Use INSERT ... ON DUPLICATE KEY UPDATE for MySQL/TiDB
            await self.session.execute(
                text("""
                INSERT INTO search_cache (id, query_hash, language, results, expires_at)
                VALUES (UUID(), :cache_key, :language, :results, :expires_at)
                ON DUPLICATE KEY UPDATE 
                    results = VALUES(results),
                    expires_at = VALUES(expires_at)
                """),
                {
                    "cache_key": cache_key,
                    "language": language,
                    "results": json.dumps(serialized_results),
                    "expires_at": expires_at,
                }
            )
            
            await self.session.commit()
            logger.debug(f"Cached {len(results)} search results")
            
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")
            await self.session.rollback()

    def _detect_language(self, code: str) -> str:
        """Simple language detection based on syntax patterns."""
        code_lower = code.lower()
        
        if "def " in code or "import " in code or "class " in code:
            return "python"
        elif "function " in code or "const " in code or "let " in code:
            return "javascript"
        elif "public class" in code or "private " in code:
            return "java"
        elif "#include" in code or "int main" in code:
            return "cpp"
        else:
            return "python"  # Default fallback

    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            # Clean search cache
            result = await self.session.execute(
                text("DELETE FROM search_cache WHERE expires_at < NOW()")
            )
            search_deleted = result.rowcount
            
            # Clean embedding cache
            result = await self.session.execute(
                text("DELETE FROM embedding_cache WHERE expires_at < NOW()")
            )
            embedding_deleted = result.rowcount
            
            await self.session.commit()
            
            total_deleted = search_deleted + embedding_deleted
            logger.info(f"Cleaned up {total_deleted} expired cache entries")
            return total_deleted
            
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            await self.session.rollback()
            return 0

    async def get_search_stats(self) -> Dict:
        """Get search service statistics."""
        try:
            # Count total snippets with embeddings
            result = await self.session.execute(
                text("SELECT COUNT(*) as total FROM code_snippets WHERE embedding IS NOT NULL")
            )
            total_snippets = result.fetchone().total
            
            # Count by language
            result = await self.session.execute(
                text("""
                SELECT language, COUNT(*) as count 
                FROM code_snippets 
                WHERE embedding IS NOT NULL 
                GROUP BY language 
                ORDER BY count DESC
                """)
            )
            by_language = {row.language: row.count for row in result.fetchall()}
            
            # Cache stats
            result = await self.session.execute(
                text("SELECT COUNT(*) as cached FROM search_cache WHERE expires_at > NOW()")
            )
            cached_queries = result.fetchone().cached
            
            return {
                "total_indexed_snippets": total_snippets,
                "snippets_by_language": by_language,
                "cached_queries": cached_queries,
                "embedding_dimensions": self.embedding_manager.dimensions,
                "model_name": self.embedding_manager.model_name,
            }
            
        except Exception as e:
            logger.error(f"Failed to get search stats: {e}")
            return {}