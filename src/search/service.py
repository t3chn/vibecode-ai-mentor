"""Search service integration and utilities for VibeCode AI Mentor."""

import logging
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from src.search.vector_search import VectorSearchService, CodeMatch, SearchFilters
from src.db.connection import get_async_session
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class SearchServiceManager:
    """High-level service manager for vector search operations."""
    
    def __init__(self):
        """Initialize search service manager."""
        self.settings = get_settings()
    
    async def create_search_service(self, session: Optional[AsyncSession] = None) -> VectorSearchService:
        """Create vector search service with database session.
        
        Args:
            session: Optional existing session, creates new one if not provided
            
        Returns:
            VectorSearchService instance
        """
        if session is None:
            session = await get_async_session()
        
        return VectorSearchService(session)
    
    async def quick_search(
        self,
        query: str,
        language: str = "python",
        limit: int = 10,
        similarity_threshold: Optional[float] = None,
        repository_filter: Optional[str] = None,
    ) -> List[Dict]:
        """Perform quick code search with minimal configuration.
        
        Args:
            query: Search query text
            language: Programming language filter
            limit: Maximum results to return
            similarity_threshold: Optional similarity threshold (0-1)
            repository_filter: Optional repository name filter
            
        Returns:
            List of search results as dictionaries
        """
        async with await get_async_session() as session:
            search_service = VectorSearchService(session)
            
            # Build filters
            filters = SearchFilters(language=language)
            
            # Convert similarity threshold to distance threshold
            distance_threshold = 1.0
            if similarity_threshold is not None:
                distance_threshold = 1.0 - similarity_threshold
            
            # Perform search
            results = await search_service.search_by_text(
                query_text=query,
                language=language,
                filters=filters,
                limit=limit,
            )
            
            # Filter by similarity threshold if provided
            if similarity_threshold is not None:
                results = [r for r in results if (1.0 - r.similarity_score) >= similarity_threshold]
            
            # Convert to dictionaries for API response
            return [result.to_dict() for result in results]
    
    async def find_similar_code(
        self,
        code_snippet: str,
        language: str = "python",
        threshold: float = 0.8,
        limit: int = 5,
        exclude_repository: Optional[str] = None,
    ) -> Dict:
        """Find similar code patterns and generate recommendations.
        
        Args:
            code_snippet: Code to find similarities for
            language: Programming language
            threshold: Similarity threshold
            limit: Maximum similar examples
            exclude_repository: Repository to exclude from results
            
        Returns:
            Dictionary with similar examples and recommendations
        """
        async with await get_async_session() as session:
            search_service = VectorSearchService(session)
            
            # Build filters
            filters = SearchFilters(language=language)
            if exclude_repository:
                filters.exclude_repositories = [exclude_repository]
            
            # Get similar examples and recommendations
            similar_examples, recommendations = await search_service.get_code_recommendations(
                code_snippet=code_snippet,
                language=language,
                limit=limit,
            )
            
            # Filter by threshold
            filtered_examples = [
                ex for ex in similar_examples 
                if (1.0 - ex.similarity_score) >= threshold
            ]
            
            return {
                "similar_examples": [ex.to_dict() for ex in filtered_examples],
                "recommendations": recommendations,
                "total_found": len(similar_examples),
                "threshold_filtered": len(filtered_examples),
            }
    
    async def detect_code_issues(
        self,
        code_snippet: str,
        language: str = "python",
        check_duplicates: bool = True,
        check_anti_patterns: bool = True,
    ) -> Dict:
        """Detect potential code issues using vector search.
        
        Args:
            code_snippet: Code to analyze
            language: Programming language
            check_duplicates: Whether to check for duplicates
            check_anti_patterns: Whether to check for anti-patterns
            
        Returns:
            Dictionary with detected issues
        """
        async with await get_async_session() as session:
            search_service = VectorSearchService(session)
            
            issues = {
                "duplicates": [],
                "anti_patterns": [],
                "complexity_issues": [],
                "recommendations": [],
            }
            
            # Check for similar/duplicate code
            if check_duplicates:
                similar_results = await search_service.search_similar_functions(
                    function_code=code_snippet,
                    threshold=0.9,  # High threshold for duplicates
                    language=language,
                    limit=10,
                )
                
                issues["duplicates"] = [
                    {
                        "file_path": result.file_path,
                        "repository_name": result.repository_name,
                        "similarity_score": 1.0 - result.similarity_score,  # Convert to similarity
                        "content_preview": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    }
                    for result in similar_results
                    if result.similarity_score < 0.1  # Very similar (low distance)
                ]
            
            # Check for anti-patterns
            if check_anti_patterns:
                anti_pattern_types = ["god_object", "long_method", "duplicate_code", "magic_numbers", "nested_loops"]
                
                for pattern_type in anti_pattern_types:
                    pattern_results = await search_service.search_anti_patterns(
                        pattern_type=pattern_type,
                        threshold=0.7,
                        language=language,
                        limit=5,
                    )
                    
                    if pattern_results:
                        issues["anti_patterns"].append({
                            "pattern_type": pattern_type,
                            "confidence": 1.0 - min(r.similarity_score for r in pattern_results),
                            "examples": len(pattern_results),
                            "description": self._get_anti_pattern_description(pattern_type),
                        })
            
            # Analyze complexity (if available in results)
            # This would be expanded with actual complexity analysis
            
            return issues
    
    async def search_repository_patterns(
        self,
        repository_id: UUID,
        pattern_query: str,
        language: Optional[str] = None,
        limit: int = 20,
    ) -> Dict:
        """Search for specific patterns within a repository.
        
        Args:
            repository_id: Repository to search within
            pattern_query: Pattern description or code example
            language: Optional language filter
            limit: Maximum results
            
        Returns:
            Dictionary with search results and statistics
        """
        async with await get_async_session() as session:
            search_service = VectorSearchService(session)
            
            # Build filters for repository-specific search
            filters = SearchFilters(
                repository_id=repository_id,
                language=language,
            )
            
            # Perform search
            results = await search_service.search_by_text(
                query_text=pattern_query,
                language=language or "python",
                filters=filters,
                limit=limit,
            )
            
            # Group results by file
            by_file = {}
            for result in results:
                file_path = result.file_path
                if file_path not in by_file:
                    by_file[file_path] = []
                by_file[file_path].append(result.to_dict())
            
            # Calculate statistics
            avg_similarity = sum(r.similarity_score for r in results) / len(results) if results else 0
            
            return {
                "results": [r.to_dict() for r in results],
                "results_by_file": by_file,
                "statistics": {
                    "total_matches": len(results),
                    "unique_files": len(by_file),
                    "average_similarity": 1.0 - avg_similarity,  # Convert to similarity score
                    "language_distribution": self._calculate_language_distribution(results),
                },
            }
    
    async def get_duplicate_report(
        self,
        repository_id: Optional[UUID] = None,
        similarity_threshold: float = 0.9,
        min_lines: int = 5,
        limit: int = 100,
    ) -> Dict:
        """Generate comprehensive duplicate code report.
        
        Args:
            repository_id: Optional repository filter
            similarity_threshold: Similarity threshold for duplicates
            min_lines: Minimum lines to consider
            limit: Maximum duplicate pairs
            
        Returns:
            Duplicate code report
        """
        async with await get_async_session() as session:
            search_service = VectorSearchService(session)
            
            # Find duplicate pairs
            duplicate_pairs = await search_service.find_duplicate_code(
                repository_id=repository_id,
                similarity_threshold=similarity_threshold,
                min_lines=min_lines,
                limit=limit,
            )
            
            # Process results
            report = {
                "duplicate_pairs": [],
                "statistics": {
                    "total_pairs": len(duplicate_pairs),
                    "files_affected": set(),
                    "repositories_affected": set(),
                    "languages_affected": set(),
                    "average_similarity": 0,
                },
            }
            
            total_similarity = 0
            for pair1, pair2 in duplicate_pairs:
                pair_data = {
                    "pair_id": f"{pair1.snippet_id}_{pair2.snippet_id}",
                    "similarity_score": 1.0 - pair1.similarity_score,  # Convert to similarity
                    "file1": {
                        "path": pair1.file_path,
                        "lines": f"{pair1.start_line}-{pair1.end_line}",
                        "content": pair1.content,
                    },
                    "file2": {
                        "path": pair2.file_path,
                        "lines": f"{pair2.start_line}-{pair2.end_line}",
                        "content": pair2.content,
                    },
                    "language": pair1.language,
                }
                
                report["duplicate_pairs"].append(pair_data)
                
                # Update statistics
                report["statistics"]["files_affected"].add(pair1.file_path)
                report["statistics"]["files_affected"].add(pair2.file_path)
                report["statistics"]["repositories_affected"].add(str(pair1.repository_id))
                report["statistics"]["languages_affected"].add(pair1.language)
                total_similarity += (1.0 - pair1.similarity_score)
            
            # Calculate averages
            if duplicate_pairs:
                report["statistics"]["average_similarity"] = total_similarity / len(duplicate_pairs)
            
            # Convert sets to lists for JSON serialization
            report["statistics"]["files_affected"] = list(report["statistics"]["files_affected"])
            report["statistics"]["repositories_affected"] = list(report["statistics"]["repositories_affected"])
            report["statistics"]["languages_affected"] = list(report["statistics"]["languages_affected"])
            
            return report
    
    async def get_service_health(self) -> Dict:
        """Get health status of search service components.
        
        Returns:
            Health status dictionary
        """
        try:
            async with await get_async_session() as session:
                search_service = VectorSearchService(session)
                
                # Get basic statistics
                stats = await search_service.get_search_stats()
                
                # Check embedding service health
                embedding_health = search_service.embedding_manager.health_check()
                
                # Test basic search functionality
                test_search_success = True
                try:
                    await search_service.search_by_text("test query", limit=1)
                except Exception as e:
                    logger.warning(f"Test search failed: {e}")
                    test_search_success = False
                
                return {
                    "status": "healthy" if test_search_success else "degraded",
                    "statistics": stats,
                    "embedding_service": embedding_health,
                    "test_search": test_search_success,
                    "timestamp": "2024-01-01T00:00:00Z",  # Would use actual timestamp
                }
                
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2024-01-01T00:00:00Z",
            }
    
    def _get_anti_pattern_description(self, pattern_type: str) -> str:
        """Get description for anti-pattern type."""
        descriptions = {
            "god_object": "Classes that are too large and handle too many responsibilities",
            "long_method": "Methods that are too long and complex, hard to understand and maintain",
            "duplicate_code": "Identical or very similar code blocks that should be refactored",
            "magic_numbers": "Hard-coded numeric values without clear meaning or constants",
            "nested_loops": "Deeply nested loops that may indicate algorithmic inefficiency",
        }
        return descriptions.get(pattern_type, f"Anti-pattern: {pattern_type}")
    
    def _calculate_language_distribution(self, results: List[CodeMatch]) -> Dict[str, int]:
        """Calculate language distribution in search results."""
        distribution = {}
        for result in results:
            language = result.language
            distribution[language] = distribution.get(language, 0) + 1
        return distribution


# Singleton instance for easy access
search_manager = SearchServiceManager()


# Convenience functions for common operations
async def quick_code_search(query: str, language: str = "python", limit: int = 10) -> List[Dict]:
    """Quick code search function."""
    return await search_manager.quick_search(query, language, limit)


async def find_code_similarities(code: str, language: str = "python", threshold: float = 0.8) -> Dict:
    """Find similar code patterns."""
    return await search_manager.find_similar_code(code, language, threshold)


async def detect_code_duplicates(repository_id: UUID, threshold: float = 0.9) -> Dict:
    """Detect duplicate code in repository."""
    return await search_manager.get_duplicate_report(repository_id, threshold)