"""Performance optimization utilities for vector search."""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Tuple
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.search.vector_search import VectorSearchService, CodeMatch, SearchFilters

logger = logging.getLogger(__name__)


class SearchOptimizer:
    """Optimizes vector search operations for better performance."""
    
    def __init__(self, search_service: VectorSearchService):
        """Initialize with vector search service."""
        self.search_service = search_service
        self.session = search_service.session
    
    async def batch_similarity_search(
        self,
        query_embeddings: List[List[float]],
        query_metadata: List[Dict],
        batch_size: int = 10,
        limit_per_query: int = 5,
    ) -> List[List[CodeMatch]]:
        """Perform batch similarity searches with optimized database usage.
        
        Args:
            query_embeddings: List of query embeddings
            query_metadata: List of metadata for each query (filters, etc.)
            batch_size: Number of queries to process in parallel
            limit_per_query: Results limit per query
            
        Returns:
            List of search results for each query
        """
        if len(query_embeddings) != len(query_metadata):
            raise ValueError("Embeddings and metadata lists must have same length")
        
        results = []
        
        # Process in batches to avoid overwhelming the database
        for i in range(0, len(query_embeddings), batch_size):
            batch_embeddings = query_embeddings[i:i + batch_size]
            batch_metadata = query_metadata[i:i + batch_size]
            
            # Create tasks for parallel execution
            tasks = []
            for embedding, metadata in zip(batch_embeddings, batch_metadata):
                filters = SearchFilters(**metadata.get("filters", {}))
                task = self.search_service.search_similar_code(
                    query_embedding=embedding,
                    filters=filters,
                    limit=limit_per_query,
                )
                tasks.append(task)
            
            # Execute batch in parallel
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions and collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Batch search failed: {result}")
                    results.append([])  # Empty result for failed query
                else:
                    results.append(result)
        
        logger.info(f"Completed batch search for {len(query_embeddings)} queries")
        return results
    
    async def precompute_similarity_matrix(
        self,
        repository_id: UUID,
        languages: Optional[List[str]] = None,
        chunk_size: int = 100,
    ) -> Dict[str, Dict[str, float]]:
        """Precompute similarity matrix for code snippets in a repository.
        
        This is useful for applications that need to frequently compute
        similarities between code snippets in the same repository.
        
        Args:
            repository_id: Repository to process
            languages: Optional language filter
            chunk_size: Number of snippets to process per chunk
            
        Returns:
            Dictionary mapping snippet_id pairs to similarity scores
        """
        # Get all snippets from repository
        language_filter = ""
        params = {"repository_id": str(repository_id)}
        
        if languages:
            placeholders = [f":lang_{i}" for i in range(len(languages))]
            language_filter = f"AND language IN ({','.join(placeholders)})"
            for i, lang in enumerate(languages):
                params[f"lang_{i}"] = lang
        
        query = f"""
        SELECT id, embedding, language, file_path
        FROM code_snippets 
        WHERE repository_id = :repository_id 
            AND embedding IS NOT NULL 
            {language_filter}
        ORDER BY id
        """
        
        result = await self.session.execute(text(query), params)
        snippets = result.fetchall()
        
        if len(snippets) < 2:
            logger.warning(f"Not enough snippets ({len(snippets)}) for similarity matrix")
            return {}
        
        logger.info(f"Computing similarity matrix for {len(snippets)} snippets")
        
        # Precompute similarities in chunks to manage memory
        similarity_matrix = {}
        
        for i in range(0, len(snippets), chunk_size):
            chunk = snippets[i:i + chunk_size]
            
            # Compute similarities for this chunk against all snippets
            for snippet1 in chunk:
                snippet1_id = str(snippet1.id)
                similarity_matrix[snippet1_id] = {}
                
                # Parse embedding
                if isinstance(snippet1.embedding, str):
                    embedding1 = [float(x) for x in snippet1.embedding.strip("[]").split(",")]
                else:
                    embedding1 = snippet1.embedding
                
                # Use database to compute cosine distances efficiently
                distance_query = """
                SELECT id, VEC_COSINE_DISTANCE(:embedding1, embedding) as distance
                FROM code_snippets
                WHERE repository_id = :repository_id 
                    AND embedding IS NOT NULL
                    AND id != :snippet1_id
                """
                
                distance_result = await self.session.execute(
                    text(distance_query),
                    {
                        "embedding1": f"[{','.join(map(str, embedding1))}]",
                        "repository_id": str(repository_id),
                        "snippet1_id": snippet1_id,
                    }
                )
                
                for row in distance_result.fetchall():
                    snippet2_id = str(row.id)
                    similarity_score = 1.0 - float(row.distance)  # Convert distance to similarity
                    similarity_matrix[snippet1_id][snippet2_id] = similarity_score
            
            logger.debug(f"Processed chunk {i//chunk_size + 1}/{(len(snippets) + chunk_size - 1)//chunk_size}")
        
        return similarity_matrix
    
    async def optimize_vector_indexes(self) -> Dict[str, str]:
        """Optimize TiDB vector indexes for better search performance.
        
        Returns:
            Dictionary with optimization results
        """
        optimization_results = {}
        
        try:
            # Check current index statistics
            index_stats_query = """
            SELECT 
                TABLE_NAME,
                INDEX_NAME,
                CARDINALITY,
                INDEX_TYPE
            FROM information_schema.STATISTICS 
            WHERE TABLE_SCHEMA = DATABASE()
                AND INDEX_NAME LIKE '%embedding%'
            """
            
            result = await self.session.execute(text(index_stats_query))
            current_indexes = result.fetchall()
            
            optimization_results["current_indexes"] = [
                {
                    "table": row.TABLE_NAME,
                    "index": row.INDEX_NAME,
                    "cardinality": row.CARDINALITY,
                    "type": row.INDEX_TYPE,
                }
                for row in current_indexes
            ]
            
            # Analyze query performance
            analyze_queries = [
                "ANALYZE TABLE code_snippets",
                "ANALYZE TABLE embedding_cache",
            ]
            
            for query in analyze_queries:
                await self.session.execute(text(query))
                logger.info(f"Analyzed table: {query}")
            
            optimization_results["status"] = "completed"
            optimization_results["message"] = "Index optimization completed successfully"
            
        except Exception as e:
            logger.error(f"Index optimization failed: {e}")
            optimization_results["status"] = "failed"
            optimization_results["error"] = str(e)
        
        return optimization_results
    
    async def get_search_performance_metrics(self) -> Dict:
        """Get performance metrics for search operations.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {}
        
        try:
            # Query execution time analysis
            query_performance = """
            SELECT 
                COUNT(*) as total_queries,
                AVG(QUERY_TIME) as avg_query_time,
                MAX(QUERY_TIME) as max_query_time,
                MIN(QUERY_TIME) as min_query_time
            FROM information_schema.PROCESSLIST 
            WHERE COMMAND = 'Query' 
                AND INFO LIKE '%VEC_COSINE_DISTANCE%'
            """
            
            # Vector index usage
            index_usage = """
            SELECT 
                cs.language,
                COUNT(*) as snippet_count,
                COUNT(cs.embedding) as indexed_count,
                (COUNT(cs.embedding) * 100.0 / COUNT(*)) as index_coverage
            FROM code_snippets cs
            GROUP BY cs.language
            ORDER BY snippet_count DESC
            """
            
            # Cache hit rates
            cache_stats = """
            SELECT 
                'search_cache' as cache_type,
                COUNT(*) as total_entries,
                SUM(CASE WHEN expires_at > NOW() THEN 1 ELSE 0 END) as valid_entries
            FROM search_cache
            UNION ALL
            SELECT 
                'embedding_cache' as cache_type,
                COUNT(*) as total_entries,
                SUM(CASE WHEN expires_at > NOW() THEN 1 ELSE 0 END) as valid_entries
            FROM embedding_cache
            """
            
            # Execute queries
            index_result = await self.session.execute(text(index_usage))
            cache_result = await self.session.execute(text(cache_stats))
            
            metrics["index_coverage"] = [
                {
                    "language": row.language,
                    "total_snippets": row.snippet_count,
                    "indexed_snippets": row.indexed_count,
                    "coverage_percent": float(row.index_coverage),
                }
                for row in index_result.fetchall()
            ]
            
            metrics["cache_statistics"] = [
                {
                    "cache_type": row.cache_type,
                    "total_entries": row.total_entries,
                    "valid_entries": row.valid_entries,
                    "hit_rate": (row.valid_entries / row.total_entries * 100) if row.total_entries > 0 else 0,
                }
                for row in cache_result.fetchall()
            ]
            
            # Database connection stats
            connection_stats = """
            SELECT 
                COUNT(*) as active_connections,
                SUM(CASE WHEN COMMAND = 'Sleep' THEN 1 ELSE 0 END) as idle_connections
            FROM information_schema.PROCESSLIST
            """
            
            conn_result = await self.session.execute(text(connection_stats))
            conn_row = conn_result.fetchone()
            
            metrics["connection_stats"] = {
                "active_connections": conn_row.active_connections,
                "idle_connections": conn_row.idle_connections,
            }
            
            metrics["timestamp"] = "2024-01-01T00:00:00Z"  # Would use actual timestamp
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def suggest_optimizations(self, repository_id: Optional[UUID] = None) -> List[Dict]:
        """Suggest performance optimizations based on current usage patterns.
        
        Args:
            repository_id: Optional repository to analyze
            
        Returns:
            List of optimization suggestions
        """
        suggestions = []
        
        try:
            # Analyze embedding coverage
            coverage_query = """
            SELECT 
                language,
                COUNT(*) as total,
                COUNT(embedding) as with_embedding,
                AVG(complexity_score) as avg_complexity
            FROM code_snippets
            """
            
            params = {}
            if repository_id:
                coverage_query += " WHERE repository_id = :repo_id"
                params["repo_id"] = str(repository_id)
            
            coverage_query += " GROUP BY language"
            
            result = await self.session.execute(text(coverage_query), params)
            coverage_data = result.fetchall()
            
            for row in coverage_data:
                coverage_percent = (row.with_embedding / row.total * 100) if row.total > 0 else 0
                
                if coverage_percent < 90:
                    suggestions.append({
                        "type": "embedding_coverage",
                        "priority": "high" if coverage_percent < 50 else "medium",
                        "language": row.language,
                        "message": f"Only {coverage_percent:.1f}% of {row.language} snippets have embeddings",
                        "recommendation": "Run embedding generation for missing snippets",
                        "impact": "Improved search accuracy and coverage",
                    })
            
            # Analyze cache efficiency
            cache_query = """
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN expires_at > NOW() THEN 1 ELSE 0 END) as valid
            FROM search_cache
            """
            
            cache_result = await self.session.execute(text(cache_query))
            cache_row = cache_result.fetchone()
            
            if cache_row.total > 0:
                cache_hit_rate = (cache_row.valid / cache_row.total * 100)
                if cache_hit_rate < 60:
                    suggestions.append({
                        "type": "cache_optimization",
                        "priority": "medium",
                        "message": f"Cache hit rate is {cache_hit_rate:.1f}%",
                        "recommendation": "Consider increasing cache TTL or warming cache",
                        "impact": "Reduced query latency and API cost",
                    })
            
            # Analyze duplicate detection opportunity
            duplicate_query = """
            SELECT 
                COUNT(*) as total_pairs,
                AVG(VEC_COSINE_DISTANCE(cs1.embedding, cs2.embedding)) as avg_distance
            FROM code_snippets cs1
            JOIN code_snippets cs2 ON cs1.id < cs2.id
            WHERE cs1.embedding IS NOT NULL 
                AND cs2.embedding IS NOT NULL
                AND cs1.language = cs2.language
            """
            
            if repository_id:
                duplicate_query += " AND cs1.repository_id = :repo_id AND cs2.repository_id = :repo_id"
            
            duplicate_query += " LIMIT 1000"  # Limit for performance
            
            dup_result = await self.session.execute(text(duplicate_query), params)
            dup_row = dup_result.fetchone()
            
            if dup_row.total_pairs > 0 and dup_row.avg_distance < 0.2:
                suggestions.append({
                    "type": "duplicate_detection",
                    "priority": "medium",
                    "message": f"High similarity detected in {dup_row.total_pairs} code pairs",
                    "recommendation": "Run duplicate detection analysis",
                    "impact": "Identify refactoring opportunities",
                })
            
            # Index optimization suggestion
            index_query = """
            SELECT COUNT(*) as unindexed
            FROM code_snippets 
            WHERE embedding IS NULL
            """
            
            if repository_id:
                index_query += " AND repository_id = :repo_id"
            
            index_result = await self.session.execute(text(index_query), params)
            unindexed_count = index_result.fetchone().unindexed
            
            if unindexed_count > 100:
                suggestions.append({
                    "type": "indexing",
                    "priority": "high",
                    "message": f"{unindexed_count} code snippets without embeddings",
                    "recommendation": "Generate embeddings for all code snippets",
                    "impact": "Enable vector search for all code",
                })
            
        except Exception as e:
            logger.error(f"Failed to generate optimization suggestions: {e}")
            suggestions.append({
                "type": "error",
                "priority": "high",
                "message": f"Failed to analyze: {str(e)}",
                "recommendation": "Check database connectivity and permissions",
                "impact": "Analysis functionality unavailable",
            })
        
        return suggestions


class QueryPlanOptimizer:
    """Optimizes TiDB query execution plans for vector searches."""
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session
    
    async def analyze_query_plan(self, query: str, params: Dict) -> Dict:
        """Analyze execution plan for a vector search query.
        
        Args:
            query: SQL query to analyze
            params: Query parameters
            
        Returns:
            Dictionary with execution plan analysis
        """
        try:
            # Get execution plan
            explain_query = f"EXPLAIN FORMAT=JSON {query}"
            result = await self.session.execute(text(explain_query), params)
            plan_json = result.fetchone()[0]
            
            # Analyze plan for optimization opportunities
            analysis = {
                "query": query,
                "execution_plan": plan_json,
                "optimizations": [],
                "estimated_cost": None,
                "index_usage": [],
            }
            
            # Simple analysis - would be expanded in production
            if "vector_index" not in plan_json.lower():
                analysis["optimizations"].append({
                    "type": "index_usage",
                    "message": "Vector index may not be used efficiently",
                    "suggestion": "Check index hints or query structure",
                })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Query plan analysis failed: {e}")
            return {"error": str(e)}
    
    async def suggest_index_hints(self, table_name: str, search_filters: SearchFilters) -> List[str]:
        """Suggest index hints for optimized query execution.
        
        Args:
            table_name: Target table name
            search_filters: Search filters being applied
            
        Returns:
            List of suggested index hints
        """
        hints = []
        
        # Vector search hint
        hints.append(f"USE INDEX FOR ORDER BY ({table_name}_embedding_idx)")
        
        # Filter-specific hints
        if search_filters.language:
            hints.append(f"USE INDEX ({table_name}_language_idx)")
        
        if search_filters.repository_id:
            hints.append(f"USE INDEX ({table_name}_repository_idx)")
        
        return hints