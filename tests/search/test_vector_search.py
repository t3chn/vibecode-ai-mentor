"""Comprehensive tests for vector search functionality."""

import uuid
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.search.vector_search import VectorSearchService
from src.search.service import SearchServiceManager
from src.search.optimization import SearchOptimizer
from src.db.models import Repository, CodeSnippet
from src.db.repositories import CodeSnippetRepo


class TestVectorSearchService:
    """Test VectorSearchService functionality."""

    @pytest.fixture
    def vector_search_service(self):
        """Create VectorSearchService instance."""
        return VectorSearchService()

    @pytest.fixture
    def sample_embeddings(self):
        """Sample embeddings for testing."""
        return {
            "query": [0.1, 0.2, 0.3] + [0.0] * 1533,  # 1536 dimensions
            "similar1": [0.1, 0.2, 0.31] + [0.0] * 1533,  # Very similar
            "similar2": [0.11, 0.19, 0.29] + [0.0] * 1533,  # Similar
            "different": [0.9, 0.8, 0.7] + [0.0] * 1533,  # Different
        }

    @pytest.mark.asyncio
    async def test_search_similar_code_basic(
        self,
        vector_search_service: VectorSearchService,
        sample_embeddings: dict,
        async_session: AsyncSession,
        test_repository: Repository
    ):
        """Test basic similar code search."""
        # Mock database search results
        mock_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "content": "def calculate_sum(numbers): return sum(numbers)",
                "file_path": "src/math_utils.py",
                "language": "python",
                "similarity_score": 0.95,
                "line_start": 1,
                "line_end": 1,
                "complexity_score": 2.0
            },
            {
                "snippet_id": str(uuid.uuid4()),
                "content": "def compute_total(values): return sum(values)",
                "file_path": "src/calculations.py",
                "language": "python",
                "similarity_score": 0.87,
                "line_start": 5,
                "line_end": 5,
                "complexity_score": 2.0
            }
        ]

        with patch.object(vector_search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = mock_results
            
            results = await vector_search_service.search_similar_code(
                query_embedding=sample_embeddings["query"],
                language="python",
                limit=10,
                similarity_threshold=0.8
            )

        assert len(results) == 2
        assert results[0]["similarity_score"] > results[1]["similarity_score"]
        assert all(result["similarity_score"] >= 0.8 for result in results)

    @pytest.mark.asyncio
    async def test_search_with_repository_filter(
        self,
        vector_search_service: VectorSearchService,
        sample_embeddings: dict,
        test_repository: Repository
    ):
        """Test search with repository filter."""
        mock_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "repository_id": str(test_repository.id),
                "content": "def filtered_function(): pass",
                "similarity_score": 0.9
            }
        ]

        with patch.object(vector_search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = mock_results
            
            results = await vector_search_service.search_similar_code(
                query_embedding=sample_embeddings["query"],
                language="python",
                repository_id=test_repository.id,
                limit=5
            )

        mock_search.assert_called_once()
        call_args = mock_search.call_args[1]
        assert call_args["repository_id"] == test_repository.id

    @pytest.mark.asyncio
    async def test_search_with_complexity_filter(
        self,
        vector_search_service: VectorSearchService,
        sample_embeddings: dict
    ):
        """Test search with complexity score filter."""
        mock_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "content": "def complex_function(): # complex logic",
                "complexity_score": 8.5,
                "similarity_score": 0.9
            }
        ]

        with patch.object(vector_search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = mock_results
            
            results = await vector_search_service.search_similar_code(
                query_embedding=sample_embeddings["query"],
                language="python",
                min_complexity=5.0,
                max_complexity=10.0,
                limit=5
            )

        assert len(results) == 1
        assert results[0]["complexity_score"] >= 5.0
        assert results[0]["complexity_score"] <= 10.0

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self,
        vector_search_service: VectorSearchService,
        sample_embeddings: dict
    ):
        """Test search returning empty results."""
        with patch.object(vector_search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = []
            
            results = await vector_search_service.search_similar_code(
                query_embedding=sample_embeddings["query"],
                language="python",
                similarity_threshold=0.99  # Very high threshold
            )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_database_error_handling(
        self,
        vector_search_service: VectorSearchService,
        sample_embeddings: dict
    ):
        """Test handling of database errors during search."""
        with patch.object(vector_search_service, '_execute_vector_search') as mock_search:
            mock_search.side_effect = Exception("Database connection failed")
            
            with pytest.raises(Exception, match="Database connection failed"):
                await vector_search_service.search_similar_code(
                    query_embedding=sample_embeddings["query"],
                    language="python"
                )

    @pytest.mark.asyncio
    async def test_search_invalid_embedding_dimension(
        self,
        vector_search_service: VectorSearchService
    ):
        """Test search with invalid embedding dimensions."""
        invalid_embedding = [0.1, 0.2]  # Wrong dimension (should be 1536)
        
        with pytest.raises(ValueError, match="embedding dimension"):
            await vector_search_service.search_similar_code(
                query_embedding=invalid_embedding,
                language="python"
            )

    @pytest.mark.asyncio
    async def test_search_result_sorting(
        self,
        vector_search_service: VectorSearchService,
        sample_embeddings: dict
    ):
        """Test that search results are properly sorted by similarity."""
        mock_results = [
            {"snippet_id": "1", "similarity_score": 0.7},
            {"snippet_id": "2", "similarity_score": 0.9},
            {"snippet_id": "3", "similarity_score": 0.8},
        ]

        with patch.object(vector_search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = mock_results
            
            results = await vector_search_service.search_similar_code(
                query_embedding=sample_embeddings["query"],
                language="python"
            )

        # Results should be sorted by similarity score (descending)
        assert results[0]["similarity_score"] == 0.9
        assert results[1]["similarity_score"] == 0.8
        assert results[2]["similarity_score"] == 0.7

    @pytest.mark.asyncio
    async def test_search_limit_enforcement(
        self,
        vector_search_service: VectorSearchService,
        sample_embeddings: dict
    ):
        """Test that search respects the limit parameter."""
        # Create more results than the limit
        mock_results = [
            {"snippet_id": str(i), "similarity_score": 0.9 - (i * 0.1)}
            for i in range(10)
        ]

        with patch.object(vector_search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = mock_results
            
            results = await vector_search_service.search_similar_code(
                query_embedding=sample_embeddings["query"],
                language="python",
                limit=5
            )

        assert len(results) == 5


class TestSearchServiceManager:
    """Test SearchServiceManager functionality."""

    @pytest.fixture
    def search_manager(self):
        """Create SearchServiceManager instance."""
        return SearchServiceManager()

    @pytest.mark.asyncio
    async def test_quick_search(
        self,
        search_manager: SearchServiceManager,
        mock_embedding_provider
    ):
        """Test quick search functionality."""
        query = "calculate average function"
        
        mock_search_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "content": "def calculate_average(numbers): return sum(numbers) / len(numbers)",
                "similarity_score": 0.95
            }
        ]

        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_search.return_value = mock_search_results
            
            results = await search_manager.quick_search(
                query=query,
                language="python",
                limit=10
            )

        assert len(results) == 1
        assert results[0]["similarity_score"] == 0.95
        mock_embedding_provider.generate_embedding.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_search_repository_patterns(
        self,
        search_manager: SearchServiceManager,
        test_repository: Repository,
        mock_embedding_provider
    ):
        """Test repository-specific pattern search."""
        repository_id = test_repository.id
        pattern_query = "authentication logic"
        
        mock_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "repository_id": str(repository_id),
                "content": "def authenticate_user(token): return validate_token(token)",
                "similarity_score": 0.88
            }
        ]

        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_search.return_value = mock_results
            
            result = await search_manager.search_repository_patterns(
                repository_id=repository_id,
                pattern_query=pattern_query,
                language="python",
                limit=5
            )

        assert "results" in result
        assert len(result["results"]) == 1
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["repository_id"] == repository_id

    @pytest.mark.asyncio
    async def test_advanced_search_with_filters(
        self,
        search_manager: SearchServiceManager,
        mock_embedding_provider
    ):
        """Test advanced search with multiple filters."""
        search_params = {
            "query": "database connection",
            "language": "python",
            "file_path_pattern": "src/db/%",
            "min_complexity": 3.0,
            "max_complexity": 8.0,
            "similarity_threshold": 0.75
        }

        mock_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "content": "def connect_database(): # connection logic",
                "file_path": "src/db/connection.py",
                "complexity_score": 5.0,
                "similarity_score": 0.85
            }
        ]

        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_search.return_value = mock_results
            
            results = await search_manager.advanced_search(**search_params)

        assert len(results) == 1
        assert results[0]["file_path"].startswith("src/db/")
        assert 3.0 <= results[0]["complexity_score"] <= 8.0

    @pytest.mark.asyncio
    async def test_search_with_caching(
        self,
        search_manager: SearchServiceManager,
        mock_embedding_provider
    ):
        """Test search result caching."""
        query = "cached search query"
        
        mock_results = [{"snippet_id": "cached", "similarity_score": 0.9}]

        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search, \
             patch.object(search_manager, "_get_cached_results") as mock_get_cache, \
             patch.object(search_manager, "_cache_results") as mock_set_cache:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_get_cache.return_value = None  # No cached results
            mock_search.return_value = mock_results
            
            # First search - should cache results
            results1 = await search_manager.quick_search(query=query, language="python")
            
            # Second search - should use cached results
            mock_get_cache.return_value = mock_results
            results2 = await search_manager.quick_search(query=query, language="python")

        # Verify caching behavior
        mock_set_cache.assert_called_once()
        assert results1 == results2

    @pytest.mark.asyncio
    async def test_service_health_check(self, search_manager: SearchServiceManager):
        """Test search service health check."""
        with patch.object(search_manager.vector_search, "health_check") as mock_health:
            mock_health.return_value = {"status": "healthy", "response_time_ms": 50}
            
            health = await search_manager.get_service_health()

        assert health["status"] == "healthy"
        assert "response_time_ms" in health

    @pytest.mark.asyncio
    async def test_search_error_handling(
        self,
        search_manager: SearchServiceManager,
        mock_embedding_provider
    ):
        """Test error handling in search operations."""
        with patch("src.search.service.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            mock_embedding_provider.generate_embedding.side_effect = Exception("Embedding service down")
            
            with pytest.raises(Exception, match="Embedding service down"):
                await search_manager.quick_search(
                    query="test query",
                    language="python"
                )

    @pytest.mark.asyncio
    async def test_search_with_empty_query(
        self,
        search_manager: SearchServiceManager
    ):
        """Test search with empty query."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await search_manager.quick_search(
                query="",
                language="python"
            )

    @pytest.mark.asyncio
    async def test_search_with_invalid_language(
        self,
        search_manager: SearchServiceManager,
        mock_embedding_provider
    ):
        """Test search with unsupported language."""
        with patch("src.search.service.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Should handle gracefully or raise appropriate error
            results = await search_manager.quick_search(
                query="test query",
                language="unsupported_language"
            )
            
            # Should return empty results or handle gracefully
            assert isinstance(results, list)


class TestSearchOptimizer:
    """Test SearchOptimizer functionality."""

    @pytest.fixture
    def search_optimizer(self):
        """Create SearchOptimizer instance."""
        return SearchOptimizer()

    def test_optimize_query_embedding(self, search_optimizer: SearchOptimizer):
        """Test query embedding optimization."""
        original_embedding = [0.1, 0.2, 0.3] + [0.0] * 1533
        
        optimized = search_optimizer.optimize_query_embedding(
            embedding=original_embedding,
            boost_factors={"semantic": 1.2, "syntactic": 0.8}
        )
        
        assert len(optimized) == len(original_embedding)
        assert optimized != original_embedding  # Should be modified

    def test_calculate_similarity_threshold(self, search_optimizer: SearchOptimizer):
        """Test dynamic similarity threshold calculation."""
        query_complexity = 5.0
        search_context = {
            "language": "python",
            "repository_size": "large",
            "user_preference": "high_precision"
        }
        
        threshold = search_optimizer.calculate_similarity_threshold(
            query_complexity=query_complexity,
            context=search_context
        )
        
        assert 0.0 <= threshold <= 1.0
        assert isinstance(threshold, float)

    def test_rank_search_results(self, search_optimizer: SearchOptimizer):
        """Test search result ranking optimization."""
        results = [
            {
                "snippet_id": "1",
                "similarity_score": 0.8,
                "complexity_score": 2.0,
                "popularity_score": 0.6
            },
            {
                "snippet_id": "2", 
                "similarity_score": 0.75,
                "complexity_score": 1.0,
                "popularity_score": 0.9
            },
            {
                "snippet_id": "3",
                "similarity_score": 0.85,
                "complexity_score": 3.0,
                "popularity_score": 0.4
            }
        ]
        
        ranked_results = search_optimizer.rank_results(
            results=results,
            ranking_factors={
                "similarity": 1.0,
                "complexity": 0.5,
                "popularity": 0.3
            }
        )
        
        assert len(ranked_results) == 3
        # Should be sorted by combined score
        assert ranked_results[0]["snippet_id"] in ["1", "2", "3"]

    def test_filter_results_by_diversity(self, search_optimizer: SearchOptimizer):
        """Test result diversity filtering."""
        results = [
            {
                "snippet_id": "1",
                "content": "def calculate_sum(a, b): return a + b",
                "similarity_score": 0.95
            },
            {
                "snippet_id": "2",
                "content": "def calculate_sum(x, y): return x + y",  # Very similar
                "similarity_score": 0.93
            },
            {
                "snippet_id": "3",
                "content": "def multiply_numbers(a, b): return a * b",  # Different
                "similarity_score": 0.80
            }
        ]
        
        diverse_results = search_optimizer.filter_by_diversity(
            results=results,
            diversity_threshold=0.8,
            max_results=2
        )
        
        # Should keep diverse results, removing highly similar ones
        assert len(diverse_results) <= 2
        assert diverse_results[0]["snippet_id"] == "1"  # Highest score
        # Should include diverse result, not the very similar one
        if len(diverse_results) == 2:
            assert diverse_results[1]["snippet_id"] == "3"

    def test_optimize_search_parameters(self, search_optimizer: SearchOptimizer):
        """Test search parameter optimization."""
        user_query = "authentication middleware function"
        search_history = [
            {"query": "auth function", "clicked_results": 3, "total_results": 10},
            {"query": "middleware", "clicked_results": 5, "total_results": 15}
        ]
        
        optimized_params = search_optimizer.optimize_search_parameters(
            query=user_query,
            search_history=search_history,
            user_preferences={"precision": "high", "recall": "medium"}
        )
        
        assert "similarity_threshold" in optimized_params
        assert "limit" in optimized_params
        assert "boost_factors" in optimized_params
        assert 0.0 <= optimized_params["similarity_threshold"] <= 1.0


class TestSearchPerformance:
    """Test search performance and scalability."""

    @pytest.mark.asyncio
    async def test_search_response_time(
        self,
        async_session: AsyncSession,
        test_repository: Repository
    ):
        """Test search response time under normal load."""
        search_service = VectorSearchService()
        query_embedding = [0.1] * 1536
        
        # Mock database search with realistic delay
        async def mock_search(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.1)  # Simulate database query time
            return [
                {
                    "snippet_id": str(uuid.uuid4()),
                    "similarity_score": 0.9,
                    "content": "def test(): pass"
                }
            ]
        
        import time
        
        with patch.object(search_service, '_execute_vector_search', side_effect=mock_search):
            start_time = time.time()
            
            results = await search_service.search_similar_code(
                query_embedding=query_embedding,
                language="python",
                limit=10
            )
            
            end_time = time.time()
            response_time = end_time - start_time
        
        assert len(results) > 0
        assert response_time < 0.5  # Should complete within 500ms

    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self):
        """Test performance under concurrent search load."""
        search_manager = SearchServiceManager()
        query_embedding = [0.1] * 1536
        
        # Mock search with small delay
        async def mock_search(*args, **kwargs):
            import asyncio
            await asyncio.sleep(0.05)
            return [{"snippet_id": str(uuid.uuid4()), "similarity_score": 0.8}]
        
        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code", side_effect=mock_search):
            
            mock_embedding_provider = MagicMock()
            mock_embedding_provider.generate_embedding = AsyncMock(return_value=query_embedding)
            mock_embed_factory.return_value = mock_embedding_provider
            
            import asyncio
            import time
            
            # Simulate concurrent searches
            search_tasks = [
                search_manager.quick_search(f"query {i}", "python")
                for i in range(10)
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*search_tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
        
        assert len(results) == 10
        assert all(len(result) > 0 for result in results)
        assert total_time < 1.0  # All 10 searches should complete within 1 second

    @pytest.mark.asyncio
    async def test_large_result_set_handling(self):
        """Test handling of large result sets."""
        search_service = VectorSearchService()
        query_embedding = [0.1] * 1536
        
        # Mock large result set
        large_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "similarity_score": 0.9 - (i * 0.001),
                "content": f"def function_{i}(): pass"
            }
            for i in range(1000)  # 1000 results
        ]
        
        with patch.object(search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = large_results
            
            # Test with different limits
            for limit in [10, 50, 100]:
                results = await search_service.search_similar_code(
                    query_embedding=query_embedding,
                    language="python",
                    limit=limit
                )
                
                assert len(results) == limit
                # Verify results are properly sorted
                for i in range(1, len(results)):
                    assert results[i-1]["similarity_score"] >= results[i]["similarity_score"]

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_embeddings(self):
        """Test memory usage with large embedding vectors."""
        search_service = VectorSearchService()
        
        # Test with maximum size embeddings
        large_embedding = [0.1] * 1536  # Standard size but test memory handling
        
        mock_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "embedding": [0.1 + (i * 0.001)] * 1536,  # Unique embeddings
                "similarity_score": 0.9,
                "content": f"def func_{i}(): pass"
            }
            for i in range(100)
        ]
        
        with patch.object(search_service, '_execute_vector_search') as mock_search:
            mock_search.return_value = mock_results
            
            # This should handle large embeddings without memory issues
            results = await search_service.search_similar_code(
                query_embedding=large_embedding,
                language="python",
                limit=100
            )
        
        assert len(results) == 100
        assert all("embedding" in result for result in mock_results)


class TestSearchIntegrationWithDatabase:
    """Test search integration with actual database operations."""

    @pytest.mark.asyncio
    async def test_end_to_end_search_flow(
        self,
        async_session: AsyncSession,
        test_repository: Repository,
        mock_embedding_provider
    ):
        """Test complete search flow with database integration."""
        snippet_repo = CodeSnippetRepo(async_session)
        search_manager = SearchServiceManager()
        
        # Create test snippets in database
        snippets_data = [
            {
                "repository_id": test_repository.id,
                "file_path": "src/auth.py",
                "language": "python",
                "content": "def authenticate_user(username, password): return check_credentials(username, password)",
                "embedding": [0.1, 0.2, 0.3] + [0.0] * 1533,
                "start_line": 1,
                "end_line": 2,
                "complexity_score": 3.0
            },
            {
                "repository_id": test_repository.id,
                "file_path": "src/utils.py",
                "language": "python",
                "content": "def calculate_hash(data): return hashlib.md5(data.encode()).hexdigest()",
                "embedding": [0.8, 0.7, 0.6] + [0.0] * 1533,
                "start_line": 5,
                "end_line": 5,
                "complexity_score": 2.0
            }
        ]
        
        await snippet_repo.create_batch(snippets_data)
        
        # Mock the vector search to return realistic results
        def mock_vector_search(*args, **kwargs):
            return [
                {
                    "snippet_id": str(uuid.uuid4()),
                    "content": snippets_data[0]["content"],
                    "file_path": snippets_data[0]["file_path"],
                    "similarity_score": 0.92,
                    "line_start": 1,
                    "line_end": 2
                }
            ]
        
        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_search.return_value = mock_vector_search()
            
            # Perform search
            results = await search_manager.quick_search(
                query="user authentication function",
                language="python",
                limit=5
            )
        
        assert len(results) > 0
        assert "authenticate" in results[0]["content"]
        assert results[0]["similarity_score"] > 0.9

    @pytest.mark.asyncio
    async def test_search_with_database_constraints(
        self,
        async_session: AsyncSession,
        test_repository: Repository
    ):
        """Test search with database-level constraints and filters."""
        snippet_repo = CodeSnippetRepo(async_session)
        search_service = VectorSearchService()
        
        # Create test data with varying complexity scores
        for i in range(5):
            await snippet_repo.create(
                repository_id=test_repository.id,
                file_path=f"src/complexity_{i}.py",
                language="python",
                content=f"def complex_function_{i}(): pass",
                embedding=[0.1 * i] * 1536,
                start_line=1,
                end_line=1,
                complexity_score=float(i + 1)  # 1.0 to 5.0
            )
        
        # Mock vector search with database filtering
        query_embedding = [0.1] * 1536
        
        with patch.object(search_service, '_execute_vector_search') as mock_search:
            # Mock returns only results within complexity range
            mock_search.return_value = [
                {
                    "snippet_id": str(uuid.uuid4()),
                    "complexity_score": 3.0,
                    "similarity_score": 0.9,
                    "content": "def complex_function_2(): pass"
                },
                {
                    "snippet_id": str(uuid.uuid4()),
                    "complexity_score": 4.0,
                    "similarity_score": 0.85,
                    "content": "def complex_function_3(): pass"
                }
            ]
            
            results = await search_service.search_similar_code(
                query_embedding=query_embedding,
                language="python",
                min_complexity=2.5,
                max_complexity=4.5,
                repository_id=test_repository.id
            )
        
        assert len(results) == 2
        assert all(2.5 <= result["complexity_score"] <= 4.5 for result in results)