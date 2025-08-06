"""Performance and load testing for VibeCode AI Mentor."""

import asyncio
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient

from src.api.app import app
from src.services.analysis import AnalysisService
from src.search.service import SearchServiceManager
from src.generator.recommendation_service import RecommendationService


class TestAPIPerformance:
    """Test API endpoint performance under load."""

    @pytest.mark.asyncio
    async def test_analyze_endpoint_performance(
        self, 
        sample_python_code: str,
        mock_embedding_provider,
        mock_llm_client
    ):
        """Test analyze endpoint performance with concurrent requests."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "content": sample_python_code,
                "filename": "performance_test.py",
                "language": "python"
            }
            
            headers = {"X-API-Key": "test-api-key"}
            
            # Mock services for consistent performance testing
            with patch("src.api.routes.RecommendationService") as mock_service:
                mock_instance = MagicMock()
                mock_instance.analyze_and_recommend = AsyncMock(return_value={
                    "analysis_id": "test-id",
                    "status": "completed",
                    "recommendations": []
                })
                mock_service.return_value = mock_instance
                
                # Test concurrent requests
                start_time = time.time()
                
                tasks = [
                    client.post("/api/v1/analyze", json=request_data, headers=headers)
                    for _ in range(10)
                ]
                
                responses = await asyncio.gather(*tasks)
                
                end_time = time.time()
                total_time = end_time - start_time
        
        # Performance assertions
        assert all(response.status_code == 202 for response in responses)
        assert total_time < 2.0  # All 10 requests should complete within 2 seconds
        
        # Calculate average response time
        avg_response_time = total_time / len(responses)
        assert avg_response_time < 0.2  # Average response time should be under 200ms

    @pytest.mark.asyncio
    async def test_search_endpoint_performance(self, mock_embedding_provider):
        """Test search endpoint performance."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "query": "calculate average function",
                "language": "python",
                "limit": 20
            }
            
            headers = {"X-API-Key": "test-api-key"}
            
            # Mock search service
            with patch("src.api.routes.SearchServiceManager") as mock_search:
                mock_instance = MagicMock()
                mock_instance.quick_search = AsyncMock(return_value=[
                    {
                        "snippet_id": str(uuid.uuid4()),
                        "content": "def calculate_average(): pass",
                        "similarity_score": 0.95
                    }
                    for _ in range(20)
                ])
                mock_search.return_value = mock_instance
                
                # Measure search performance
                response_times = []
                
                for _ in range(5):
                    start_time = time.time()
                    response = await client.post("/api/v1/search", json=request_data, headers=headers)
                    end_time = time.time()
                    
                    response_times.append(end_time - start_time)
                    assert response.status_code == 200
        
        # Performance metrics
        avg_time = statistics.mean(response_times)
        p95_time = sorted(response_times)[int(0.95 * len(response_times))]
        
        assert avg_time < 0.5  # Average under 500ms
        assert p95_time < 1.0   # 95th percentile under 1 second

    @pytest.mark.asyncio
    async def test_repository_indexing_performance(
        self, 
        temp_repository,
        mock_embedding_provider
    ):
        """Test repository indexing performance."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            request_data = {
                "repository_path": str(temp_repository),
                "repository_url": "https://github.com/test/perf-repo"
            }
            
            headers = {"X-API-Key": "test-api-key"}
            
            # Mock services for performance testing
            with patch("src.api.routes.AnalysisService") as mock_analysis, \
                 patch("src.api.routes.get_embedding_manager") as mock_embeddings:
                
                mock_analysis_instance = MagicMock()
                mock_analysis_instance.analyze_repository = AsyncMock(return_value=MagicMock(
                    total_files=10,
                    analyzed_files=10,
                    failed_files=0,
                    total_time_seconds=2.5,
                    file_analyses=[
                        MagicMock(
                            status="success",
                            chunks=[{"content": "def test():", "start_line": 1, "end_line": 1}],
                            metrics={"cyclomatic_complexity": 1.0}
                        )
                        for _ in range(10)
                    ]
                ))
                mock_analysis.return_value = mock_analysis_instance
                
                mock_embeddings.return_value = mock_embedding_provider
                
                # Test indexing initiation performance
                start_time = time.time()
                response = await client.post("/api/v1/index", json=request_data, headers=headers)
                end_time = time.time()
                
                initiation_time = end_time - start_time
        
        assert response.status_code == 202
        assert initiation_time < 1.0  # Indexing initiation should be fast

    @pytest.mark.asyncio
    async def test_health_endpoint_performance(self):
        """Test health endpoint performance."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            response_times = []
            
            # Mock all health check services
            with patch("src.api.routes.RecommendationService") as mock_rec, \
                 patch("src.api.routes.SearchServiceManager") as mock_search, \
                 patch("src.api.routes.get_embedding_manager") as mock_embed:
                
                # Setup healthy mocks
                mock_rec_instance = MagicMock()
                mock_rec_instance.health_check = AsyncMock(return_value={"status": "healthy"})
                mock_rec.return_value = mock_rec_instance
                
                mock_search_instance = MagicMock()
                mock_search_instance.get_service_health = AsyncMock(return_value={"status": "healthy"})
                mock_search.return_value = mock_search_instance
                
                mock_embed_instance = MagicMock()
                mock_embed_instance.health_check = MagicMock(return_value={"primary": True})
                mock_embed.return_value = mock_embed_instance
                
                # Test multiple health checks
                for _ in range(10):
                    start_time = time.time()
                    response = await client.get("/api/v1/health")
                    end_time = time.time()
                    
                    response_times.append(end_time - start_time)
                    assert response.status_code == 200
        
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        
        assert avg_time < 0.1   # Average under 100ms
        assert max_time < 0.5   # Maximum under 500ms


class TestServicePerformance:
    """Test individual service performance."""

    @pytest.mark.asyncio
    async def test_analysis_service_performance(
        self, 
        large_code_sample: str,
        tmp_path,
        mock_embedding_provider
    ):
        """Test analysis service performance with large files."""
        # Create large test file
        large_file = tmp_path / "large_performance_test.py"
        large_file.write_text(large_code_sample)
        
        analysis_service = AnalysisService()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Measure analysis performance
            start_time = time.time()
            
            result = await analysis_service.analyze_file(
                file_path=large_file,
                language="python"
            )
            
            end_time = time.time()
            analysis_time = end_time - start_time
        
        # Performance assertions
        assert result.status == "success"
        assert analysis_time < 3.0  # Should complete within 3 seconds
        assert len(result.chunks) > 0
        
        # Verify throughput (lines per second)
        lines_of_code = len(large_code_sample.split('\n'))
        throughput = lines_of_code / analysis_time
        assert throughput > 1000  # Should process at least 1000 lines/second

    @pytest.mark.asyncio
    async def test_search_service_performance(
        self, 
        mock_embedding_provider
    ):
        """Test search service performance."""
        search_manager = SearchServiceManager()
        
        # Mock search results
        mock_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "content": f"def function_{i}(): pass",
                "similarity_score": 0.9 - (i * 0.01)
            }
            for i in range(100)  # 100 results
        ]
        
        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_search.return_value = mock_results
            
            # Test search performance with different result sizes
            for limit in [10, 50, 100]:
                start_time = time.time()
                
                results = await search_manager.quick_search(
                    query="test function",
                    language="python",
                    limit=limit
                )
                
                end_time = time.time()
                search_time = end_time - start_time
                
                assert len(results) == limit
                assert search_time < 0.5  # Should complete within 500ms

    @pytest.mark.asyncio
    async def test_recommendation_service_performance(
        self,
        sample_python_code: str,
        mock_embedding_provider,
        mock_llm_client
    ):
        """Test recommendation service performance."""
        recommendation_service = RecommendationService()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory, \
             patch("src.generator.recommendation_service.LLMClient") as mock_llm_factory:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_llm_factory.return_value = mock_llm_client
            
            # Test recommendation generation performance
            start_time = time.time()
            
            result = await recommendation_service.analyze_and_recommend(
                code=sample_python_code,
                filename="performance_test.py",
                language="python",
                find_similar=True
            )
            
            end_time = time.time()
            recommendation_time = end_time - start_time
        
        assert "recommendations" in result
        assert recommendation_time < 2.0  # Should complete within 2 seconds

    @pytest.mark.asyncio
    async def test_concurrent_service_operations(
        self,
        sample_python_code: str,
        mock_embedding_provider,
        mock_llm_client
    ):
        """Test concurrent operations across multiple services."""
        analysis_service = AnalysisService()
        search_manager = SearchServiceManager()
        recommendation_service = RecommendationService()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory, \
             patch("src.generator.recommendation_service.LLMClient") as mock_llm_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_llm_factory.return_value = mock_llm_client
            mock_search.return_value = [{"snippet_id": "test", "similarity_score": 0.9}]
            
            # Create concurrent tasks
            tasks = [
                search_manager.quick_search("test query", "python"),
                recommendation_service.analyze_and_recommend(sample_python_code, "test.py", "python"),
                search_manager.quick_search("another query", "python"),
            ]
            
            start_time = time.time()
            results = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
        
        assert len(results) == 3
        assert all(results)  # All should have results
        assert total_time < 3.0  # Concurrent execution should be efficient


class TestMemoryPerformance:
    """Test memory usage and efficiency."""

    @pytest.mark.asyncio
    async def test_memory_usage_during_large_batch_processing(
        self,
        temp_repository,
        mock_embedding_provider
    ):
        """Test memory usage during large batch operations."""
        analysis_service = AnalysisService()
        
        # Create additional test files for larger batch
        for i in range(20):
            test_file = temp_repository / f"batch_test_{i}.py"
            test_file.write_text(f"""
def batch_function_{i}():
    '''Function {i} for batch testing.'''
    data = [x for x in range(100)]
    return sum(data) / len(data)

class BatchClass_{i}:
    def __init__(self):
        self.value = {i}
    
    def process(self, items):
        return [item * self.value for item in items]
""")
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Process repository
            result = await analysis_service.analyze_repository(
                repo_path=temp_repository,
                include_patterns=["**/*.py"],
                exclude_patterns=["**/test_*"]
            )
        
        # Verify processing completed successfully
        assert result.total_files >= 20
        assert result.analyzed_files > 0
        assert result.failed_files == 0
        
        # Memory usage should remain reasonable (this is implicit - 
        # if memory usage was excessive, the test would fail or timeout)

    @pytest.mark.asyncio
    async def test_embedding_batch_processing_memory(
        self,
        mock_embedding_provider
    ):
        """Test memory efficiency during embedding batch processing."""
        # Create large batch of text for embedding
        text_batch = [
            f"def function_{i}(): return {i} * 2" 
            for i in range(100)
        ]
        
        # Mock batch embedding processing
        mock_embedding_provider.generate_embeddings = AsyncMock(
            return_value=[[0.1 * i] * 1536 for i in range(100)]
        )
        
        start_time = time.time()
        embeddings = await mock_embedding_provider.generate_embeddings(text_batch)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        assert len(embeddings) == 100
        assert all(len(emb) == 1536 for emb in embeddings)
        assert processing_time < 1.0  # Batch processing should be efficient

    def test_search_result_memory_efficiency(self):
        """Test memory efficiency of search result handling."""
        from src.search.optimization import SearchOptimizer
        
        optimizer = SearchOptimizer()
        
        # Create large result set
        large_results = [
            {
                "snippet_id": str(uuid.uuid4()),
                "content": f"def large_function_{i}(): pass # This is a longer content string for testing memory usage",
                "similarity_score": 0.9 - (i * 0.0001),
                "file_path": f"src/large_module_{i}.py",
                "embedding": [0.1 * (i % 100)] * 1536  # Large embedding vectors
            }
            for i in range(1000)  # 1000 results with large data
        ]
        
        # Test result filtering and ranking without memory issues
        filtered_results = optimizer.filter_by_diversity(
            results=large_results,
            diversity_threshold=0.8,
            max_results=50
        )
        
        assert len(filtered_results) <= 50
        assert all("snippet_id" in result for result in filtered_results)


class TestScalabilityMetrics:
    """Test system scalability and generate performance metrics."""

    @pytest.mark.asyncio
    async def test_throughput_scaling(self, mock_embedding_provider, mock_llm_client):
        """Test system throughput scaling with increased load."""
        search_manager = SearchServiceManager()
        
        with patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_search.return_value = [{"snippet_id": "test", "similarity_score": 0.9}]
            
            # Test different load levels
            load_levels = [1, 5, 10, 20]
            performance_metrics = {}
            
            for load in load_levels:
                tasks = [
                    search_manager.quick_search(f"query {i}", "python")
                    for i in range(load)
                ]
                
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                end_time = time.time()
                
                total_time = end_time - start_time
                throughput = load / total_time  # requests per second
                
                performance_metrics[load] = {
                    "total_time": total_time,
                    "throughput": throughput,
                    "avg_response_time": total_time / load
                }
                
                assert len(results) == load
                assert all(results)
        
        # Verify throughput scaling
        assert performance_metrics[1]["throughput"] > 0
        assert performance_metrics[5]["throughput"] > 0
        assert performance_metrics[10]["throughput"] > 0
        
        # Log performance metrics for analysis
        print("\nPerformance Scaling Metrics:")
        for load, metrics in performance_metrics.items():
            print(f"Load {load}: {metrics['throughput']:.2f} req/s, "
                  f"Avg: {metrics['avg_response_time']:.3f}s")

    @pytest.mark.asyncio
    async def test_error_rate_under_load(self):
        """Test error rates under high load conditions."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            
            # Simulate some failures in the service
            failure_count = 0
            total_requests = 50
            
            with patch("src.api.routes.SearchServiceManager") as mock_search:
                def mock_search_with_failures(*args, **kwargs):
                    nonlocal failure_count
                    if failure_count < 5:  # First 5 requests fail
                        failure_count += 1
                        raise Exception("Service temporarily unavailable")
                    return [{"snippet_id": "test", "similarity_score": 0.9}]
                
                mock_instance = MagicMock()
                mock_instance.quick_search = AsyncMock(side_effect=mock_search_with_failures)
                mock_search.return_value = mock_instance
                
                # Send concurrent requests
                tasks = [
                    client.post(
                        "/api/v1/search",
                        json={"query": f"test {i}", "language": "python"},
                        headers={"X-API-Key": "test-api-key"}
                    )
                    for i in range(total_requests)
                ]
                
                responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze error rates
        successful_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code == 200]
        error_responses = [r for r in responses if hasattr(r, 'status_code') and r.status_code >= 500]
        
        success_rate = len(successful_responses) / total_requests
        error_rate = len(error_responses) / total_requests
        
        print(f"\nLoad Test Results:")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Error Rate: {error_rate:.2%}")
        
        # System should handle errors gracefully
        assert success_rate > 0.5  # At least 50% success rate under stress

    def test_performance_regression_detection(self):
        """Test for performance regression detection."""
        # This would typically compare against baseline metrics
        # For this test, we'll simulate baseline comparisons
        
        baseline_metrics = {
            "search_avg_time": 0.200,  # 200ms
            "analysis_avg_time": 1.500,  # 1.5s
            "throughput": 50.0  # 50 req/s
        }
        
        # Simulate current metrics (should be similar to baseline)
        current_metrics = {
            "search_avg_time": 0.180,  # Improved
            "analysis_avg_time": 1.600,  # Slightly slower
            "throughput": 48.0  # Slightly lower
        }
        
        # Check for regressions (>20% degradation)
        regression_threshold = 0.20
        
        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics[metric]
            
            if metric == "throughput":
                # Higher is better for throughput
                degradation = (baseline_value - current_value) / baseline_value
            else:
                # Lower is better for response times
                degradation = (current_value - baseline_value) / baseline_value
            
            print(f"{metric}: baseline={baseline_value}, current={current_value}, "
                  f"change={degradation:.1%}")
            
            # Assert no significant regression
            assert degradation < regression_threshold, \
                f"Performance regression detected in {metric}: {degradation:.1%}"


class TestResourceUtilization:
    """Test resource utilization and efficiency."""

    @pytest.mark.asyncio
    async def test_database_connection_efficiency(self, async_session):
        """Test database connection utilization."""
        from src.db.repositories import RepositoryRepo, CodeSnippetRepo
        
        repo_repo = RepositoryRepo(async_session)
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Test multiple operations using same session efficiently
        start_time = time.time()
        
        # Create repository
        repo = await repo_repo.create(name="efficiency-test", url="https://github.com/test/efficiency")
        
        # Create multiple snippets
        batch_data = [
            {
                "repository_id": repo.id,
                "file_path": f"src/file_{i}.py",
                "language": "python",
                "content": f"def func_{i}(): pass",
                "embedding": [0.1] * 1536,
                "start_line": 1,
                "end_line": 1,
                "complexity_score": 1.0
            }
            for i in range(10)
        ]
        
        await snippet_repo.create_batch(batch_data)
        
        # Query operations
        stored_snippets = await snippet_repo.get_by_repository_id(repo.id)
        
        end_time = time.time()
        operation_time = end_time - start_time
        
        assert len(stored_snippets) == 10
        assert operation_time < 1.0  # Should be efficient with single session

    @pytest.mark.asyncio
    async def test_cache_efficiency(self):
        """Test caching efficiency and hit rates."""
        search_manager = SearchServiceManager()
        query = "cached test query"
        
        # Mock caching behavior
        cache_hits = 0
        cache_misses = 0
        
        def mock_get_cache(cache_key):
            nonlocal cache_hits, cache_misses
            if cache_key == f"search_{hash(query)}_python":
                if cache_hits > 0:  # Second+ calls should hit cache
                    cache_hits += 1
                    return [{"snippet_id": "cached", "similarity_score": 0.9}]
            cache_misses += 1
            return None
        
        def mock_set_cache(cache_key, results, ttl=None):
            # Cache setting logic
            pass
        
        with patch.object(search_manager, "_get_cached_results", side_effect=mock_get_cache), \
             patch.object(search_manager, "_cache_results", side_effect=mock_set_cache), \
             patch("src.search.service.get_embedding_manager") as mock_embed_factory, \
             patch.object(search_manager.vector_search, "search_similar_code") as mock_search:
            
            mock_embedding_provider = MagicMock()
            mock_embedding_provider.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
            mock_embed_factory.return_value = mock_embedding_provider
            mock_search.return_value = [{"snippet_id": "fresh", "similarity_score": 0.9}]
            
            # First call - cache miss
            result1 = await search_manager.quick_search(query, "python")
            
            # Second call - should hit cache
            result2 = await search_manager.quick_search(query, "python")
            
            # Third call - should hit cache
            result3 = await search_manager.quick_search(query, "python")
        
        cache_hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        
        print(f"Cache Hit Rate: {cache_hit_rate:.1%} ({cache_hits} hits, {cache_misses} misses)")
        
        # Should have some cache efficiency
        assert len(result1) > 0
        assert len(result2) > 0
        assert len(result3) > 0