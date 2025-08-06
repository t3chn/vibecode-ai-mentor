"""Comprehensive API endpoint tests for VibeCode AI Mentor."""

import json
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.models import AnalyzeRequest, IndexRequest, SearchRequest
from src.db.models import Repository, CodeSnippet, RepositoryStatus


class TestAnalyzeEndpoint:
    """Test the /analyze endpoint."""

    @pytest.mark.asyncio
    async def test_analyze_code_success(
        self, async_client: AsyncClient, sample_python_code: str
    ):
        """Test successful code analysis."""
        request_data = {
            "content": sample_python_code,
            "filename": "test.py",
            "language": "python"
        }
        
        with patch("src.api.routes.RecommendationService") as mock_service:
            mock_instance = MagicMock()
            mock_instance.analyze_and_recommend = AsyncMock(return_value={
                "analysis_id": "test-id",
                "status": "completed",
                "recommendations": []
            })
            mock_service.return_value = mock_instance
            
            response = await async_client.post(
                "/api/v1/analyze",
                json=request_data,
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "analysis_id" in data
        assert data["status"] == "processing"
        assert "test.py" in data["message"]
        assert data["estimated_time_seconds"] > 0

    @pytest.mark.asyncio
    async def test_analyze_empty_code(self, async_client: AsyncClient):
        """Test analysis with empty code content."""
        request_data = {
            "content": "",
            "filename": "empty.py",
            "language": "python"
        }
        
        response = await async_client.post(
            "/api/v1/analyze",
            json=request_data,
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot be empty" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_analyze_missing_api_key(
        self, async_client: AsyncClient, sample_python_code: str
    ):
        """Test analysis without API key."""
        request_data = {
            "content": sample_python_code,
            "filename": "test.py",
            "language": "python"
        }
        
        response = await async_client.post("/api/v1/analyze", json=request_data)
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    @pytest.mark.asyncio
    async def test_analyze_invalid_json(self, async_client: AsyncClient):
        """Test analysis with invalid JSON."""
        response = await async_client.post(
            "/api/v1/analyze",
            content="invalid json",
            headers={
                "X-API-Key": "test-api-key",
                "Content-Type": "application/json"
            }
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_analyze_service_failure(
        self, async_client: AsyncClient, sample_python_code: str
    ):
        """Test analysis when service fails."""
        request_data = {
            "content": sample_python_code,
            "filename": "test.py",
            "language": "python"
        }
        
        with patch("src.api.routes.RecommendationService") as mock_service:
            mock_service.side_effect = Exception("Service unavailable")
            
            response = await async_client.post(
                "/api/v1/analyze",
                json=request_data,
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestIndexEndpoint:
    """Test the /index endpoint."""

    @pytest.mark.asyncio
    async def test_index_repository_success(
        self, async_client: AsyncClient, temp_repository, async_session: AsyncSession
    ):
        """Test successful repository indexing."""
        request_data = {
            "repository_path": str(temp_repository),
            "repository_url": "https://github.com/test/repo"
        }
        
        with patch("src.api.routes.AnalysisService") as mock_analysis, \
             patch("src.api.routes.get_embedding_manager") as mock_embeddings:
            
            # Mock analysis service
            mock_analysis_instance = MagicMock()
            mock_analysis_instance.analyze_repository = AsyncMock(return_value=MagicMock(
                total_files=2,
                analyzed_files=2,
                failed_files=0,
                total_time_seconds=1.5,
                file_analyses=[
                    MagicMock(
                        status="success",
                        file_path="src/main.py",
                        language="python",
                        chunks=[{"content": "def test():", "start_line": 1, "end_line": 1}],
                        metrics={"cyclomatic_complexity": 1.0}
                    )
                ]
            ))
            mock_analysis.return_value = mock_analysis_instance
            
            # Mock embedding manager
            mock_embedding_instance = MagicMock()
            mock_embedding_instance.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
            mock_embeddings.return_value = mock_embedding_instance
            
            response = await async_client.post(
                "/api/v1/index",
                json=request_data,
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "repository_id" in data
        assert data["name"] == "test_repo"
        assert data["status"] == "indexing"
        assert data["total_files"] > 0

    @pytest.mark.asyncio
    async def test_index_nonexistent_path(self, async_client: AsyncClient):
        """Test indexing nonexistent repository path."""
        request_data = {
            "repository_path": "/nonexistent/path",
            "repository_url": "https://github.com/test/repo"
        }
        
        response = await async_client.post(
            "/api/v1/index",
            json=request_data,
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "does not exist" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_index_file_instead_of_directory(
        self, async_client: AsyncClient, temp_python_file
    ):
        """Test indexing a file instead of directory."""
        request_data = {
            "repository_path": str(temp_python_file),
            "repository_url": "https://github.com/test/repo"
        }
        
        response = await async_client.post(
            "/api/v1/index",
            json=request_data,
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "must be a directory" in response.json()["detail"]


class TestSearchEndpoint:
    """Test the /search endpoint."""

    @pytest.mark.asyncio
    async def test_search_patterns_success(
        self, async_client: AsyncClient, sample_search_results
    ):
        """Test successful pattern search."""
        request_data = {
            "query": "calculate average function",
            "language": "python",
            "limit": 10
        }
        
        with patch("src.api.routes.SearchServiceManager") as mock_search:
            mock_instance = MagicMock()
            mock_instance.quick_search = AsyncMock(return_value=sample_search_results)
            mock_search.return_value = mock_instance
            
            response = await async_client.post(
                "/api/v1/search",
                json=request_data,
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert data["total_count"] == len(sample_search_results)
        assert data["query"] == request_data["query"]
        assert "search_time_ms" in data

    @pytest.mark.asyncio
    async def test_search_empty_query(self, async_client: AsyncClient):
        """Test search with empty query."""
        request_data = {
            "query": "",
            "language": "python"
        }
        
        response = await async_client.post(
            "/api/v1/search",
            json=request_data,
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "cannot be empty" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_search_with_filters(self, async_client: AsyncClient):
        """Test search with various filters."""
        request_data = {
            "query": "authentication function",
            "language": "python",
            "limit": 5,
            "similarity_threshold": 0.8,
            "repository_filter": "auth-service"
        }
        
        with patch("src.api.routes.SearchServiceManager") as mock_search:
            mock_instance = MagicMock()
            mock_instance.quick_search = AsyncMock(return_value=[])
            mock_search.return_value = mock_instance
            
            response = await async_client.post(
                "/api/v1/search",
                json=request_data,
                headers={"X-API-Key": "test-api-key"}
            )
            
            # Verify search was called with correct parameters
            mock_instance.quick_search.assert_called_once_with(
                query="authentication function",
                language="python",
                limit=5,
                similarity_threshold=0.8,
                repository_filter="auth-service"
            )
        
        assert response.status_code == status.HTTP_200_OK


class TestRecommendationsEndpoint:
    """Test the /recommendations endpoint."""

    @pytest.mark.asyncio
    async def test_get_recommendations_success(
        self, async_client: AsyncClient, sample_analysis_result
    ):
        """Test getting recommendations for completed analysis."""
        analysis_id = str(uuid.uuid4())
        
        with patch("src.api.routes.analysis_cache", {
            analysis_id: {
                "status": "completed",
                **sample_analysis_result
            }
        }):
            response = await async_client.get(
                f"/api/v1/recommendations/{analysis_id}",
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["analysis_id"] == analysis_id
        assert data["status"] == "completed"
        assert "recommendations" in data
        assert len(data["recommendations"]) > 0
        assert "summary" in data
        assert "score" in data

    @pytest.mark.asyncio
    async def test_get_recommendations_processing(self, async_client: AsyncClient):
        """Test getting recommendations for processing analysis."""
        analysis_id = str(uuid.uuid4())
        
        with patch("src.api.routes.analysis_cache", {
            analysis_id: {"status": "processing"}
        }):
            response = await async_client.get(
                f"/api/v1/recommendations/{analysis_id}",
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "processing"
        assert data["recommendations"] == []

    @pytest.mark.asyncio
    async def test_get_recommendations_not_found(self, async_client: AsyncClient):
        """Test getting recommendations for nonexistent analysis."""
        analysis_id = str(uuid.uuid4())
        
        response = await async_client.get(
            f"/api/v1/recommendations/{analysis_id}",
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_get_recommendations_failed(self, async_client: AsyncClient):
        """Test getting recommendations for failed analysis."""
        analysis_id = str(uuid.uuid4())
        
        with patch("src.api.routes.analysis_cache", {
            analysis_id: {
                "status": "failed",
                "error": "Analysis timeout"
            }
        }):
            response = await async_client.get(
                f"/api/v1/recommendations/{analysis_id}",
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Analysis timeout" in response.json()["detail"]


class TestRepositoriesEndpoint:
    """Test the /repositories endpoint."""

    @pytest.mark.asyncio
    async def test_list_repositories_success(
        self, async_client: AsyncClient, test_repository
    ):
        """Test listing repositories."""
        with patch("src.api.routes.db.execute") as mock_execute:
            # Mock database result
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = [test_repository]
            mock_execute.return_value = mock_result
            
            response = await async_client.get(
                "/api/v1/repositories",
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    @pytest.mark.asyncio
    async def test_list_repositories_with_filters(self, async_client: AsyncClient):
        """Test listing repositories with filters."""
        with patch("src.api.routes.db.execute") as mock_execute:
            mock_result = MagicMock()
            mock_result.scalars.return_value.all.return_value = []
            mock_execute.return_value = mock_result
            
            response = await async_client.get(
                "/api/v1/repositories?skip=0&limit=10&status_filter=completed",
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_200_OK

    @pytest.mark.asyncio
    async def test_list_repositories_invalid_status(self, async_client: AsyncClient):
        """Test listing repositories with invalid status filter."""
        response = await async_client.get(
            "/api/v1/repositories?status_filter=invalid_status",
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST


class TestHealthEndpoint:
    """Test the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, async_client: AsyncClient):
        """Test health check when all services are healthy."""
        with patch("src.api.routes.RecommendationService") as mock_rec, \
             patch("src.api.routes.SearchServiceManager") as mock_search, \
             patch("src.api.routes.get_embedding_manager") as mock_embed:
            
            # Mock healthy services
            mock_rec_instance = MagicMock()
            mock_rec_instance.health_check = AsyncMock(return_value={"status": "healthy"})
            mock_rec.return_value = mock_rec_instance
            
            mock_search_instance = MagicMock()
            mock_search_instance.get_service_health = AsyncMock(return_value={"status": "healthy"})
            mock_search.return_value = mock_search_instance
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.health_check = MagicMock(return_value={"primary": True, "fallback": True})
            mock_embed.return_value = mock_embed_instance
            
            response = await async_client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_check_degraded(self, async_client: AsyncClient):
        """Test health check when some services are unhealthy."""
        with patch("src.api.routes.RecommendationService") as mock_rec, \
             patch("src.api.routes.SearchServiceManager") as mock_search, \
             patch("src.api.routes.get_embedding_manager") as mock_embed:
            
            # Mock mixed health status
            mock_rec_instance = MagicMock()
            mock_rec_instance.health_check = AsyncMock(return_value={"status": "unhealthy"})
            mock_rec.return_value = mock_rec_instance
            
            mock_search_instance = MagicMock()
            mock_search_instance.get_service_health = AsyncMock(return_value={"status": "healthy"})
            mock_search.return_value = mock_search_instance
            
            mock_embed_instance = MagicMock()
            mock_embed_instance.health_check = MagicMock(return_value={"primary": False, "fallback": True})
            mock_embed.return_value = mock_embed_instance
            
            response = await async_client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "degraded"


class TestWebSocketEndpoint:
    """Test WebSocket endpoints."""

    @pytest.mark.asyncio
    async def test_websocket_indexing_progress(self, async_client: AsyncClient):
        """Test WebSocket connection for indexing progress."""
        repository_id = str(uuid.uuid4())
        
        with patch("src.api.routes.indexing_cache", {
            repository_id: {
                "status": "indexing",
                "analyzed_files": 5,
                "total_files": 10
            }
        }):
            # Note: WebSocket testing with httpx requires special handling
            # This is a basic structure - full WebSocket testing would require
            # a more complex setup with websockets library
            pass

    @pytest.mark.asyncio
    async def test_websocket_invalid_api_key(self, async_client: AsyncClient):
        """Test WebSocket connection with invalid API key."""
        # WebSocket authentication test
        pass


class TestRateLimiting:
    """Test API rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, async_client: AsyncClient):
        """Test rate limiting when limits are exceeded."""
        # This would require configuring rate limits and making multiple requests
        # Implementation depends on the actual rate limiting configuration
        pass


class TestErrorHandling:
    """Test error handling across endpoints."""

    @pytest.mark.asyncio
    async def test_internal_server_error_handling(self, async_client: AsyncClient):
        """Test that internal server errors are handled gracefully."""
        with patch("src.api.routes.RecommendationService") as mock_service:
            mock_service.side_effect = Exception("Unexpected error")
            
            response = await async_client.post(
                "/api/v1/analyze",
                json={
                    "content": "def test(): pass",
                    "filename": "test.py",
                    "language": "python"
                },
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Internal server error" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_validation_errors(self, async_client: AsyncClient):
        """Test request validation errors."""
        # Missing required fields
        response = await async_client.post(
            "/api/v1/analyze",
            json={"filename": "test.py"},  # Missing content
            headers={"X-API-Key": "test-api-key"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.asyncio
    async def test_database_connection_error(self, async_client: AsyncClient):
        """Test handling of database connection errors."""
        with patch("src.api.routes.RepositoryRepo") as mock_repo:
            mock_repo.side_effect = Exception("Database connection failed")
            
            response = await async_client.get(
                "/api/v1/repositories",
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR


class TestRepositorySpecificSearch:
    """Test repository-specific search functionality."""

    @pytest.mark.asyncio
    async def test_search_in_repository_success(
        self, async_client: AsyncClient, test_repository
    ):
        """Test searching within a specific repository."""
        with patch("src.api.routes.RepositoryRepo") as mock_repo, \
             patch("src.api.routes.SearchServiceManager") as mock_search:
            
            # Mock repository lookup
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_id = AsyncMock(return_value=test_repository)
            mock_repo.return_value = mock_repo_instance
            
            # Mock search results
            mock_search_instance = MagicMock()
            mock_search_instance.search_repository_patterns = AsyncMock(return_value={
                "results": [{"snippet_id": "123", "similarity_score": 0.9}]
            })
            mock_search.return_value = mock_search_instance
            
            response = await async_client.post(
                f"/api/v1/repositories/{test_repository.id}/search",
                json={"query": "test function"},
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "results" in data
        assert "search_time_ms" in data

    @pytest.mark.asyncio
    async def test_search_in_nonexistent_repository(self, async_client: AsyncClient):
        """Test searching in nonexistent repository."""
        fake_id = str(uuid.uuid4())
        
        with patch("src.api.routes.RepositoryRepo") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_id = AsyncMock(return_value=None)
            mock_repo.return_value = mock_repo_instance
            
            response = await async_client.post(
                f"/api/v1/repositories/{fake_id}/search",
                json={"query": "test function"},
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @pytest.mark.asyncio
    async def test_search_in_incomplete_repository(
        self, async_client: AsyncClient, async_session: AsyncSession
    ):
        """Test searching in repository that hasn't completed indexing."""
        # Create repository with pending status
        pending_repo = Repository(
            name="pending-repo",
            status=RepositoryStatus.INDEXING,
            total_files=0
        )
        async_session.add(pending_repo)
        await async_session.commit()
        await async_session.refresh(pending_repo)
        
        with patch("src.api.routes.RepositoryRepo") as mock_repo:
            mock_repo_instance = MagicMock()
            mock_repo_instance.get_by_id = AsyncMock(return_value=pending_repo)
            mock_repo.return_value = mock_repo_instance
            
            response = await async_client.post(
                f"/api/v1/repositories/{pending_repo.id}/search",
                json={"query": "test function"},
                headers={"X-API-Key": "test-api-key"}
            )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not completed" in response.json()["detail"]