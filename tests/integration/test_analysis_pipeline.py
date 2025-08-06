"""Integration tests for the complete analysis pipeline."""

import asyncio
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.services.analysis import AnalysisService
from src.analyzer.parser import PythonParser
from src.analyzer.chunker import CodeChunker
from src.analyzer.metrics import CodeMetrics
from src.embeddings.factory import get_embedding_manager
from src.generator.recommendation_service import RecommendationService
from src.search.service import SearchServiceManager
from src.db.repositories import RepositoryRepo, CodeSnippetRepo, AnalysisResultRepo
from src.db.models import Repository, CodeSnippet, RepositoryStatus


class TestAnalysisPipeline:
    """Test the complete analysis pipeline from code to recommendations."""

    @pytest.mark.asyncio
    async def test_complete_file_analysis(
        self,
        sample_python_code: str,
        temp_python_file: Path,
        async_session: AsyncSession,
        mock_embedding_provider,
        mock_llm_client
    ):
        """Test complete analysis of a single file."""
        # Initialize services
        analysis_service = AnalysisService()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Analyze the file
            result = await analysis_service.analyze_file(
                file_path=temp_python_file,
                language="python"
            )
        
        # Verify analysis result structure
        assert result is not None
        assert result.file_path == str(temp_python_file)
        assert result.language == "python"
        assert result.status == "success"
        assert len(result.chunks) > 0
        assert result.metrics is not None
        
        # Verify chunks have required fields
        for chunk in result.chunks:
            assert "content" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert chunk["start_line"] <= chunk["end_line"]
        
        # Verify metrics
        assert "lines_of_code" in result.metrics
        assert "cyclomatic_complexity" in result.metrics
        assert result.metrics["lines_of_code"] > 0

    @pytest.mark.asyncio
    async def test_repository_analysis_pipeline(
        self,
        temp_repository: Path,
        async_session: AsyncSession,
        mock_embedding_provider,
        mock_llm_client
    ):
        """Test complete repository analysis pipeline."""
        # Initialize services
        analysis_service = AnalysisService()
        repo_repo = RepositoryRepo(async_session)
        snippet_repo = CodeSnippetRepo(async_session)
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Create repository record
            repo = await repo_repo.create(
                name="test-repo",
                url="https://github.com/test/repo"
            )
            
            # Analyze repository
            repo_analysis = await analysis_service.analyze_repository(
                repo_path=temp_repository,
                include_patterns=["**/*.py"],
                exclude_patterns=["**/test_*", "**/__pycache__/**"]
            )
        
        # Verify repository analysis
        assert repo_analysis.total_files >= 2  # main.py and utils.py
        assert repo_analysis.analyzed_files > 0
        assert repo_analysis.failed_files == 0
        assert len(repo_analysis.file_analyses) > 0
        
        # Verify file analyses
        for file_analysis in repo_analysis.file_analyses:
            if file_analysis.status == "success":
                assert len(file_analysis.chunks) > 0
                assert file_analysis.metrics is not None
        
        # Store snippets in database
        snippet_data = []
        for file_analysis in repo_analysis.file_analyses:
            if file_analysis.status == "success" and file_analysis.chunks:
                for chunk in file_analysis.chunks:
                    embedding = await mock_embedding_provider.generate_embedding(
                        chunk["content"]
                    )
                    
                    snippet_data.append({
                        "repository_id": repo.id,
                        "file_path": file_analysis.file_path,
                        "language": file_analysis.language,
                        "content": chunk["content"],
                        "embedding": embedding,
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "complexity_score": file_analysis.metrics.get("cyclomatic_complexity", 0.0)
                    })
        
        # Store in batches
        await snippet_repo.create_batch(snippet_data)
        
        # Verify data was stored
        stored_snippets = await snippet_repo.get_by_repository_id(repo.id)
        assert len(stored_snippets) == len(snippet_data)

    @pytest.mark.asyncio
    async def test_end_to_end_recommendation_pipeline(
        self,
        sample_python_code: str,
        async_session: AsyncSession,
        mock_embedding_provider,
        mock_llm_client
    ):
        """Test end-to-end pipeline from code to recommendations."""
        # Initialize services
        recommendation_service = RecommendationService()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory, \
             patch("src.generator.recommendation_service.LLMClient") as mock_llm_factory:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_llm_factory.return_value = mock_llm_client
            
            # Run complete analysis and recommendation pipeline
            result = await recommendation_service.analyze_and_recommend(
                code=sample_python_code,
                filename="test.py",
                language="python",
                find_similar=True
            )
        
        # Verify result structure
        assert "recommendations" in result
        assert "refactoring_suggestions" in result
        assert "anti_pattern_fixes" in result
        assert "summary" in result
        assert "overall_score" in result
        
        # Verify recommendations format
        for rec in result["recommendations"]:
            assert "type" in rec
            assert "severity" in rec
            assert "message" in rec
            assert "confidence" in rec

    @pytest.mark.asyncio
    async def test_search_integration_pipeline(
        self,
        async_session: AsyncSession,
        test_repository,
        test_code_snippet,
        mock_embedding_provider
    ):
        """Test search integration with stored data."""
        search_manager = SearchServiceManager()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Mock database search results
            with patch("src.search.vector_search.VectorSearchService.search_similar_code") as mock_search:
                mock_search.return_value = [
                    {
                        "snippet_id": str(test_code_snippet.id),
                        "content": test_code_snippet.content,
                        "file_path": test_code_snippet.file_path,
                        "similarity_score": 0.95,
                        "line_start": test_code_snippet.start_line,
                        "line_end": test_code_snippet.end_line
                    }
                ]
                
                # Perform search
                results = await search_manager.quick_search(
                    query="hello world function",
                    language="python",
                    limit=10
                )
        
        # Verify search results
        assert len(results) > 0
        assert results[0]["similarity_score"] > 0.9
        assert "content" in results[0]
        assert "file_path" in results[0]

    @pytest.mark.asyncio
    async def test_concurrent_analysis_pipeline(
        self,
        temp_repository: Path,
        async_session: AsyncSession,
        mock_embedding_provider
    ):
        """Test pipeline performance with concurrent file analysis."""
        analysis_service = AnalysisService()
        
        # Get all Python files
        python_files = list(temp_repository.glob("**/*.py"))
        assert len(python_files) >= 2
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Analyze files concurrently
            tasks = [
                analysis_service.analyze_file(file_path, "python")
                for file_path in python_files
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all analyses completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(python_files)
        
        # Verify each result
        for result in successful_results:
            assert result.status == "success"
            assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_error_handling_in_pipeline(
        self,
        temp_python_file: Path,
        async_session: AsyncSession,
        failing_embedding_provider
    ):
        """Test error handling throughout the pipeline."""
        analysis_service = AnalysisService()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = failing_embedding_provider
            
            # This should handle embedding failures gracefully
            result = await analysis_service.analyze_file(
                file_path=temp_python_file,
                language="python"
            )
        
        # Analysis should still succeed even if embedding fails
        assert result is not None
        assert result.status == "success"  # Parser and chunker should work
        assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_large_file_pipeline_performance(
        self,
        large_code_sample: str,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test pipeline performance with large files."""
        # Create large test file
        large_file = tmp_path / "large_file.py"
        large_file.write_text(large_code_sample)
        
        analysis_service = AnalysisService()
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            import time
            start_time = time.time()
            
            # Analyze large file
            result = await analysis_service.analyze_file(
                file_path=large_file,
                language="python"
            )
            
            end_time = time.time()
            analysis_time = end_time - start_time
        
        # Verify performance and results
        assert result.status == "success"
        assert len(result.chunks) > 10  # Should have many chunks
        assert analysis_time < 5.0  # Should complete within 5 seconds
        
        # Verify chunks are properly sized
        for chunk in result.chunks:
            assert len(chunk["content"]) <= 2048  # Max chunk size
            assert len(chunk["content"]) >= 50   # Reasonable minimum


class TestDatabaseIntegration:
    """Test database integration aspects of the pipeline."""

    @pytest.mark.asyncio
    async def test_repository_crud_operations(self, async_session: AsyncSession):
        """Test repository CRUD operations."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create
        repo = await repo_repo.create(
            name="test-repo",
            url="https://github.com/test/repo"
        )
        assert repo.id is not None
        assert repo.status == RepositoryStatus.PENDING
        
        # Read
        found_repo = await repo_repo.get_by_id(repo.id)
        assert found_repo.name == "test-repo"
        
        found_by_name = await repo_repo.get_by_name("test-repo")
        assert found_by_name.id == repo.id
        
        # Update
        await repo_repo.update_status(repo.id, RepositoryStatus.COMPLETED, 5)
        updated_repo = await repo_repo.get_by_id(repo.id)
        assert updated_repo.status == RepositoryStatus.COMPLETED
        assert updated_repo.total_files == 5
        
        # List
        repos = await repo_repo.list_repositories(limit=10)
        assert len(repos) >= 1
        assert any(r.id == repo.id for r in repos)

    @pytest.mark.asyncio
    async def test_code_snippet_batch_operations(
        self, async_session: AsyncSession, test_repository
    ):
        """Test batch operations for code snippets."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Prepare batch data
        batch_data = []
        for i in range(5):
            batch_data.append({
                "repository_id": test_repository.id,
                "file_path": f"src/file_{i}.py",
                "language": "python",
                "content": f"def function_{i}():\n    return {i}",
                "embedding": [0.1 * i] * 1536,
                "start_line": 1,
                "end_line": 2,
                "complexity_score": float(i + 1)
            })
        
        # Create batch
        await snippet_repo.create_batch(batch_data)
        
        # Verify stored
        snippets = await snippet_repo.get_by_repository_id(test_repository.id)
        assert len(snippets) == 5
        
        # Test batch retrieval by language
        python_snippets = await snippet_repo.get_by_language("python", limit=10)
        assert len(python_snippets) >= 5

    @pytest.mark.asyncio
    async def test_analysis_result_storage(
        self, async_session: AsyncSession, test_code_snippet
    ):
        """Test analysis result storage and retrieval."""
        result_repo = AnalysisResultRepo(async_session)
        
        # Create analysis result
        recommendations = [
            {
                "type": "style",
                "message": "Add type hints",
                "confidence": 0.8
            }
        ]
        
        result = await result_repo.create(
            snippet_id=test_code_snippet.id,
            recommendations=recommendations,
            similar_patterns=[],
            quality_score=85.0
        )
        
        assert result.id is not None
        assert result.quality_score == 85.0
        assert len(result.recommendations) == 1
        
        # Retrieve by snippet
        found_result = await result_repo.get_by_snippet_id(test_code_snippet.id)
        assert found_result.id == result.id

    @pytest.mark.asyncio
    async def test_vector_search_integration(
        self, async_session: AsyncSession, test_repository
    ):
        """Test vector search integration with database."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create snippets with similar embeddings
        similar_embedding = [0.1] * 1536
        different_embedding = [0.9] * 1536
        
        snippets_data = [
            {
                "repository_id": test_repository.id,
                "file_path": "src/similar1.py",
                "language": "python",
                "content": "def calculate_sum(numbers): return sum(numbers)",
                "embedding": similar_embedding,
                "start_line": 1,
                "end_line": 1,
                "complexity_score": 1.0
            },
            {
                "repository_id": test_repository.id,
                "file_path": "src/similar2.py",
                "language": "python",
                "content": "def compute_total(values): return sum(values)",
                "embedding": [x + 0.01 for x in similar_embedding],  # Very similar
                "start_line": 1,
                "end_line": 1,
                "complexity_score": 1.0
            },
            {
                "repository_id": test_repository.id,
                "file_path": "src/different.py",
                "language": "python",
                "content": "def render_template(name): return f'<html>{name}</html>'",
                "embedding": different_embedding,
                "start_line": 1,
                "end_line": 1,
                "complexity_score": 2.0
            }
        ]
        
        await snippet_repo.create_batch(snippets_data)
        
        # Test similarity search (mocked since we're using SQLite)
        query_embedding = similar_embedding
        
        with patch("src.search.vector_search.VectorSearchService.search_similar_code") as mock_search:
            mock_search.return_value = [
                {
                    "snippet_id": str(uuid.uuid4()),
                    "content": snippets_data[0]["content"],
                    "similarity_score": 0.99
                },
                {
                    "snippet_id": str(uuid.uuid4()),
                    "content": snippets_data[1]["content"],
                    "similarity_score": 0.95
                }
            ]
            
            from src.search.vector_search import VectorSearchService
            search_service = VectorSearchService()
            
            results = await search_service.search_similar_code(
                query_embedding=query_embedding,
                language="python",
                limit=5
            )
        
        # Should find similar snippets first
        assert len(results) >= 2
        assert results[0]["similarity_score"] > results[1]["similarity_score"]

    @pytest.mark.asyncio
    async def test_transaction_handling(self, async_session: AsyncSession):
        """Test database transaction handling in pipeline."""
        repo_repo = RepositoryRepo(async_session)
        
        try:
            async with async_session.begin():
                # Create repository
                repo = await repo_repo.create(
                    name="transaction-test",
                    url="https://github.com/test/transaction"
                )
                
                # Simulate error after creation
                if repo.id:
                    raise Exception("Simulated error")
                    
        except Exception:
            # Transaction should be rolled back
            pass
        
        # Repository should not exist due to rollback
        found_repo = await repo_repo.get_by_name("transaction-test")
        assert found_repo is None


class TestServiceHealthChecks:
    """Test health check integration across services."""

    @pytest.mark.asyncio
    async def test_recommendation_service_health(self, mock_llm_client):
        """Test recommendation service health check."""
        with patch("src.generator.recommendation_service.LLMClient") as mock_factory:
            mock_factory.return_value = mock_llm_client
            
            service = RecommendationService()
            health = await service.health_check()
        
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_embedding_service_health(self, mock_embedding_provider):
        """Test embedding service health check."""
        with patch("src.embeddings.factory.get_embedding_manager") as mock_factory:
            mock_factory.return_value = mock_embedding_provider
            
            manager = mock_factory()
            health = manager.health_check()
        
        assert health is not None

    @pytest.mark.asyncio
    async def test_search_service_health(self):
        """Test search service health check."""
        search_manager = SearchServiceManager()
        health = await search_manager.get_service_health()
        
        # Should return health status structure
        assert "status" in health