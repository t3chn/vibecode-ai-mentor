"""Comprehensive tests for database repositories."""

import uuid
from datetime import datetime, timedelta
from typing import List

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from src.db.models import Repository, CodeSnippet, AnalysisResult, RepositoryStatus
from src.db.repositories import RepositoryRepo, CodeSnippetRepo, AnalysisResultRepo


class TestRepositoryRepo:
    """Test RepositoryRepo operations."""

    @pytest.mark.asyncio
    async def test_create_repository(self, async_session: AsyncSession):
        """Test creating a new repository."""
        repo_repo = RepositoryRepo(async_session)
        
        repo = await repo_repo.create(
            name="test-repo",
            url="https://github.com/test/repo"
        )
        
        assert repo.id is not None
        assert repo.name == "test-repo"
        assert repo.url == "https://github.com/test/repo"
        assert repo.status == RepositoryStatus.PENDING
        assert repo.total_files == 0
        assert repo.created_at is not None

    @pytest.mark.asyncio
    async def test_create_repository_duplicate_name(self, async_session: AsyncSession):
        """Test creating repository with duplicate name."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create first repository
        await repo_repo.create(name="duplicate-repo", url="https://github.com/test/repo1")
        
        # Try to create second with same name - should succeed (names aren't unique)
        repo2 = await repo_repo.create(name="duplicate-repo", url="https://github.com/test/repo2")
        assert repo2.id is not None

    @pytest.mark.asyncio
    async def test_get_repository_by_id(self, async_session: AsyncSession):
        """Test retrieving repository by ID."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create repository
        created_repo = await repo_repo.create(name="test-repo", url="https://github.com/test/repo")
        
        # Retrieve by ID
        found_repo = await repo_repo.get_by_id(created_repo.id)
        
        assert found_repo is not None
        assert found_repo.id == created_repo.id
        assert found_repo.name == "test-repo"

    @pytest.mark.asyncio
    async def test_get_repository_by_nonexistent_id(self, async_session: AsyncSession):
        """Test retrieving repository by nonexistent ID."""
        repo_repo = RepositoryRepo(async_session)
        
        nonexistent_id = uuid.uuid4()
        found_repo = await repo_repo.get_by_id(nonexistent_id)
        
        assert found_repo is None

    @pytest.mark.asyncio
    async def test_get_repository_by_name(self, async_session: AsyncSession):
        """Test retrieving repository by name."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create repository
        await repo_repo.create(name="named-repo", url="https://github.com/test/repo")
        
        # Retrieve by name
        found_repo = await repo_repo.get_by_name("named-repo")
        
        assert found_repo is not None
        assert found_repo.name == "named-repo"

    @pytest.mark.asyncio
    async def test_get_repository_by_nonexistent_name(self, async_session: AsyncSession):
        """Test retrieving repository by nonexistent name."""
        repo_repo = RepositoryRepo(async_session)
        
        found_repo = await repo_repo.get_by_name("nonexistent")
        assert found_repo is None

    @pytest.mark.asyncio
    async def test_update_repository_status(self, async_session: AsyncSession):
        """Test updating repository status."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create repository
        repo = await repo_repo.create(name="status-test", url="https://github.com/test/repo")
        assert repo.status == RepositoryStatus.PENDING
        
        # Update status
        await repo_repo.update_status(repo.id, RepositoryStatus.INDEXING, 10)
        
        # Verify update
        updated_repo = await repo_repo.get_by_id(repo.id)
        assert updated_repo.status == RepositoryStatus.INDEXING
        assert updated_repo.total_files == 10

    @pytest.mark.asyncio
    async def test_update_repository_completion(self, async_session: AsyncSession):
        """Test updating repository to completed status."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create repository
        repo = await repo_repo.create(name="completion-test", url="https://github.com/test/repo")
        
        # Update to completed
        await repo_repo.update_status(repo.id, RepositoryStatus.COMPLETED)
        
        # Verify completion timestamp is set
        completed_repo = await repo_repo.get_by_id(repo.id)
        assert completed_repo.status == RepositoryStatus.COMPLETED
        assert completed_repo.last_indexed_at is not None

    @pytest.mark.asyncio
    async def test_list_repositories(self, async_session: AsyncSession):
        """Test listing repositories with pagination."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create multiple repositories
        repos = []
        for i in range(5):
            repo = await repo_repo.create(
                name=f"repo-{i}",
                url=f"https://github.com/test/repo-{i}"
            )
            repos.append(repo)
        
        # Test listing all
        all_repos = await repo_repo.list_repositories(limit=10)
        assert len(all_repos) >= 5
        
        # Test pagination
        first_page = await repo_repo.list_repositories(skip=0, limit=3)
        assert len(first_page) == 3
        
        second_page = await repo_repo.list_repositories(skip=3, limit=3)
        assert len(second_page) >= 2
        
        # Verify no overlap
        first_ids = {repo.id for repo in first_page}
        second_ids = {repo.id for repo in second_page}
        assert first_ids.isdisjoint(second_ids)

    @pytest.mark.asyncio
    async def test_list_repositories_by_status(self, async_session: AsyncSession):
        """Test listing repositories filtered by status."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create repositories with different statuses
        pending_repo = await repo_repo.create(name="pending", url="https://github.com/test/pending")
        
        indexing_repo = await repo_repo.create(name="indexing", url="https://github.com/test/indexing")
        await repo_repo.update_status(indexing_repo.id, RepositoryStatus.INDEXING)
        
        completed_repo = await repo_repo.create(name="completed", url="https://github.com/test/completed")
        await repo_repo.update_status(completed_repo.id, RepositoryStatus.COMPLETED)
        
        # Test filtering by status
        pending_repos = await repo_repo.list_repositories(status=RepositoryStatus.PENDING)
        assert any(repo.id == pending_repo.id for repo in pending_repos)
        assert all(repo.status == RepositoryStatus.PENDING for repo in pending_repos)
        
        completed_repos = await repo_repo.list_repositories(status=RepositoryStatus.COMPLETED)
        assert any(repo.id == completed_repo.id for repo in completed_repos)
        assert all(repo.status == RepositoryStatus.COMPLETED for repo in completed_repos)

    @pytest.mark.asyncio
    async def test_delete_repository(self, async_session: AsyncSession):
        """Test deleting a repository."""
        repo_repo = RepositoryRepo(async_session)
        
        # Create repository
        repo = await repo_repo.create(name="delete-test", url="https://github.com/test/delete")
        repo_id = repo.id
        
        # Delete repository
        deleted = await repo_repo.delete(repo_id)
        assert deleted is True
        
        # Verify deletion
        found_repo = await repo_repo.get_by_id(repo_id)
        assert found_repo is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_repository(self, async_session: AsyncSession):
        """Test deleting nonexistent repository."""
        repo_repo = RepositoryRepo(async_session)
        
        nonexistent_id = uuid.uuid4()
        deleted = await repo_repo.delete(nonexistent_id)
        assert deleted is False


class TestCodeSnippetRepo:
    """Test CodeSnippetRepo operations."""

    @pytest.mark.asyncio
    async def test_create_code_snippet(self, async_session: AsyncSession, test_repository):
        """Test creating a code snippet."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        snippet_data = {
            "repository_id": test_repository.id,
            "file_path": "src/example.py",
            "language": "python",
            "content": "def hello():\n    return 'world'",
            "embedding": [0.1] * 1536,
            "start_line": 1,
            "end_line": 2,
            "complexity_score": 1.5
        }
        
        snippet = await snippet_repo.create(**snippet_data)
        
        assert snippet.id is not None
        assert snippet.repository_id == test_repository.id
        assert snippet.file_path == "src/example.py"
        assert snippet.language == "python"
        assert snippet.content == "def hello():\n    return 'world'"
        assert len(snippet.embedding) == 1536
        assert snippet.start_line == 1
        assert snippet.end_line == 2
        assert snippet.complexity_score == 1.5

    @pytest.mark.asyncio
    async def test_create_batch_code_snippets(self, async_session: AsyncSession, test_repository):
        """Test creating multiple code snippets in batch."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        batch_data = []
        for i in range(3):
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
        
        created_snippets = await snippet_repo.create_batch(batch_data)
        
        assert len(created_snippets) == 3
        for i, snippet in enumerate(created_snippets):
            assert snippet.file_path == f"src/file_{i}.py"
            assert snippet.complexity_score == float(i + 1)

    @pytest.mark.asyncio
    async def test_get_snippet_by_id(self, async_session: AsyncSession, test_code_snippet):
        """Test retrieving code snippet by ID."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        found_snippet = await snippet_repo.get_by_id(test_code_snippet.id)
        
        assert found_snippet is not None
        assert found_snippet.id == test_code_snippet.id
        assert found_snippet.content == test_code_snippet.content

    @pytest.mark.asyncio
    async def test_get_snippets_by_repository_id(
        self, async_session: AsyncSession, test_repository
    ):
        """Test retrieving snippets by repository ID."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create multiple snippets for the repository
        for i in range(3):
            await snippet_repo.create(
                repository_id=test_repository.id,
                file_path=f"src/file_{i}.py",
                language="python",
                content=f"def func_{i}(): pass",
                embedding=[0.1] * 1536,
                start_line=1,
                end_line=1,
                complexity_score=1.0
            )
        
        # Retrieve snippets
        snippets = await snippet_repo.get_by_repository_id(test_repository.id)
        
        assert len(snippets) >= 3
        assert all(snippet.repository_id == test_repository.id for snippet in snippets)

    @pytest.mark.asyncio
    async def test_get_snippets_by_language(self, async_session: AsyncSession, test_repository):
        """Test retrieving snippets by programming language."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create snippets in different languages
        await snippet_repo.create(
            repository_id=test_repository.id,
            file_path="src/script.py",
            language="python",
            content="print('python')",
            embedding=[0.1] * 1536,
            start_line=1,
            end_line=1,
            complexity_score=1.0
        )
        
        await snippet_repo.create(
            repository_id=test_repository.id,
            file_path="src/script.js",
            language="javascript",
            content="console.log('javascript');",
            embedding=[0.2] * 1536,
            start_line=1,
            end_line=1,
            complexity_score=1.0
        )
        
        # Retrieve Python snippets
        python_snippets = await snippet_repo.get_by_language("python", limit=10)
        assert len(python_snippets) >= 1
        assert all(snippet.language == "python" for snippet in python_snippets)
        
        # Retrieve JavaScript snippets
        js_snippets = await snippet_repo.get_by_language("javascript", limit=10)
        assert len(js_snippets) >= 1
        assert all(snippet.language == "javascript" for snippet in js_snippets)

    @pytest.mark.asyncio
    async def test_get_snippets_by_file_path(
        self, async_session: AsyncSession, test_code_snippet
    ):
        """Test retrieving snippets by file path pattern."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create snippets with different file paths
        await snippet_repo.create(
            repository_id=test_code_snippet.repository_id,
            file_path="src/utils/helpers.py",
            language="python",
            content="def helper(): pass",
            embedding=[0.1] * 1536,
            start_line=1,
            end_line=1,
            complexity_score=1.0
        )
        
        await snippet_repo.create(
            repository_id=test_code_snippet.repository_id,
            file_path="tests/test_helpers.py",
            language="python",
            content="def test_helper(): pass",
            embedding=[0.2] * 1536,
            start_line=1,
            end_line=1,
            complexity_score=1.0
        )
        
        # Search by file path pattern
        src_snippets = await snippet_repo.get_by_file_path_pattern("src/%")
        assert len(src_snippets) >= 1
        assert all("src/" in snippet.file_path for snippet in src_snippets)
        
        test_snippets = await snippet_repo.get_by_file_path_pattern("tests/%")
        assert len(test_snippets) >= 1
        assert all("tests/" in snippet.file_path for snippet in test_snippets)

    @pytest.mark.asyncio
    async def test_update_snippet_embedding(
        self, async_session: AsyncSession, test_code_snippet
    ):
        """Test updating snippet embedding."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        new_embedding = [0.9] * 1536
        updated_snippet = await snippet_repo.update_embedding(
            test_code_snippet.id, new_embedding
        )
        
        assert updated_snippet.embedding == new_embedding
        
        # Verify in database
        found_snippet = await snippet_repo.get_by_id(test_code_snippet.id)
        assert found_snippet.embedding == new_embedding

    @pytest.mark.asyncio
    async def test_delete_snippet(self, async_session: AsyncSession, test_code_snippet):
        """Test deleting a code snippet."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        snippet_id = test_code_snippet.id
        deleted = await snippet_repo.delete(snippet_id)
        assert deleted is True
        
        # Verify deletion
        found_snippet = await snippet_repo.get_by_id(snippet_id)
        assert found_snippet is None

    @pytest.mark.asyncio
    async def test_search_snippets_by_content(
        self, async_session: AsyncSession, test_repository
    ):
        """Test searching snippets by content."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create snippets with searchable content
        await snippet_repo.create(
            repository_id=test_repository.id,
            file_path="src/auth.py",
            language="python",
            content="def authenticate_user(username, password):\n    return check_credentials(username, password)",
            embedding=[0.1] * 1536,
            start_line=1,
            end_line=2,
            complexity_score=2.0
        )
        
        await snippet_repo.create(
            repository_id=test_repository.id,
            file_path="src/math.py",
            language="python",
            content="def calculate_sum(numbers):\n    return sum(numbers)",
            embedding=[0.2] * 1536,
            start_line=1,
            end_line=2,
            complexity_score=1.0
        )
        
        # Search by content
        auth_snippets = await snippet_repo.search_by_content("authenticate")
        assert len(auth_snippets) >= 1
        assert any("authenticate" in snippet.content for snippet in auth_snippets)
        
        math_snippets = await snippet_repo.search_by_content("calculate")
        assert len(math_snippets) >= 1
        assert any("calculate" in snippet.content for snippet in math_snippets)

    @pytest.mark.asyncio
    async def test_get_snippets_with_high_complexity(
        self, async_session: AsyncSession, test_repository
    ):
        """Test retrieving snippets with high complexity scores."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create snippets with different complexity scores
        await snippet_repo.create(
            repository_id=test_repository.id,
            file_path="src/simple.py",
            language="python",
            content="def simple(): return 1",
            embedding=[0.1] * 1536,
            start_line=1,
            end_line=1,
            complexity_score=1.0
        )
        
        await snippet_repo.create(
            repository_id=test_repository.id,
            file_path="src/complex.py",
            language="python",
            content="def complex_function(): # ... complex logic",
            embedding=[0.2] * 1536,
            start_line=1,
            end_line=10,
            complexity_score=8.5
        )
        
        # Get high complexity snippets
        complex_snippets = await snippet_repo.get_by_complexity_threshold(5.0)
        assert len(complex_snippets) >= 1
        assert all(snippet.complexity_score >= 5.0 for snippet in complex_snippets)


class TestAnalysisResultRepo:
    """Test AnalysisResultRepo operations."""

    @pytest.mark.asyncio
    async def test_create_analysis_result(
        self, async_session: AsyncSession, test_code_snippet
    ):
        """Test creating an analysis result."""
        result_repo = AnalysisResultRepo(async_session)
        
        recommendations = [
            {
                "type": "style",
                "message": "Add type hints",
                "confidence": 0.8
            },
            {
                "type": "performance",
                "message": "Use list comprehension",
                "confidence": 0.9
            }
        ]
        
        similar_patterns = [
            {
                "snippet_id": str(uuid.uuid4()),
                "similarity_score": 0.95
            }
        ]
        
        result = await result_repo.create(
            snippet_id=test_code_snippet.id,
            recommendations=recommendations,
            similar_patterns=similar_patterns,
            quality_score=85.5
        )
        
        assert result.id is not None
        assert result.snippet_id == test_code_snippet.id
        assert len(result.recommendations) == 2
        assert len(result.similar_patterns) == 1
        assert result.quality_score == 85.5

    @pytest.mark.asyncio
    async def test_get_analysis_result_by_snippet_id(
        self, async_session: AsyncSession, test_analysis_result
    ):
        """Test retrieving analysis result by snippet ID."""
        result_repo = AnalysisResultRepo(async_session)
        
        found_result = await result_repo.get_by_snippet_id(
            test_analysis_result.snippet_id
        )
        
        assert found_result is not None
        assert found_result.id == test_analysis_result.id
        assert found_result.snippet_id == test_analysis_result.snippet_id

    @pytest.mark.asyncio
    async def test_update_analysis_result(
        self, async_session: AsyncSession, test_analysis_result
    ):
        """Test updating an analysis result."""
        result_repo = AnalysisResultRepo(async_session)
        
        new_recommendations = [
            {
                "type": "security",
                "message": "Validate input parameters",
                "confidence": 0.95
            }
        ]
        
        updated_result = await result_repo.update(
            test_analysis_result.id,
            recommendations=new_recommendations,
            quality_score=92.0
        )
        
        assert len(updated_result.recommendations) == 1
        assert updated_result.recommendations[0]["type"] == "security"
        assert updated_result.quality_score == 92.0

    @pytest.mark.asyncio
    async def test_get_results_by_quality_score_range(
        self, async_session: AsyncSession, test_code_snippet
    ):
        """Test retrieving results by quality score range."""
        result_repo = AnalysisResultRepo(async_session)
        
        # Create results with different quality scores
        await result_repo.create(
            snippet_id=test_code_snippet.id,
            recommendations=[],
            similar_patterns=[],
            quality_score=95.0
        )
        
        # Create another snippet for variety
        snippet_repo = CodeSnippetRepo(async_session)
        another_snippet = await snippet_repo.create(
            repository_id=test_code_snippet.repository_id,
            file_path="src/other.py",
            language="python",
            content="def other(): pass",
            embedding=[0.5] * 1536,
            start_line=1,
            end_line=1,
            complexity_score=1.0
        )
        
        await result_repo.create(
            snippet_id=another_snippet.id,
            recommendations=[],
            similar_patterns=[],
            quality_score=70.0
        )
        
        # Get high quality results
        high_quality = await result_repo.get_by_quality_score_range(90.0, 100.0)
        assert len(high_quality) >= 1
        assert all(result.quality_score >= 90.0 for result in high_quality)
        
        # Get medium quality results
        medium_quality = await result_repo.get_by_quality_score_range(60.0, 90.0)
        assert len(medium_quality) >= 1
        assert all(60.0 <= result.quality_score < 90.0 for result in medium_quality)

    @pytest.mark.asyncio
    async def test_get_recent_analysis_results(
        self, async_session: AsyncSession, test_code_snippet
    ):
        """Test retrieving recent analysis results."""
        result_repo = AnalysisResultRepo(async_session)
        
        # Create recent result
        recent_result = await result_repo.create(
            snippet_id=test_code_snippet.id,
            recommendations=[],
            similar_patterns=[],
            quality_score=80.0
        )
        
        # Get results from last 24 hours
        since = datetime.utcnow() - timedelta(hours=24)
        recent_results = await result_repo.get_recent_results(since=since)
        
        assert len(recent_results) >= 1
        assert any(result.id == recent_result.id for result in recent_results)

    @pytest.mark.asyncio  
    async def test_delete_analysis_result(
        self, async_session: AsyncSession, test_analysis_result
    ):
        """Test deleting an analysis result."""
        result_repo = AnalysisResultRepo(async_session)
        
        result_id = test_analysis_result.id
        deleted = await result_repo.delete(result_id)
        assert deleted is True
        
        # Verify deletion
        found_result = await result_repo.get_by_id(result_id)
        assert found_result is None

    @pytest.mark.asyncio
    async def test_cascade_delete_with_snippet(
        self, async_session: AsyncSession, test_repository
    ):
        """Test cascading delete when snippet is deleted."""
        snippet_repo = CodeSnippetRepo(async_session)
        result_repo = AnalysisResultRepo(async_session)
        
        # Create snippet and result
        snippet = await snippet_repo.create(
            repository_id=test_repository.id,
            file_path="src/cascade_test.py",
            language="python",
            content="def test(): pass",
            embedding=[0.1] * 1536,
            start_line=1,
            end_line=1,
            complexity_score=1.0
        )
        
        result = await result_repo.create(
            snippet_id=snippet.id,
            recommendations=[],
            similar_patterns=[],
            quality_score=80.0
        )
        
        # Delete snippet (should cascade to result)
        await snippet_repo.delete(snippet.id)
        
        # Verify result is also deleted
        found_result = await result_repo.get_by_id(result.id)
        assert found_result is None


class TestRepositoryTransactions:
    """Test transaction handling in repository operations."""

    @pytest.mark.asyncio
    async def test_transaction_rollback_on_error(self, async_session: AsyncSession):
        """Test that transactions are rolled back on errors."""
        repo_repo = RepositoryRepo(async_session)
        
        initial_count = len(await repo_repo.list_repositories())
        
        try:
            async with async_session.begin():
                # Create repository
                repo = await repo_repo.create(
                    name="transaction-test",
                    url="https://github.com/test/transaction"
                )
                
                # Force an error (simulate constraint violation)
                await async_session.execute(
                    "INSERT INTO repositories (id, name) VALUES (?, ?)",
                    [str(repo.id), "duplicate-id"]  # This should fail due to duplicate ID
                )
        except Exception:
            # Exception expected due to constraint violation
            pass
        
        # Verify rollback - count should be unchanged
        final_count = len(await repo_repo.list_repositories())
        assert final_count == initial_count

    @pytest.mark.asyncio
    async def test_batch_operations_transaction(
        self, async_session: AsyncSession, test_repository
    ):
        """Test batch operations within transactions."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        batch_data = []
        for i in range(5):
            batch_data.append({
                "repository_id": test_repository.id,
                "file_path": f"src/batch_{i}.py",
                "language": "python",
                "content": f"def batch_func_{i}(): pass",
                "embedding": [0.1 * i] * 1536,
                "start_line": 1,
                "end_line": 1,
                "complexity_score": float(i + 1)
            })
        
        # Perform batch operation
        created_snippets = await snippet_repo.create_batch(batch_data)
        
        # Verify all created
        assert len(created_snippets) == 5
        
        # Verify all are in database
        repo_snippets = await snippet_repo.get_by_repository_id(test_repository.id)
        batch_ids = {snippet.id for snippet in created_snippets}
        found_ids = {snippet.id for snippet in repo_snippets if snippet.id in batch_ids}
        
        assert len(found_ids) == 5


class TestRepositoryPerformance:
    """Test repository performance and optimization."""

    @pytest.mark.asyncio
    async def test_large_batch_insert_performance(
        self, async_session: AsyncSession, test_repository
    ):
        """Test performance of large batch inserts."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create large batch
        batch_size = 100
        batch_data = []
        for i in range(batch_size):
            batch_data.append({
                "repository_id": test_repository.id,
                "file_path": f"src/perf_test_{i}.py",
                "language": "python",
                "content": f"def perf_func_{i}(): return {i}",
                "embedding": [0.1 * (i % 10)] * 1536,
                "start_line": 1,
                "end_line": 2,
                "complexity_score": float(i % 5 + 1)
            })
        
        import time
        start_time = time.time()
        
        # Perform batch insert
        created_snippets = await snippet_repo.create_batch(batch_data)
        
        end_time = time.time()
        insert_time = end_time - start_time
        
        # Verify results and performance
        assert len(created_snippets) == batch_size
        assert insert_time < 5.0  # Should complete within 5 seconds
        
        # Verify data integrity
        sample_snippet = created_snippets[50]  # Check middle item
        assert sample_snippet.file_path == "src/perf_test_50.py"
        assert sample_snippet.complexity_score == 1.0  # 50 % 5 + 1 = 1

    @pytest.mark.asyncio
    async def test_pagination_performance(
        self, async_session: AsyncSession, test_repository
    ):
        """Test pagination performance with large datasets."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create test data
        batch_data = []
        for i in range(50):
            batch_data.append({
                "repository_id": test_repository.id,
                "file_path": f"src/pagination_{i}.py",
                "language": "python",
                "content": f"def func_{i}(): pass",
                "embedding": [0.1] * 1536,
                "start_line": 1,
                "end_line": 1,
                "complexity_score": 1.0
            })
        
        await snippet_repo.create_batch(batch_data)
        
        # Test pagination
        page_size = 10
        all_snippets = []
        
        for page in range(5):  # 50 items / 10 per page = 5 pages
            page_snippets = await snippet_repo.get_by_repository_id(
                test_repository.id,
                skip=page * page_size,
                limit=page_size
            )
            all_snippets.extend(page_snippets)
        
        # Verify pagination
        assert len(all_snippets) >= 50
        
        # Verify no duplicates
        snippet_ids = [snippet.id for snippet in all_snippets]
        assert len(snippet_ids) == len(set(snippet_ids))

    @pytest.mark.asyncio
    async def test_search_query_performance(
        self, async_session: AsyncSession, test_repository
    ):
        """Test search query performance."""
        snippet_repo = CodeSnippetRepo(async_session)
        
        # Create searchable test data
        search_terms = ["authenticate", "calculate", "validate", "process", "handle"]
        
        for i, term in enumerate(search_terms):
            for j in range(10):  # 10 snippets per term
                await snippet_repo.create(
                    repository_id=test_repository.id,
                    file_path=f"src/{term}_{j}.py",
                    language="python",
                    content=f"def {term}_data_{j}(): # {term} function logic here",
                    embedding=[0.1 * i] * 1536,
                    start_line=1,
                    end_line=1,
                    complexity_score=1.0
                )
        
        import time
        
        # Test search performance
        for term in search_terms:
            start_time = time.time()
            
            results = await snippet_repo.search_by_content(term)
            
            end_time = time.time()
            search_time = end_time - start_time
            
            # Verify results and performance
            assert len(results) >= 10  # Should find at least 10 results
            assert search_time < 1.0   # Should complete within 1 second
            assert all(term in result.content for result in results)