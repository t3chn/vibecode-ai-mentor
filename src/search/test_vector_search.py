"""Tests for vector search functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timedelta

from src.search.vector_search import (
    VectorSearchService,
    CodeMatch,
    SearchFilters,
)


class TestCodeMatch:
    """Test CodeMatch class."""
    
    def test_code_match_creation(self):
        """Test CodeMatch object creation."""
        match = CodeMatch(
            snippet_id=uuid4(),
            repository_id=uuid4(),
            repository_name="test-repo",
            file_path="src/test.py",
            language="python",
            content="def hello(): pass",
            start_line=1,
            end_line=2,
            similarity_score=0.85,
            complexity_score=2.5,
        )
        
        assert match.language == "python"
        assert match.similarity_score == 0.85
        assert match.complexity_score == 2.5
    
    def test_code_match_serialization(self):
        """Test CodeMatch to/from dict conversion."""
        original = CodeMatch(
            snippet_id=uuid4(),
            repository_id=uuid4(),
            repository_name="test-repo",
            file_path="src/test.py",
            language="python",
            content="def hello(): pass",
            start_line=1,
            end_line=2,
            similarity_score=0.85,
        )
        
        # Convert to dict and back
        data = original.to_dict()
        restored = CodeMatch.from_dict(data)
        
        assert restored.snippet_id == original.snippet_id
        assert restored.repository_id == original.repository_id
        assert restored.language == original.language
        assert restored.similarity_score == original.similarity_score


class TestSearchFilters:
    """Test SearchFilters class."""
    
    def test_search_filters_creation(self):
        """Test SearchFilters creation with various types."""
        repo_uuid = uuid4()
        
        filters = SearchFilters(
            language="python",
            repository_id=repo_uuid,
            min_complexity=1.0,
            max_complexity=5.0,
            file_extension="py",
            exclude_repositories=[uuid4(), "string-id"],
        )
        
        assert filters.language == "python"
        assert filters.repository_id == str(repo_uuid)
        assert filters.min_complexity == 1.0
        assert len(filters.exclude_repositories) == 2


@pytest.fixture
def mock_session():
    """Create mock async session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def mock_embedding_manager():
    """Create mock embedding manager."""
    manager = MagicMock()
    manager.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    manager.generate_embeddings_batch = AsyncMock(return_value=[[0.1] * 1536])
    manager.preprocess_code = MagicMock(return_value="processed code")
    manager.dimensions = 1536
    manager.model_name = "test-model"
    return manager


@pytest.fixture
def vector_search_service(mock_session, mock_embedding_manager):
    """Create VectorSearchService with mocked dependencies."""
    with patch('src.search.vector_search.get_embedding_manager', return_value=mock_embedding_manager):
        with patch('src.search.vector_search.get_settings') as mock_settings:
            mock_settings.return_value = MagicMock()
            service = VectorSearchService(mock_session)
            return service


class TestVectorSearchService:
    """Test VectorSearchService functionality."""
    
    @pytest.mark.asyncio
    async def test_search_similar_code_basic(self, vector_search_service, mock_session):
        """Test basic vector similarity search."""
        # Mock database response
        mock_row = MagicMock()
        mock_row.id = str(uuid4())
        mock_row.repository_id = str(uuid4())
        mock_row.repository_name = "test-repo"
        mock_row.file_path = "test.py"
        mock_row.language = "python"
        mock_row.content = "def test(): pass"
        mock_row.start_line = 1
        mock_row.end_line = 2
        mock_row.similarity = 0.8
        mock_row.complexity_score = 2.0
        mock_row.repository_url = "https://github.com/test/repo"
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        
        # Test search
        query_embedding = [0.1] * 1536
        results = await vector_search_service.search_similar_code(query_embedding)
        
        assert len(results) == 1
        assert results[0].language == "python"
        assert results[0].similarity_score == 0.8
        assert results[0].repository_name == "test-repo"
    
    @pytest.mark.asyncio
    async def test_search_similar_code_with_filters(self, vector_search_service, mock_session):
        """Test vector search with filters."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        filters = SearchFilters(
            language="python",
            min_complexity=1.0,
            max_complexity=5.0,
        )
        
        query_embedding = [0.1] * 1536
        results = await vector_search_service.search_similar_code(
            query_embedding, filters=filters, limit=5
        )
        
        # Verify execute was called with correct parameters
        mock_session.execute.assert_called_once()
        call_args = mock_session.execute.call_args
        
        # Check that parameters include filter values
        params = call_args[0][1]  # Second argument is parameters
        assert params["language"] == "python"
        assert params["min_complexity"] == 1.0
        assert params["max_complexity"] == 5.0
        assert params["limit"] == 5
    
    @pytest.mark.asyncio
    async def test_search_by_text(self, vector_search_service, mock_session):
        """Test text-based search."""
        # Mock successful search
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        # Mock cache miss
        mock_cache_result = MagicMock()
        mock_cache_result.fetchone.return_value = None
        mock_session.execute.side_effect = [mock_cache_result, mock_result, MagicMock()]
        
        results = await vector_search_service.search_by_text(
            "def hello_world():", language="python"
        )
        
        # Verify embedding generation was called
        vector_search_service.embedding_manager.preprocess_code.assert_called_once()
        vector_search_service.embedding_manager.generate_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_similar_functions(self, vector_search_service, mock_session):
        """Test function similarity search."""
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result
        
        function_code = "def calculate_sum(a, b): return a + b"
        results = await vector_search_service.search_similar_functions(
            function_code, threshold=0.9, language="python"
        )
        
        # Verify preprocessing and embedding generation
        vector_search_service.embedding_manager.preprocess_code.assert_called_once_with(function_code)
        vector_search_service.embedding_manager.generate_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_search_anti_patterns(self, vector_search_service, mock_session):
        """Test anti-pattern detection."""
        # Mock cache miss and search result
        mock_cache_result = MagicMock()
        mock_cache_result.fetchone.return_value = None
        
        mock_search_result = MagicMock()
        mock_search_result.fetchall.return_value = []
        
        mock_cache_insert = MagicMock()
        
        mock_session.execute.side_effect = [
            mock_cache_result,  # Cache lookup
            mock_search_result,  # Vector search
            mock_cache_insert   # Cache insert
        ]
        
        results = await vector_search_service.search_anti_patterns(
            "god_object", threshold=0.7, language="python"
        )
        
        # Verify embedding was generated for anti-pattern template
        vector_search_service.embedding_manager.generate_embedding.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_duplicate_code(self, vector_search_service, mock_session):
        """Test duplicate code detection."""
        # Mock database response with duplicate pairs
        mock_row = MagicMock()
        mock_row.id1 = str(uuid4())
        mock_row.id2 = str(uuid4())
        mock_row.repository_id = str(uuid4())
        mock_row.path1 = "file1.py"
        mock_row.path2 = "file2.py"
        mock_row.content1 = "def func(): pass"
        mock_row.content2 = "def func(): pass"
        mock_row.start1 = 1
        mock_row.end1 = 2
        mock_row.start2 = 10
        mock_row.end2 = 11
        mock_row.similarity = 0.05  # Very similar (low distance)
        mock_row.language = "python"
        
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [mock_row]
        mock_session.execute.return_value = mock_result
        
        duplicate_pairs = await vector_search_service.find_duplicate_code(
            similarity_threshold=0.95, min_lines=3
        )
        
        assert len(duplicate_pairs) == 1
        pair1, pair2 = duplicate_pairs[0]
        assert pair1.file_path == "file1.py"
        assert pair2.file_path == "file2.py"
        assert pair1.similarity_score == 0.05
    
    @pytest.mark.asyncio
    async def test_get_code_recommendations(self, vector_search_service, mock_session):
        """Test code recommendation generation."""
        # Mock similar code search
        snippet_id = uuid4()
        mock_search_row = MagicMock()
        mock_search_row.id = str(snippet_id)
        mock_search_row.repository_id = str(uuid4())
        mock_search_row.repository_name = "test-repo"
        mock_search_row.file_path = "test.py"
        mock_search_row.language = "python"
        mock_search_row.content = "def test(): pass"
        mock_search_row.start_line = 1
        mock_search_row.end_line = 2
        mock_search_row.similarity = 0.8
        mock_search_row.complexity_score = 2.0
        mock_search_row.repository_url = None
        
        # Mock recommendations query
        mock_rec_row = MagicMock()
        mock_rec_row.recommendations = [{"type": "improvement", "message": "Add docstring"}]
        mock_rec_row.quality_score = 8.5
        mock_rec_row.content = "def test(): pass"
        
        mock_session.execute.side_effect = [
            # Cache lookup (miss)
            MagicMock(fetchone=MagicMock(return_value=None)),
            # Similar code search
            MagicMock(fetchall=MagicMock(return_value=[mock_search_row])),
            # Cache insert
            MagicMock(),
            # Recommendations query
            MagicMock(fetchall=MagicMock(return_value=[mock_rec_row]))
        ]
        
        similar_examples, recommendations = await vector_search_service.get_code_recommendations(
            "def test(): pass", language="python"
        )
        
        assert len(similar_examples) == 1
        assert len(recommendations) == 1
        assert recommendations[0]["type"] == "improvement"
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, vector_search_service, mock_session):
        """Test result caching."""
        # Test cache hit
        cached_data = [{
            "snippet_id": str(uuid4()),
            "repository_id": str(uuid4()),
            "repository_name": "test-repo",
            "file_path": "test.py",
            "language": "python",
            "content": "def test(): pass",
            "start_line": 1,
            "end_line": 2,
            "similarity_score": 0.8,
            "complexity_score": 2.0,
            "repository_url": None,
        }]
        
        mock_cache_result = MagicMock()
        mock_cache_row = MagicMock()
        mock_cache_row.results = cached_data
        mock_cache_result.fetchone.return_value = mock_cache_row
        mock_session.execute.return_value = mock_cache_result
        
        results = await vector_search_service.search_by_text(
            "def test():", language="python", use_cache=True
        )
        
        assert len(results) == 1
        assert results[0].language == "python"
        # Should not call embedding generation due to cache hit
        vector_search_service.embedding_manager.generate_embedding.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_cache(self, vector_search_service, mock_session):
        """Test cache cleanup functionality."""
        # Mock deletion results
        mock_search_result = MagicMock()
        mock_search_result.rowcount = 5
        
        mock_embedding_result = MagicMock()
        mock_embedding_result.rowcount = 3
        
        mock_session.execute.side_effect = [mock_search_result, mock_embedding_result]
        
        deleted_count = await vector_search_service.cleanup_expired_cache()
        
        assert deleted_count == 8  # 5 + 3
        assert mock_session.commit.called
    
    @pytest.mark.asyncio
    async def test_get_search_stats(self, vector_search_service, mock_session):
        """Test search statistics retrieval."""
        # Mock statistics queries
        mock_total = MagicMock()
        mock_total.total = 1000
        
        mock_lang_row1 = MagicMock()
        mock_lang_row1.language = "python"
        mock_lang_row1.count = 600
        
        mock_lang_row2 = MagicMock()
        mock_lang_row2.language = "javascript"
        mock_lang_row2.count = 400
        
        mock_cached = MagicMock()
        mock_cached.cached = 50
        
        mock_session.execute.side_effect = [
            MagicMock(fetchone=MagicMock(return_value=mock_total)),
            MagicMock(fetchall=MagicMock(return_value=[mock_lang_row1, mock_lang_row2])),
            MagicMock(fetchone=MagicMock(return_value=mock_cached)),
        ]
        
        stats = await vector_search_service.get_search_stats()
        
        assert stats["total_indexed_snippets"] == 1000
        assert stats["snippets_by_language"]["python"] == 600
        assert stats["snippets_by_language"]["javascript"] == 400
        assert stats["cached_queries"] == 50
        assert stats["embedding_dimensions"] == 1536
        assert stats["model_name"] == "test-model"
    
    def test_language_detection(self, vector_search_service):
        """Test programming language detection."""
        python_code = "def hello(): import os"
        assert vector_search_service._detect_language(python_code) == "python"
        
        js_code = "function hello() { const x = 1; }"
        assert vector_search_service._detect_language(js_code) == "javascript"
        
        java_code = "public class Hello { private int x; }"
        assert vector_search_service._detect_language(java_code) == "java"
        
        cpp_code = "#include <iostream>\nint main() { return 0; }"
        assert vector_search_service._detect_language(cpp_code) == "cpp"
        
        unknown_code = "some random text"
        assert vector_search_service._detect_language(unknown_code) == "python"  # Default
    
    def test_build_filter_clause(self, vector_search_service):
        """Test SQL filter clause building."""
        filters = SearchFilters(
            language="python",
            repository_id="test-repo",
            min_complexity=1.0,
            max_complexity=5.0,
            file_extension="py",
            exclude_repositories=["repo1", "repo2"],
        )
        
        params = {}
        filter_clause = vector_search_service._build_filter_clause(filters, params)
        
        assert "AND language = :language" in filter_clause
        assert "AND repository_id = :repository_id" in filter_clause
        assert "AND complexity_score >= :min_complexity" in filter_clause
        assert "AND complexity_score <= :max_complexity" in filter_clause
        assert "AND file_path LIKE :file_extension" in filter_clause
        assert "AND repository_id NOT IN" in filter_clause
        
        assert params["language"] == "python"
        assert params["repository_id"] == "test-repo"
        assert params["min_complexity"] == 1.0
        assert params["max_complexity"] == 5.0
        assert params["file_extension"] == "%.py"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])