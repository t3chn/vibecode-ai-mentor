"""Shared pytest fixtures for VibeCode AI Mentor tests."""

import asyncio
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from src.api.app import app
from src.core.config import Settings
from src.db.models import Base, Repository, CodeSnippet, AnalysisResult, RepositoryStatus
from src.embeddings.base import EmbeddingProvider
from src.generator.llm_client import LLMClient
from src.analyzer.parser import PythonParser
from src.analyzer.chunker import CodeChunker
from src.analyzer.metrics import CodeMetrics


# Test configuration
@pytest.fixture(scope="session")
def test_settings():
    """Test configuration settings."""
    return Settings(
        environment="test",
        debug=True,
        tidb_host="localhost",
        tidb_port=4000,
        tidb_user="test",
        tidb_password="test",
        tidb_database="test_vibecode",
        api_key="test-api-key",
        gemini_api_key="test-gemini-key",
        openai_api_key="test-openai-key",
        log_level="DEBUG",
        batch_size=10,
        max_chunk_size=2048,
        min_chunk_size=128,
    )


# Database fixtures
@pytest.fixture(scope="session")
def engine():
    """Create test database engine."""
    # Use SQLite for testing to avoid TiDB dependency
    engine = create_engine(
        "sqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    
    # Create all tables
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture(scope="session")
def async_engine():
    """Create async test database engine."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )
    return engine


@pytest_asyncio.fixture
async def async_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create async database session for testing."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async_session_maker = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session


@pytest.fixture
def sync_session(engine):
    """Create sync database session for testing."""
    from sqlalchemy.orm import sessionmaker
    
    session_maker = sessionmaker(bind=engine)
    session = session_maker()
    
    try:
        yield session
    finally:
        session.close()


# API Test Client fixtures
@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


# Mock fixtures
@pytest.fixture
def mock_embedding_provider():
    """Mock embedding provider."""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.generate_embedding = AsyncMock(return_value=[0.1] * 1536)
    provider.generate_embeddings = AsyncMock(return_value=[[0.1] * 1536])
    provider.health_check = MagicMock(return_value=True)
    return provider


@pytest.fixture
def mock_llm_client():
    """Mock LLM client."""
    client = MagicMock(spec=LLMClient)
    client.generate_recommendations = AsyncMock(return_value={
        "recommendations": [
            {
                "type": "improvement",
                "severity": "warning",
                "message": "Consider using list comprehension",
                "suggestion": "Use [x for x in items] instead of map()",
                "line_start": 5,
                "line_end": 7,
                "confidence": 0.8
            }
        ],
        "summary": "Code quality analysis completed",
        "overall_score": 85
    })
    client.health_check = AsyncMock(return_value={"status": "healthy"})
    return client


# Sample data fixtures
@pytest.fixture
def sample_python_code():
    """Sample Python code for testing."""
    return '''
import os
import sys
from typing import List, Optional

def calculate_average(numbers: List[float]) -> Optional[float]:
    """Calculate the average of a list of numbers."""
    if not numbers:
        return None
    return sum(numbers) / len(numbers)

class DataProcessor:
    """Process data with various operations."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
    
    def add_data(self, value: float) -> None:
        """Add a data point."""
        self.data.append(value)
    
    def get_average(self) -> Optional[float]:
        """Get average of all data points."""
        return calculate_average(self.data)
    
    def process_batch(self, values: List[float]) -> List[float]:
        """Process a batch of values."""
        results = []
        for value in values:
            if value > 0:
                results.append(value * 2)
            else:
                results.append(0)
        return results

# Global configuration
DEBUG = True
MAX_ITEMS = 1000
'''


@pytest.fixture
def sample_code_chunks():
    """Sample code chunks for testing."""
    return [
        {
            "content": "def calculate_average(numbers: List[float]) -> Optional[float]:",
            "start_line": 5,
            "end_line": 5,
            "chunk_type": "function_definition",
            "complexity": 2
        },
        {
            "content": "class DataProcessor:\n    def __init__(self, name: str):",
            "start_line": 10,
            "end_line": 12,
            "chunk_type": "class_definition",
            "complexity": 1
        },
        {
            "content": "def process_batch(self, values: List[float]) -> List[float]:",
            "start_line": 20,
            "end_line": 27,
            "chunk_type": "method_definition",
            "complexity": 3
        }
    ]


@pytest.fixture
def sample_analysis_result():
    """Sample analysis result for testing."""
    return {
        "recommendations": [
            {
                "type": "performance",
                "severity": "info",
                "message": "Consider using list comprehension for better performance",
                "suggestion": "Replace loop with [value * 2 if value > 0 else 0 for value in values]",
                "line_start": 23,
                "line_end": 27,
                "confidence": 0.9
            }
        ],
        "refactoring_suggestions": [
            {
                "refactor_type": "extract_method",
                "description": "Extract validation logic to separate method",
                "confidence": 0.7
            }
        ],
        "anti_pattern_fixes": [],
        "summary": "Code follows good practices with minor optimization opportunities",
        "overall_score": 85
    }


# Database test data fixtures
@pytest.fixture
async def test_repository(async_session: AsyncSession):
    """Create test repository."""
    repo = Repository(
        id=uuid.uuid4(),
        name="test-repo",
        url="https://github.com/test/repo",
        status=RepositoryStatus.COMPLETED,
        total_files=10,
        last_indexed_at=datetime.utcnow()
    )
    async_session.add(repo)
    await async_session.commit()
    await async_session.refresh(repo)
    return repo


@pytest.fixture
async def test_code_snippet(async_session: AsyncSession, test_repository):
    """Create test code snippet."""
    snippet = CodeSnippet(
        id=uuid.uuid4(),
        repository_id=test_repository.id,
        file_path="src/example.py",
        language="python",
        content="def hello():\n    return 'world'",
        embedding=[0.1] * 1536,
        start_line=1,
        end_line=2,
        complexity_score=1.0
    )
    async_session.add(snippet)
    await async_session.commit()
    await async_session.refresh(snippet)
    return snippet


@pytest.fixture
async def test_analysis_result(async_session: AsyncSession, test_code_snippet):
    """Create test analysis result."""
    result = AnalysisResult(
        id=uuid.uuid4(),
        snippet_id=test_code_snippet.id,
        recommendations=[
            {
                "type": "style",
                "message": "Consider adding type hints",
                "confidence": 0.8
            }
        ],
        similar_patterns=[],
        quality_score=85.0
    )
    async_session.add(result)
    await async_session.commit()
    await async_session.refresh(result)
    return result


# Component fixtures
@pytest.fixture
def python_parser():
    """Create Python parser instance."""
    return PythonParser()


@pytest.fixture
def code_chunker():
    """Create code chunker instance."""
    return CodeChunker(
        min_chunk_size=128,
        max_chunk_size=2048,
        overlap_size=64
    )


@pytest.fixture
def code_metrics():
    """Create code metrics calculator."""
    return CodeMetrics()


# File system fixtures
@pytest.fixture
def temp_python_file(tmp_path, sample_python_code):
    """Create temporary Python file."""
    file_path = tmp_path / "test_file.py"
    file_path.write_text(sample_python_code)
    return file_path


@pytest.fixture
def temp_repository(tmp_path, sample_python_code):
    """Create temporary repository structure."""
    repo_path = tmp_path / "test_repo"
    repo_path.mkdir()
    
    # Create source files
    src_path = repo_path / "src"
    src_path.mkdir()
    
    (src_path / "main.py").write_text(sample_python_code)
    (src_path / "utils.py").write_text("""
def helper_function(x: int) -> int:
    return x * 2

class Helper:
    def process(self, data):
        return [helper_function(x) for x in data]
""")
    
    # Create test files
    tests_path = repo_path / "tests"
    tests_path.mkdir()
    
    (tests_path / "test_main.py").write_text("""
import pytest
from src.main import calculate_average

def test_calculate_average():
    assert calculate_average([1, 2, 3]) == 2.0
    assert calculate_average([]) is None
""")
    
    return repo_path


# Search test fixtures
@pytest.fixture
def sample_search_results():
    """Sample search results for testing."""
    return [
        {
            "snippet_id": str(uuid.uuid4()),
            "content": "def calculate_sum(numbers):\n    return sum(numbers)",
            "file_path": "src/math_utils.py",
            "similarity_score": 0.95,
            "line_start": 10,
            "line_end": 11
        },
        {
            "snippet_id": str(uuid.uuid4()),
            "content": "def process_list(items):\n    return [x for x in items if x > 0]",
            "file_path": "src/filters.py",
            "similarity_score": 0.87,
            "line_start": 5,
            "line_end": 6
        }
    ]


# Performance test fixtures
@pytest.fixture
def large_code_sample():
    """Large code sample for performance testing."""
    return '''
import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class Config:
    """Application configuration."""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    max_connections: int = 100

class DatabaseManager:
    """Manage database connections and operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.connections = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect(self) -> bool:
        """Establish database connection."""
        try:
            # Simulate connection logic
            await asyncio.sleep(0.1)
            self.connections["main"] = {"status": "connected"}
            self.logger.info("Database connected successfully")
            return True
        except Exception as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute database query."""
        if "main" not in self.connections:
            raise ConnectionError("Database not connected")
        
        # Simulate query execution
        await asyncio.sleep(0.05)
        
        # Mock result based on query type
        if query.startswith("SELECT"):
            return [{"id": 1, "name": "test"}, {"id": 2, "name": "example"}]
        elif query.startswith("INSERT"):
            return [{"rows_affected": 1}]
        else:
            return []
    
    async def close(self) -> None:
        """Close database connections."""
        self.connections.clear()
        self.logger.info("Database connections closed")

class APIService:
    """Main API service."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.cache = {}
        self.request_count = 0
    
    async def handle_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle API request."""
        self.request_count += 1
        
        # Check cache first
        cache_key = f"{endpoint}:{hash(str(sorted(data.items())))}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Process request based on endpoint
        if endpoint == "analyze":
            result = await self._analyze_code(data.get("code", ""))
        elif endpoint == "search":
            result = await self._search_patterns(data.get("query", ""))
        else:
            result = {"error": "Unknown endpoint"}
        
        # Cache result
        self.cache[cache_key] = result
        return result
    
    async def _analyze_code(self, code: str) -> Dict[str, Any]:
        """Analyze code quality."""
        # Simulate analysis
        await asyncio.sleep(0.2)
        
        lines = code.split("\\n")
        complexity = min(len(lines) // 10, 10)
        
        return {
            "status": "success",
            "lines_of_code": len(lines),
            "complexity_score": complexity,
            "recommendations": [
                "Consider adding type hints",
                "Add docstrings to functions",
                "Use consistent naming conventions"
            ][:complexity]
        }
    
    async def _search_patterns(self, query: str) -> Dict[str, Any]:
        """Search for code patterns."""
        # Simulate search
        await asyncio.sleep(0.1)
        
        results = []
        for i in range(min(len(query) // 5, 10)):
            results.append({
                "id": i + 1,
                "file": f"src/module_{i}.py",
                "line": (i + 1) * 10,
                "similarity": 0.9 - (i * 0.05)
            })
        
        return {
            "status": "success",
            "query": query,
            "results": results,
            "total": len(results)
        }

async def main():
    """Main application entry point."""
    config = Config(debug=True)
    db_manager = DatabaseManager(config)
    api_service = APIService(db_manager)
    
    # Initialize services
    if not await db_manager.connect():
        raise RuntimeError("Failed to connect to database")
    
    # Process some sample requests
    requests = [
        {"endpoint": "analyze", "data": {"code": "def hello(): return 'world'"}},
        {"endpoint": "search", "data": {"query": "authentication function"}},
        {"endpoint": "analyze", "data": {"code": "class Example: pass"}},
    ]
    
    for req in requests:
        result = await api_service.handle_request(req["endpoint"], req["data"])
        print(f"Request {req['endpoint']}: {result['status']}")
    
    await db_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
''' * 3  # Make it 3x larger for performance testing


# WebSocket test fixtures
@pytest.fixture
def websocket_mock():
    """Mock WebSocket connection."""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.close = AsyncMock()
    return ws


# Error simulation fixtures
@pytest.fixture
def failing_embedding_provider():
    """Embedding provider that fails for testing error handling."""
    provider = MagicMock(spec=EmbeddingProvider)
    provider.generate_embedding = AsyncMock(side_effect=Exception("API rate limit exceeded"))
    provider.generate_embeddings = AsyncMock(side_effect=Exception("Service unavailable"))
    provider.health_check = MagicMock(return_value=False)
    return provider


@pytest.fixture
def failing_llm_client():
    """LLM client that fails for testing error handling."""
    client = MagicMock(spec=LLMClient)
    client.generate_recommendations = AsyncMock(side_effect=Exception("Model overloaded"))
    client.health_check = AsyncMock(return_value={"status": "unhealthy", "error": "Connection timeout"})
    return client


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Clean up fixtures
@pytest.fixture(autouse=True)
def cleanup_caches():
    """Clean up any caches between tests."""
    yield
    # Clear any global caches if they exist
    # This would be implemented based on actual cache structure