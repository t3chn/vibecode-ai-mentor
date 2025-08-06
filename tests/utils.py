"""Test utilities and helpers for VibeCode AI Mentor tests."""

import asyncio
import hashlib
import json
import random
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Repository, CodeSnippet, AnalysisResult, RepositoryStatus
from src.db.repositories import RepositoryRepo, CodeSnippetRepo, AnalysisResultRepo


class TestDataGenerator:
    """Generate test data for various testing scenarios."""

    @staticmethod
    def generate_random_string(length: int = 10) -> str:
        """Generate random string of specified length."""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

    @staticmethod
    def generate_python_code(
        num_functions: int = 3,
        num_classes: int = 1,
        complexity_level: str = "medium"
    ) -> str:
        """Generate Python code with specified characteristics."""
        code_parts = []
        
        # Add imports
        imports = [
            "import os",
            "import sys", 
            "from typing import List, Dict, Optional",
            "import json",
            "import asyncio"
        ]
        code_parts.extend(random.sample(imports, k=min(3, len(imports))))
        code_parts.append("")
        
        # Generate functions
        for i in range(num_functions):
            func_name = f"function_{i}"
            
            if complexity_level == "simple":
                code_parts.append(f"def {func_name}():")
                code_parts.append("    return None")
            elif complexity_level == "medium":
                code_parts.append(f"def {func_name}(data: List[str]) -> Dict[str, Any]:")
                code_parts.append('    """Process data and return results."""')
                code_parts.append("    result = {}")
                code_parts.append("    for item in data:")
                code_parts.append("        if item:")
                code_parts.append("            result[item] = len(item)")
                code_parts.append("    return result")
            else:  # complex
                code_parts.append(f"async def {func_name}(data: Dict[str, Any], config: Optional[Dict] = None) -> Tuple[bool, List[str]]:")
                code_parts.append('    """Complex async function with error handling."""')
                code_parts.append("    errors = []")
                code_parts.append("    try:")
                code_parts.append("        if not data:")
                code_parts.append("            raise ValueError('Data cannot be empty')")
                code_parts.append("        ")
                code_parts.append("        processed = []")
                code_parts.append("        for key, value in data.items():")
                code_parts.append("            if isinstance(value, (str, int, float)):")
                code_parts.append("                processed.append(f'{key}:{value}')")
                code_parts.append("            else:")
                code_parts.append("                errors.append(f'Invalid type for {key}')")
                code_parts.append("        ")
                code_parts.append("        if config and config.get('validate'):")
                code_parts.append("            await asyncio.sleep(0.1)  # Simulate validation")
                code_parts.append("        ")
                code_parts.append("        return len(errors) == 0, errors")
                code_parts.append("    except Exception as e:")
                code_parts.append("        errors.append(str(e))")
                code_parts.append("        return False, errors")
            
            code_parts.append("")
        
        # Generate classes
        for i in range(num_classes):
            class_name = f"TestClass{i}"
            code_parts.append(f"class {class_name}:")
            code_parts.append('    """Test class for demonstration."""')
            code_parts.append("    ")
            code_parts.append("    def __init__(self, name: str):")
            code_parts.append("        self.name = name")
            code_parts.append("        self.data = []")
            code_parts.append("    ")
            code_parts.append("    def add_item(self, item: Any) -> None:")
            code_parts.append("        self.data.append(item)")
            code_parts.append("    ")
            code_parts.append("    def get_summary(self) -> Dict[str, Any]:")
            code_parts.append("        return {")
            code_parts.append("            'name': self.name,")
            code_parts.append("            'item_count': len(self.data),")
            code_parts.append("            'last_item': self.data[-1] if self.data else None")
            code_parts.append("        }")
            code_parts.append("")
        
        return "\n".join(code_parts)

    @staticmethod
    def generate_code_snippets(count: int = 10) -> List[Dict[str, Any]]:
        """Generate list of code snippet data."""
        snippets = []
        
        snippet_templates = [
            {
                "content": "def calculate_sum(numbers):\n    return sum(numbers)",
                "language": "python",
                "complexity": 1.0
            },
            {
                "content": "def validate_email(email):\n    return '@' in email and '.' in email",
                "language": "python", 
                "complexity": 2.0
            },
            {
                "content": "async def fetch_data(url):\n    async with aiohttp.ClientSession() as session:\n        async with session.get(url) as response:\n            return await response.json()",
                "language": "python",
                "complexity": 4.0
            },
            {
                "content": "class DataProcessor:\n    def __init__(self):\n        self.cache = {}\n    \n    def process(self, data):\n        key = hash(str(data))\n        if key in self.cache:\n            return self.cache[key]\n        result = self._heavy_computation(data)\n        self.cache[key] = result\n        return result",
                "language": "python",
                "complexity": 6.0
            }
        ]
        
        for i in range(count):
            template = random.choice(snippet_templates)
            snippet = {
                "file_path": f"src/module_{i}.py",
                "language": template["language"],
                "content": template["content"],
                "embedding": [random.uniform(-1.0, 1.0) for _ in range(1536)],
                "start_line": random.randint(1, 100),
                "end_line": random.randint(1, 100) + 5,
                "complexity_score": template["complexity"] + random.uniform(-0.5, 0.5)
            }
            snippet["end_line"] = max(snippet["start_line"], snippet["end_line"])
            snippets.append(snippet)
        
        return snippets

    @staticmethod
    def generate_search_results(
        count: int = 5,
        similarity_range: Tuple[float, float] = (0.7, 0.95)
    ) -> List[Dict[str, Any]]:
        """Generate mock search results."""
        results = []
        
        for i in range(count):
            similarity = random.uniform(*similarity_range)
            results.append({
                "snippet_id": str(uuid.uuid4()),
                "content": f"def search_result_{i}(): # Example function {i}",
                "file_path": f"src/result_{i}.py",
                "language": "python",
                "similarity_score": similarity,
                "line_start": random.randint(1, 50),
                "line_end": random.randint(51, 100),
                "complexity_score": random.uniform(1.0, 8.0),
                "repository_name": f"repo_{i % 3}"  # Some overlap
            })
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results

    @staticmethod
    def generate_analysis_recommendations(count: int = 5) -> List[Dict[str, Any]]:
        """Generate mock analysis recommendations."""
        recommendation_types = ["style", "performance", "security", "maintainability", "complexity"]
        severities = ["info", "warning", "error"]
        
        recommendations = []
        
        for i in range(count):
            rec_type = random.choice(recommendation_types)
            severity = random.choice(severities)
            
            recommendations.append({
                "type": rec_type,
                "severity": severity,
                "message": f"Consider improving {rec_type} in this code section",
                "suggestion": f"Suggested fix for {rec_type} issue",
                "line_start": random.randint(1, 50),
                "line_end": random.randint(51, 100),
                "confidence": random.uniform(0.5, 0.95)
            })
        
        return recommendations

    @staticmethod
    def generate_embedding_vector(dimension: int = 1536) -> List[float]:
        """Generate normalized embedding vector."""
        # Generate random vector
        vector = [random.gauss(0, 1) for _ in range(dimension)]
        
        # Normalize to unit length
        magnitude = sum(x*x for x in vector) ** 0.5
        if magnitude > 0:
            vector = [x / magnitude for x in vector]
        
        return vector

    @staticmethod
    def generate_similar_embeddings(
        base_embedding: List[float],
        count: int = 5,
        similarity_level: float = 0.9
    ) -> List[List[float]]:
        """Generate embeddings similar to a base embedding."""
        similar_embeddings = []
        
        for _ in range(count):
            # Create variation of base embedding
            noise_factor = 1.0 - similarity_level
            similar = []
            
            for value in base_embedding:
                noise = random.gauss(0, noise_factor * 0.1)
                similar.append(value + noise)
            
            # Normalize
            magnitude = sum(x*x for x in similar) ** 0.5
            if magnitude > 0:
                similar = [x / magnitude for x in similar]
            
            similar_embeddings.append(similar)
        
        return similar_embeddings


class DatabaseTestHelper:
    """Helper functions for database testing."""

    @staticmethod
    async def create_test_repository(
        session: AsyncSession,
        name: Optional[str] = None,
        status: RepositoryStatus = RepositoryStatus.COMPLETED
    ) -> Repository:
        """Create a test repository."""
        repo_repo = RepositoryRepo(session)
        
        name = name or f"test-repo-{TestDataGenerator.generate_random_string(6)}"
        
        repo = await repo_repo.create(
            name=name,
            url=f"https://github.com/test/{name}"
        )
        
        if status != RepositoryStatus.PENDING:
            await repo_repo.update_status(repo.id, status)
        
        return await repo_repo.get_by_id(repo.id)

    @staticmethod
    async def create_test_snippets(
        session: AsyncSession,
        repository: Repository,
        count: int = 5
    ) -> List[CodeSnippet]:
        """Create test code snippets for a repository."""
        snippet_repo = CodeSnippetRepo(session)
        
        snippets_data = []
        for i in range(count):
            snippets_data.append({
                "repository_id": repository.id,
                "file_path": f"src/test_file_{i}.py",
                "language": "python",
                "content": f"def test_function_{i}():\n    return {i}",
                "embedding": TestDataGenerator.generate_embedding_vector(),
                "start_line": 1,
                "end_line": 2,
                "complexity_score": float(i + 1)
            })
        
        created_snippets = await snippet_repo.create_batch(snippets_data)
        return created_snippets

    @staticmethod
    async def create_test_analysis_results(
        session: AsyncSession,
        snippets: List[CodeSnippet],
        results_per_snippet: int = 1
    ) -> List[AnalysisResult]:
        """Create test analysis results for snippets."""
        result_repo = AnalysisResultRepo(session)
        results = []
        
        for snippet in snippets:
            for i in range(results_per_snippet):
                recommendations = TestDataGenerator.generate_analysis_recommendations(3)
                
                result = await result_repo.create(
                    snippet_id=snippet.id,
                    recommendations=recommendations,
                    similar_patterns=[],
                    quality_score=random.uniform(60.0, 95.0)
                )
                results.append(result)
        
        return results

    @staticmethod
    async def cleanup_test_data(
        session: AsyncSession,
        repository_ids: Optional[List[uuid.UUID]] = None
    ):
        """Clean up test data from database."""
        if repository_ids:
            repo_repo = RepositoryRepo(session)
            for repo_id in repository_ids:
                await repo_repo.delete(repo_id)


class MockServiceFactory:
    """Factory for creating mock services."""

    @staticmethod
    def create_mock_embedding_provider(
        embedding_dim: int = 1536,
        response_time: float = 0.1
    ):
        """Create mock embedding provider."""
        provider = MagicMock()
        
        async def mock_generate_embedding(text: str):
            await asyncio.sleep(response_time)  # Simulate API call
            # Generate deterministic embedding based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            seed = int(text_hash[:8], 16)
            random.seed(seed)
            return TestDataGenerator.generate_embedding_vector(embedding_dim)
        
        async def mock_generate_embeddings(texts: List[str]):
            return [await mock_generate_embedding(text) for text in texts]
        
        provider.generate_embedding = AsyncMock(side_effect=mock_generate_embedding)
        provider.generate_embeddings = AsyncMock(side_effect=mock_generate_embeddings)
        provider.health_check = MagicMock(return_value=True)
        
        return provider

    @staticmethod
    def create_mock_llm_client(response_delay: float = 0.5):
        """Create mock LLM client."""
        client = MagicMock()
        
        async def mock_generate_recommendations(code: str, **kwargs):
            await asyncio.sleep(response_delay)  # Simulate LLM processing
            
            # Generate deterministic recommendations based on code
            recommendations = TestDataGenerator.generate_analysis_recommendations(
                count=random.randint(2, 5)
            )
            
            return {
                "recommendations": recommendations,
                "refactoring_suggestions": [
                    {
                        "refactor_type": "extract_method",
                        "description": "Consider extracting complex logic to separate method",
                        "confidence": 0.8
                    }
                ],
                "anti_pattern_fixes": [],
                "summary": "Code analysis completed with suggestions for improvement",
                "overall_score": random.randint(70, 95)
            }
        
        client.generate_recommendations = AsyncMock(side_effect=mock_generate_recommendations)
        client.health_check = AsyncMock(return_value={"status": "healthy"})
        
        return client

    @staticmethod
    def create_failing_service(failure_rate: float = 0.3):
        """Create a service that fails randomly for error testing."""
        service = MagicMock()
        
        async def failing_method(*args, **kwargs):
            if random.random() < failure_rate:
                raise Exception("Service temporarily unavailable")
            await asyncio.sleep(0.1)
            return {"status": "success", "data": "mock_result"}
        
        service.process = AsyncMock(side_effect=failing_method)
        return service


class FileSystemTestHelper:
    """Helper functions for file system testing."""

    @staticmethod
    def create_test_repository_structure(
        base_path: Path,
        file_count: int = 10,
        include_tests: bool = True
    ) -> Path:
        """Create a realistic repository structure for testing."""
        repo_path = base_path / "test_repository"
        repo_path.mkdir(exist_ok=True)
        
        # Create source directory
        src_path = repo_path / "src"
        src_path.mkdir(exist_ok=True)
        
        # Create main files
        for i in range(file_count):
            file_path = src_path / f"module_{i}.py"
            code = TestDataGenerator.generate_python_code(
                num_functions=random.randint(2, 5),
                num_classes=random.randint(0, 2),
                complexity_level=random.choice(["simple", "medium", "complex"])
            )
            file_path.write_text(code)
        
        # Create subdirectories
        utils_path = src_path / "utils"
        utils_path.mkdir(exist_ok=True)
        
        for i in range(3):
            file_path = utils_path / f"helper_{i}.py"
            code = TestDataGenerator.generate_python_code(
                num_functions=2,
                num_classes=0,
                complexity_level="simple"
            )
            file_path.write_text(code)
        
        # Create test files if requested
        if include_tests:
            tests_path = repo_path / "tests"
            tests_path.mkdir(exist_ok=True)
            
            for i in range(file_count // 2):
                test_file = tests_path / f"test_module_{i}.py"
                test_code = f"""
import pytest
from src.module_{i} import *

def test_function_{i}():
    # Test implementation
    assert True

def test_edge_case_{i}():
    # Edge case testing
    assert True
"""
                test_file.write_text(test_code)
        
        # Create configuration files
        (repo_path / "requirements.txt").write_text("pytest>=7.0.0\nblack>=22.0.0\nmypy>=0.910\n")
        (repo_path / "setup.py").write_text("from setuptools import setup, find_packages\n\nsetup(name='test_repo', packages=find_packages())\n")
        (repo_path / ".gitignore").write_text("__pycache__/\n*.pyc\n.pytest_cache/\n")
        
        return repo_path

    @staticmethod
    def create_large_python_file(file_path: Path, lines_count: int = 1000) -> Path:
        """Create a large Python file for performance testing."""
        code_blocks = []
        
        # Add imports
        imports = [
            "import os", "import sys", "import json", "import asyncio",
            "from typing import List, Dict, Any, Optional",
            "from dataclasses import dataclass",
            "import logging"
        ]
        code_blocks.extend(imports)
        code_blocks.append("")
        
        # Generate functions to reach target line count
        current_lines = len(code_blocks)
        function_count = 0
        
        while current_lines < lines_count:
            func_code = TestDataGenerator.generate_python_code(
                num_functions=1,
                num_classes=0,
                complexity_level=random.choice(["medium", "complex"])
            ).split('\n')[2:]  # Skip imports
            
            # Add function with unique name
            func_code[0] = func_code[0].replace("function_0", f"function_{function_count}")
            
            code_blocks.extend(func_code)
            code_blocks.append("")
            
            current_lines = len(code_blocks)
            function_count += 1
        
        file_path.write_text('\n'.join(code_blocks))
        return file_path


class PerformanceTestHelper:
    """Helper functions for performance testing."""

    @staticmethod
    def measure_execution_time(func):
        """Decorator to measure function execution time."""
        import time
        import functools
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            # Store timing info in result if it's a dict
            if isinstance(result, dict):
                result['_execution_time'] = execution_time
            
            return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            if isinstance(result, dict):
                result['_execution_time'] = execution_time
            
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    @staticmethod
    def create_load_test_scenario(
        request_count: int,
        concurrent_users: int,
        ramp_up_time: float = 1.0
    ) -> List[float]:
        """Create timing schedule for load testing."""
        # Calculate delay between requests for ramp-up
        if concurrent_users > 1:
            delay_between_starts = ramp_up_time / concurrent_users
        else:
            delay_between_starts = 0
        
        start_times = []
        for i in range(request_count):
            user_index = i % concurrent_users
            base_time = user_index * delay_between_starts
            request_in_user_session = i // concurrent_users
            
            # Add some randomness to make it more realistic
            jitter = random.uniform(-0.1, 0.1)
            start_time = base_time + (request_in_user_session * 1.0) + jitter
            
            start_times.append(max(0, start_time))
        
        return sorted(start_times)

    @staticmethod
    def calculate_performance_metrics(response_times: List[float]) -> Dict[str, float]:
        """Calculate performance metrics from response times."""
        if not response_times:
            return {}
        
        sorted_times = sorted(response_times)
        count = len(sorted_times)
        
        return {
            "count": count,
            "min": min(sorted_times),
            "max": max(sorted_times),
            "mean": sum(sorted_times) / count,
            "median": sorted_times[count // 2],
            "p90": sorted_times[int(0.9 * count)],
            "p95": sorted_times[int(0.95 * count)],
            "p99": sorted_times[int(0.99 * count)],
            "throughput": count / sum(sorted_times) if sum(sorted_times) > 0 else 0
        }


class AssertionHelper:
    """Custom assertions for testing."""

    @staticmethod
    def assert_response_time_within_bounds(
        response_time: float,
        max_time: float,
        operation_name: str = "operation"
    ):
        """Assert response time is within acceptable bounds."""
        assert response_time <= max_time, \
            f"{operation_name} took {response_time:.3f}s, expected <= {max_time}s"

    @staticmethod
    def assert_similarity_score_valid(score: float):
        """Assert similarity score is valid."""
        assert 0.0 <= score <= 1.0, f"Similarity score {score} must be between 0.0 and 1.0"

    @staticmethod
    def assert_embedding_vector_valid(embedding: List[float], expected_dim: int = 1536):
        """Assert embedding vector is valid."""
        assert len(embedding) == expected_dim, \
            f"Embedding dimension {len(embedding)} != expected {expected_dim}"
        
        assert all(isinstance(x, (int, float)) for x in embedding), \
            "All embedding values must be numeric"
        
        # Check if normalized (optional, for unit vectors)
        magnitude = sum(x*x for x in embedding) ** 0.5
        assert 0.5 <= magnitude <= 2.0, \
            f"Embedding magnitude {magnitude} seems unusual"

    @staticmethod
    def assert_database_constraints_met(record: Any, required_fields: List[str]):
        """Assert database record meets basic constraints."""
        for field in required_fields:
            assert hasattr(record, field), f"Record missing required field: {field}"
            value = getattr(record, field)
            assert value is not None, f"Required field {field} cannot be None"

    @staticmethod
    def assert_search_results_sorted(results: List[Dict[str, Any]], sort_key: str = "similarity_score"):
        """Assert search results are properly sorted."""
        if len(results) <= 1:
            return
        
        for i in range(1, len(results)):
            current_score = results[i].get(sort_key, 0)
            previous_score = results[i-1].get(sort_key, 0)
            assert current_score <= previous_score, \
                f"Results not sorted: {previous_score} should be >= {current_score}"


# Global test configuration
TEST_CONFIG = {
    "performance_thresholds": {
        "api_response_time": 0.5,
        "search_response_time": 1.0,
        "analysis_response_time": 2.0,
        "indexing_throughput": 1000,  # lines per second
    },
    "load_test_parameters": {
        "max_concurrent_users": 50,
        "ramp_up_time": 5.0,
        "test_duration": 30.0,
    },
    "database_limits": {
        "max_batch_size": 1000,
        "query_timeout": 10.0,
        "connection_pool_size": 20,
    }
}