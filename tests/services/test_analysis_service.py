"""Tests for the AnalysisService."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.analysis import AnalysisService
from src.analyzer.parser import PythonParser
from src.analyzer.chunker import CodeChunker
from src.analyzer.metrics import CodeMetrics
from tests.utils import TestDataGenerator, FileSystemTestHelper, PerformanceTestHelper


class TestAnalysisService:
    """Test AnalysisService functionality."""

    @pytest.fixture
    def analysis_service(self):
        """Create AnalysisService instance."""
        return AnalysisService()

    @pytest.mark.asyncio
    async def test_analyze_file_success(
        self,
        analysis_service: AnalysisService,
        temp_python_file: Path,
        mock_embedding_provider
    ):
        """Test successful file analysis."""
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=temp_python_file,
                language="python"
            )
        
        assert result is not None
        assert result.file_path == str(temp_python_file)
        assert result.language == "python"
        assert result.status == "success"
        assert len(result.chunks) > 0
        assert result.metrics is not None
        assert result.processing_time_seconds > 0

    @pytest.mark.asyncio
    async def test_analyze_file_nonexistent(self, analysis_service: AnalysisService):
        """Test analysis of nonexistent file."""
        nonexistent_file = Path("/nonexistent/file.py")
        
        result = await analysis_service.analyze_file(
            file_path=nonexistent_file,
            language="python"
        )
        
        assert result.status == "failed"
        assert "not found" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_analyze_file_empty(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path
    ):
        """Test analysis of empty file."""
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("")
        
        result = await analysis_service.analyze_file(
            file_path=empty_file,
            language="python"
        )
        
        # Should handle empty files gracefully
        assert result.status in ["success", "skipped"]
        if result.status == "success":
            assert len(result.chunks) == 0

    @pytest.mark.asyncio
    async def test_analyze_file_malformed_code(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test analysis of malformed Python code."""
        malformed_file = tmp_path / "malformed.py"
        malformed_file.write_text("def broken_function(\n    return None")
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=malformed_file,
                language="python"
            )
        
        # Should handle malformed code gracefully
        assert result is not None
        # May succeed with warnings or fail gracefully
        if result.status == "success":
            assert len(result.chunks) >= 0
        else:
            assert "parse" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_analyze_file_with_embedding_failure(
        self,
        analysis_service: AnalysisService,
        temp_python_file: Path,
        failing_embedding_provider
    ):
        """Test file analysis when embedding generation fails."""
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = failing_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=temp_python_file,
                language="python"
            )
        
        # Analysis should still succeed even if embeddings fail
        assert result.status == "success"
        assert len(result.chunks) > 0
        # Chunks may not have embeddings, but that's okay

    @pytest.mark.asyncio
    async def test_analyze_repository_success(
        self,
        analysis_service: AnalysisService,
        temp_repository: Path,
        mock_embedding_provider
    ):
        """Test successful repository analysis."""
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_repository(
                repo_path=temp_repository,
                include_patterns=["**/*.py"],
                exclude_patterns=["**/test_*", "**/__pycache__/**"]
            )
        
        assert result is not None
        assert result.total_files >= 2  # main.py and utils.py
        assert result.analyzed_files > 0
        assert result.failed_files >= 0
        assert len(result.file_analyses) > 0
        assert result.total_time_seconds > 0

    @pytest.mark.asyncio
    async def test_analyze_repository_with_filters(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test repository analysis with include/exclude patterns."""
        # Create test repository structure
        repo_path = FileSystemTestHelper.create_test_repository_structure(
            tmp_path, file_count=5, include_tests=True
        )
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Analyze only source files, exclude tests
            result = await analysis_service.analyze_repository(
                repo_path=repo_path,
                include_patterns=["src/**/*.py"],
                exclude_patterns=["**/test_*"]
            )
        
        assert result.analyzed_files > 0
        
        # Verify no test files were analyzed
        for file_analysis in result.file_analyses:
            assert "test_" not in file_analysis.file_path

    @pytest.mark.asyncio
    async def test_analyze_repository_nonexistent(
        self,
        analysis_service: AnalysisService
    ):
        """Test analysis of nonexistent repository."""
        nonexistent_repo = Path("/nonexistent/repository")
        
        result = await analysis_service.analyze_repository(
            repo_path=nonexistent_repo,
            include_patterns=["**/*.py"]
        )
        
        assert result.total_files == 0
        assert result.analyzed_files == 0
        assert len(result.file_analyses) == 0

    @pytest.mark.asyncio
    async def test_analyze_repository_empty(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path
    ):
        """Test analysis of empty repository."""
        empty_repo = tmp_path / "empty_repo"
        empty_repo.mkdir()
        
        result = await analysis_service.analyze_repository(
            repo_path=empty_repo,
            include_patterns=["**/*.py"]
        )
        
        assert result.total_files == 0
        assert result.analyzed_files == 0
        assert len(result.file_analyses) == 0

    @pytest.mark.asyncio
    async def test_concurrent_file_analysis(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test concurrent analysis of multiple files."""
        # Create multiple test files
        files = []
        for i in range(5):
            file_path = tmp_path / f"concurrent_test_{i}.py"
            code = TestDataGenerator.generate_python_code(
                num_functions=3, complexity_level="medium"
            )
            file_path.write_text(code)
            files.append(file_path)
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Analyze files concurrently
            tasks = [
                analysis_service.analyze_file(file_path, "python")
                for file_path in files
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all analyses completed
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) == len(files)
        
        for result in successful_results:
            assert result.status == "success"
            assert len(result.chunks) > 0

    @pytest.mark.asyncio
    async def test_analyze_large_file_performance(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test performance with large files."""
        # Create large file
        large_file = FileSystemTestHelper.create_large_python_file(
            tmp_path / "large_file.py", lines_count=2000
        )
        
        with patch("src.embeddings.factory.get_embedding_provider") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=large_file,
                language="python"
            )
        
        assert result.status == "success"
        assert len(result.chunks) > 10  # Should create multiple chunks
        assert result.processing_time_seconds < 5.0  # Should complete within 5 seconds

    @pytest.mark.asyncio
    async def test_analyze_file_metrics_calculation(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test that file analysis calculates metrics correctly."""
        # Create file with known characteristics
        test_file = tmp_path / "metrics_test.py"
        code = '''
def simple_function():
    return True

def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0

class TestClass:
    def method1(self):
        pass
    
    def method2(self, a, b):
        if a > b:
            return a
        return b
'''
        test_file.write_text(code)
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=test_file,
                language="python"
            )
        
        assert result.status == "success"
        assert result.metrics is not None
        
        # Check expected metrics
        assert "lines_of_code" in result.metrics
        assert "cyclomatic_complexity" in result.metrics
        assert "number_of_functions" in result.metrics
        assert "number_of_classes" in result.metrics
        
        assert result.metrics["lines_of_code"] > 0
        assert result.metrics["number_of_functions"] >= 2
        assert result.metrics["number_of_classes"] >= 1

    @pytest.mark.asyncio
    async def test_chunk_generation_consistency(
        self,
        analysis_service: AnalysisService,
        sample_python_code: str,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test that chunk generation is consistent."""
        test_file = tmp_path / "chunk_test.py"
        test_file.write_text(sample_python_code)
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            # Analyze the same file multiple times
            results = []
            for _ in range(3):
                result = await analysis_service.analyze_file(
                    file_path=test_file,
                    language="python"
                )
                results.append(result)
        
        # Results should be consistent
        assert all(r.status == "success" for r in results)
        
        # Chunk counts should be the same
        chunk_counts = [len(r.chunks) for r in results]
        assert len(set(chunk_counts)) == 1, "Chunk counts should be consistent"
        
        # Chunk content should be the same
        for i in range(len(results[0].chunks)):
            chunk_contents = [r.chunks[i]["content"] for r in results]
            assert len(set(chunk_contents)) == 1, f"Chunk {i} content should be consistent"

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path
    ):
        """Test error handling and recovery mechanisms."""
        # Create files with various issues
        files_and_issues = [
            ("valid.py", "def valid(): return True", "success"),
            ("empty.py", "", "success"),  # Should handle empty files
            ("invalid_syntax.py", "def broken(\nreturn", "failed"),
            ("binary.py", b"\x00\x01\x02\x03", "failed"),  # Binary content
        ]
        
        results = []
        for filename, content, expected_status in files_and_issues:
            file_path = tmp_path / filename
            if isinstance(content, bytes):
                file_path.write_bytes(content)
            else:
                file_path.write_text(content)
            
            result = await analysis_service.analyze_file(
                file_path=file_path,
                language="python"
            )
            results.append((filename, result, expected_status))
        
        # Check that service handles various error conditions
        for filename, result, expected_status in results:
            if expected_status == "success":
                assert result.status == "success", f"Failed to analyze {filename}"
            else:
                assert result.status in ["failed", "skipped"], f"Should have failed for {filename}"

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_repository(
        self,
        analysis_service: AnalysisService,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test memory efficiency with large repositories."""
        # Create repository with many files
        repo_path = FileSystemTestHelper.create_test_repository_structure(
            tmp_path, file_count=50, include_tests=False
        )
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_repository(
                repo_path=repo_path,
                include_patterns=["**/*.py"],
                batch_size=10  # Process in small batches for memory efficiency
            )
        
        assert result.total_files >= 50
        assert result.analyzed_files > 0
        
        # Memory usage should remain reasonable (implicit test - 
        # excessive memory would cause test failure or timeout)

    @pytest.mark.asyncio
    async def test_analysis_service_configuration(self, mock_embedding_provider):
        """Test analysis service with different configurations."""
        # Test with custom configuration
        custom_service = AnalysisService(
            max_file_size=1024*1024,  # 1MB limit
            timeout_seconds=30,
            batch_size=5
        )
        
        # Mock a small file
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory, \
             patch.object(Path, "stat") as mock_stat, \
             patch.object(Path, "read_text") as mock_read:
            
            mock_embed_factory.return_value = mock_embedding_provider
            mock_stat.return_value.st_size = 500  # Small file
            mock_read.return_value = "def test(): pass"
            
            result = await custom_service.analyze_file(
                file_path=Path("mock_file.py"),
                language="python"
            )
        
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_supported_languages(self, analysis_service: AnalysisService, tmp_path: Path):
        """Test analysis service with different programming languages."""
        language_samples = {
            "python": "def hello(): return 'world'",
            "javascript": "function hello() { return 'world'; }",
            "java": "public class Hello { public String hello() { return \"world\"; } }",
        }
        
        results = {}
        for language, code in language_samples.items():
            file_path = tmp_path / f"test.{language[:2]}"
            file_path.write_text(code)
            
            result = await analysis_service.analyze_file(
                file_path=file_path,
                language=language
            )
            results[language] = result
        
        # Python should definitely work
        assert results["python"].status == "success"
        
        # Other languages may work or be gracefully unsupported
        for language, result in results.items():
            assert result.status in ["success", "failed", "unsupported"]
            if result.status == "failed":
                assert result.error_message is not None


class TestAnalysisServiceIntegration:
    """Integration tests for AnalysisService with other components."""

    @pytest.mark.asyncio
    async def test_integration_with_parser(
        self,
        analysis_service: AnalysisService,
        temp_python_file: Path,
        mock_embedding_provider
    ):
        """Test integration with PythonParser."""
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=temp_python_file,
                language="python"
            )
        
        # Verify parser extracted expected elements
        assert result.status == "success"
        assert len(result.chunks) > 0
        
        # Check that chunks contain parsed elements
        chunk_contents = [chunk["content"] for chunk in result.chunks]
        combined_content = "\n".join(chunk_contents)
        
        # Should contain function definitions
        assert "def " in combined_content

    @pytest.mark.asyncio
    async def test_integration_with_chunker(
        self,
        analysis_service: AnalysisService,
        large_code_sample: str,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test integration with CodeChunker."""
        large_file = tmp_path / "large_chunker_test.py"
        large_file.write_text(large_code_sample)
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=large_file,
                language="python"
            )
        
        assert result.status == "success"
        assert len(result.chunks) > 1  # Should be chunked
        
        # Verify chunk properties
        for chunk in result.chunks:
            assert "content" in chunk
            assert "start_line" in chunk
            assert "end_line" in chunk
            assert chunk["start_line"] <= chunk["end_line"]
            assert len(chunk["content"]) > 0

    @pytest.mark.asyncio
    async def test_integration_with_metrics(
        self,
        analysis_service: AnalysisService,
        sample_python_code: str,
        tmp_path: Path,
        mock_embedding_provider
    ):
        """Test integration with CodeMetrics."""
        test_file = tmp_path / "metrics_integration_test.py"
        test_file.write_text(sample_python_code)
        
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=test_file,
                language="python"
            )
        
        assert result.status == "success"
        assert result.metrics is not None
        
        # Verify expected metrics are calculated
        expected_metrics = [
            "lines_of_code",
            "cyclomatic_complexity", 
            "maintainability_index",
            "number_of_functions",
            "number_of_classes"
        ]
        
        for metric in expected_metrics:
            assert metric in result.metrics
            assert isinstance(result.metrics[metric], (int, float))

    @pytest.mark.asyncio
    async def test_integration_with_embeddings(
        self,
        analysis_service: AnalysisService,
        temp_python_file: Path,
        mock_embedding_provider
    ):
        """Test integration with embedding generation."""
        with patch("src.embeddings.factory.get_embedding_manager") as mock_embed_factory:
            mock_embed_factory.return_value = mock_embedding_provider
            
            result = await analysis_service.analyze_file(
                file_path=temp_python_file,
                language="python"
            )
        
        assert result.status == "success"
        
        # Verify embeddings were requested for chunks
        assert mock_embedding_provider.generate_embedding.call_count >= len(result.chunks)
        
        # Check that embedding calls used chunk content
        call_args_list = mock_embedding_provider.generate_embedding.call_args_list
        for call_args in call_args_list:
            text_arg = call_args[0][0]  # First positional argument
            assert isinstance(text_arg, str)
            assert len(text_arg) > 0