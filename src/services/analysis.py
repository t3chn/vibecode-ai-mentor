"""Analysis orchestration service for VibeCode AI Mentor.

This service coordinates the code analysis pipeline including parsing,
metrics calculation, chunking, and result aggregation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from uuid import uuid4

from src.analyzer.chunker import CodeChunker, ChunkStrategy
from src.analyzer.metrics import MetricsCalculator
from src.analyzer.parser import PythonParser

# Make module work without full config
try:
    from src.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


@dataclass
class FileAnalysis:
    """Result of analyzing a single file."""
    file_path: str
    language: str
    status: str  # success, error, skipped
    
    # Parse results
    functions: List[Dict] = field(default_factory=list)
    classes: List[Dict] = field(default_factory=list)
    imports: List[Dict] = field(default_factory=list)
    
    # Metrics
    metrics: Optional[Dict] = None
    
    # Chunks for embedding
    chunks: List[Dict] = field(default_factory=list)
    
    # Analysis metadata
    analysis_time_ms: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class SnippetAnalysis:
    """Result of analyzing a code snippet."""
    snippet_id: str
    language: str
    
    # Parse results
    ast_elements: Dict = field(default_factory=dict)
    
    # Metrics
    metrics: Optional[Dict] = None
    
    # Issues found
    issues: List[Dict] = field(default_factory=list)
    
    # Analysis metadata
    analysis_time_ms: float = 0.0


@dataclass
class RepositoryAnalysis:
    """Result of analyzing an entire repository."""
    repository_path: str
    total_files: int
    analyzed_files: int
    skipped_files: int
    failed_files: int
    
    # Aggregate metrics
    total_lines: int = 0
    average_complexity: float = 0.0
    total_functions: int = 0
    total_classes: int = 0
    
    # File analyses
    file_analyses: List[FileAnalysis] = field(default_factory=list)
    
    # Errors and warnings
    errors: List[Dict] = field(default_factory=list)
    
    # Timing
    total_time_seconds: float = 0.0
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


# Type for progress callbacks
ProgressCallback = Callable[[Dict], None]


class AnalysisService:
    """Orchestrates code analysis pipeline."""
    
    def __init__(self):
        """Initialize analysis service."""
        self.parser = PythonParser()
        self.metrics_calculator = MetricsCalculator()
        self.chunker = CodeChunker()
        
    async def analyze_file(
        self,
        file_path: str | Path,
        callback: Optional[ProgressCallback] = None
    ) -> FileAnalysis:
        """Analyze a single file.
        
        Args:
            file_path: Path to the file to analyze
            callback: Optional progress callback
            
        Returns:
            FileAnalysis result
        """
        file_path = Path(file_path)
        start_time = asyncio.get_event_loop().time()
        
        # Initialize result
        analysis = FileAnalysis(
            file_path=str(file_path),
            language="python",
            status="processing"
        )
        
        try:
            # Notify progress
            if callback:
                callback({
                    "type": "file_start",
                    "file": str(file_path),
                    "message": f"Analyzing {file_path.name}"
                })
            
            # Parse file
            logger.debug(f"Parsing {file_path}")
            tree = await self.parser.parse_file(file_path)
            
            # Extract AST elements
            analysis.functions = self.parser.extract_functions()
            analysis.classes = self.parser.extract_classes()
            analysis.imports = self.parser.extract_imports()
            
            # Calculate metrics
            logger.debug(f"Calculating metrics for {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            metrics = self.metrics_calculator.calculate_metrics(code)
            
            # Convert metrics to dict for serialization
            analysis.metrics = {
                "lines_of_code": metrics.lines_of_code,
                "logical_lines": metrics.logical_lines,
                "source_lines": metrics.source_lines,
                "blank_lines": metrics.blank_lines,
                "comment_lines": metrics.comment_lines,
                "cyclomatic_complexity": metrics.cyclomatic_complexity,
                "average_complexity": metrics.average_complexity,
                "complexity_rank": metrics.complexity_rank,
                "maintainability_index": metrics.maintainability_index,
                "maintainability_grade": metrics.maintainability_grade,
                "complexity_score": metrics.complexity_score,
                "risk_level": metrics.risk_level,
                "functions_count": len(metrics.functions),
                "anti_patterns": metrics.anti_patterns
            }
            
            # Add function-level metrics
            if metrics.functions:
                analysis.metrics["function_metrics"] = [
                    {
                        "name": f.name,
                        "complexity": f.cyclomatic_complexity,
                        "lines": f.lines_of_code,
                        "parameters": f.parameter_count,
                        "is_complex": f.is_complex,
                        "is_long": f.is_long
                    }
                    for f in metrics.functions
                ]
            
            # Chunk the code
            logger.debug(f"Chunking {file_path}")
            chunks = await self.chunker.chunk_file(file_path)
            
            # Convert chunks to dicts
            analysis.chunks = [
                {
                    "content": chunk.content,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "chunk_type": chunk.chunk_type,
                    "metadata": chunk.metadata,
                    "token_count": chunk.token_count
                }
                for chunk in chunks
            ]
            
            # Check for warnings
            if metrics.anti_patterns:
                for pattern in metrics.anti_patterns:
                    if pattern["severity"] == "warning":
                        analysis.warnings.append(pattern["message"])
            
            analysis.status = "success"
            
        except FileNotFoundError:
            analysis.status = "error"
            analysis.error_message = f"File not found: {file_path}"
            logger.error(analysis.error_message)
            
        except Exception as e:
            analysis.status = "error"
            analysis.error_message = f"Analysis failed: {str(e)}"
            logger.error(f"Error analyzing {file_path}: {e}", exc_info=True)
        
        finally:
            # Calculate elapsed time
            analysis.analysis_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # Notify completion
            if callback:
                callback({
                    "type": "file_complete",
                    "file": str(file_path),
                    "status": analysis.status,
                    "time_ms": analysis.analysis_time_ms
                })
        
        return analysis
    
    async def analyze_repository(
        self,
        repo_path: str | Path,
        callback: Optional[ProgressCallback] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None
    ) -> RepositoryAnalysis:
        """Analyze an entire repository.
        
        Args:
            repo_path: Path to the repository
            callback: Optional progress callback
            include_patterns: File patterns to include (default: ["**/*.py"])
            exclude_patterns: File patterns to exclude
            
        Returns:
            RepositoryAnalysis result
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise ValueError(f"Repository path does not exist: {repo_path}")
        
        # Default patterns
        if include_patterns is None:
            include_patterns = ["**/*.py"]
        if exclude_patterns is None:
            exclude_patterns = ["**/__pycache__/**", "**/.*", "**/node_modules/**"]
        
        # Initialize result
        result = RepositoryAnalysis(
            repository_path=str(repo_path),
            total_files=0,
            analyzed_files=0,
            skipped_files=0,
            failed_files=0
        )
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Find all matching files
            files = []
            for pattern in include_patterns:
                files.extend(repo_path.glob(pattern))
            
            # Remove excluded files
            excluded_files = set()
            for pattern in exclude_patterns:
                excluded_files.update(repo_path.glob(pattern))
            
            files = [f for f in files if f not in excluded_files and f.is_file()]
            result.total_files = len(files)
            
            logger.info(f"Found {result.total_files} files to analyze in {repo_path}")
            
            # Notify start
            if callback:
                callback({
                    "type": "repository_start",
                    "repository": str(repo_path),
                    "total_files": result.total_files
                })
            
            # Analyze each file
            for i, file_path in enumerate(files):
                # Check if should skip
                if file_path.stat().st_size == 0:
                    result.skipped_files += 1
                    logger.debug(f"Skipping empty file: {file_path}")
                    continue
                
                # Progress update
                if callback:
                    callback({
                        "type": "progress",
                        "current_file": str(file_path),
                        "files_processed": i,
                        "total_files": result.total_files,
                        "percentage": (i / result.total_files) * 100
                    })
                
                # Analyze file
                try:
                    file_analysis = await self.analyze_file(file_path)
                    result.file_analyses.append(file_analysis)
                    
                    if file_analysis.status == "success":
                        result.analyzed_files += 1
                        
                        # Update aggregate metrics
                        if file_analysis.metrics:
                            result.total_lines += file_analysis.metrics["lines_of_code"]
                            result.total_functions += len(file_analysis.functions)
                            result.total_classes += len(file_analysis.classes)
                    else:
                        result.failed_files += 1
                        result.errors.append({
                            "file": str(file_path),
                            "error": file_analysis.error_message
                        })
                        
                except Exception as e:
                    result.failed_files += 1
                    result.errors.append({
                        "file": str(file_path),
                        "error": str(e)
                    })
                    logger.error(f"Failed to analyze {file_path}: {e}")
            
            # Calculate average complexity
            if result.file_analyses:
                total_complexity = sum(
                    fa.metrics.get("average_complexity", 0)
                    for fa in result.file_analyses
                    if fa.metrics
                )
                result.average_complexity = total_complexity / len(result.file_analyses)
            
        except Exception as e:
            logger.error(f"Repository analysis failed: {e}", exc_info=True)
            result.errors.append({
                "type": "repository_error",
                "error": str(e)
            })
        
        finally:
            # Finalize timing
            result.total_time_seconds = asyncio.get_event_loop().time() - start_time
            result.completed_at = datetime.utcnow()
            
            # Final callback
            if callback:
                callback({
                    "type": "repository_complete",
                    "repository": str(repo_path),
                    "analyzed_files": result.analyzed_files,
                    "failed_files": result.failed_files,
                    "total_time_seconds": result.total_time_seconds
                })
        
        return result
    
    def analyze_code_snippet(
        self,
        code: str,
        language: str = "python"
    ) -> SnippetAnalysis:
        """Analyze a code snippet.
        
        Args:
            code: Code snippet to analyze
            language: Programming language (currently only python)
            
        Returns:
            SnippetAnalysis result
        """
        if language != "python":
            raise ValueError(f"Unsupported language: {language}")
        
        start_time = asyncio.get_event_loop().time()
        
        # Initialize result
        analysis = SnippetAnalysis(
            snippet_id=str(uuid4()),
            language=language
        )
        
        try:
            # Parse code
            tree = self.parser.parse_code(code)
            
            # Extract elements
            analysis.ast_elements = {
                "functions": self.parser.extract_functions(),
                "classes": self.parser.extract_classes(),
                "imports": self.parser.extract_imports(),
                "globals": self.parser.extract_global_variables()
            }
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(code)
            
            # Convert to dict
            analysis.metrics = {
                "lines_of_code": metrics.lines_of_code,
                "cyclomatic_complexity": metrics.cyclomatic_complexity,
                "maintainability_index": metrics.maintainability_index,
                "complexity_score": metrics.complexity_score,
                "risk_level": metrics.risk_level
            }
            
            # Extract issues from anti-patterns
            if metrics.anti_patterns:
                analysis.issues = [
                    {
                        "type": pattern["type"],
                        "severity": pattern["severity"],
                        "message": pattern["message"],
                        "line": pattern.get("line", 1)
                    }
                    for pattern in metrics.anti_patterns
                ]
            
        except Exception as e:
            logger.error(f"Snippet analysis failed: {e}")
            analysis.issues.append({
                "type": "parse_error",
                "severity": "error",
                "message": str(e),
                "line": 1
            })
        
        finally:
            analysis.analysis_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return analysis


# Convenience functions
async def analyze_file(
    file_path: str | Path,
    callback: Optional[ProgressCallback] = None
) -> FileAnalysis:
    """Analyze a single file."""
    service = AnalysisService()
    return await service.analyze_file(file_path, callback)


async def analyze_repository(
    repo_path: str | Path,
    callback: Optional[ProgressCallback] = None,
    include_patterns: Optional[List[str]] = None,
    exclude_patterns: Optional[List[str]] = None
) -> RepositoryAnalysis:
    """Analyze a repository."""
    service = AnalysisService()
    return await service.analyze_repository(
        repo_path, callback, include_patterns, exclude_patterns
    )


def analyze_snippet(code: str, language: str = "python") -> SnippetAnalysis:
    """Analyze a code snippet."""
    service = AnalysisService()
    return service.analyze_code_snippet(code, language)