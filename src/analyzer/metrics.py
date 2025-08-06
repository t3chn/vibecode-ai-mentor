"""Code complexity metrics calculator using radon library."""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from radon.complexity import cc_rank, cc_visit
from radon.metrics import h_visit, mi_compute, mi_rank
from radon.raw import analyze

# Make module work without full config
try:
    from src.core.logging import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


@dataclass
class FunctionMetrics:
    """Metrics for a single function."""
    name: str
    line_start: int
    line_end: int
    cyclomatic_complexity: int
    complexity_rank: str  # A-F grade
    parameter_count: int
    lines_of_code: int
    max_nesting_depth: int
    is_long: bool = False  # > 50 lines
    is_complex: bool = False  # CC > 10
    has_many_params: bool = False  # > 5 params
    is_deeply_nested: bool = False  # > 4 levels


@dataclass
class HalsteadMetrics:
    """Halstead complexity metrics."""
    volume: float
    difficulty: float
    effort: float
    time: float  # Estimated time to program (seconds)
    bugs: float  # Estimated number of bugs
    vocabulary: int  # n1 + n2 (unique operators + operands)
    length: int  # N1 + N2 (total operators + operands)


@dataclass
class CodeMetrics:
    """Complete code metrics for a file or snippet."""
    # Raw metrics
    lines_of_code: int
    logical_lines: int
    source_lines: int
    blank_lines: int
    comment_lines: int
    multi_line_strings: int
    
    # Complexity metrics
    cyclomatic_complexity: int
    average_complexity: float
    complexity_rank: str
    maintainability_index: float
    maintainability_grade: str
    
    # Halstead metrics
    halstead: Optional[HalsteadMetrics] = None
    
    # Function-level metrics
    functions: List[FunctionMetrics] = field(default_factory=list)
    
    # Quality assessment
    complexity_score: int = 0  # 0-100, higher is better
    risk_level: str = "low"  # low/medium/high/very_high
    
    # Anti-patterns detected
    anti_patterns: List[Dict[str, Any]] = field(default_factory=list)


class MetricsCalculator:
    """Calculate code complexity metrics."""
    
    # Thresholds for quality assessment
    CC_THRESHOLDS = {
        "low": 5,
        "medium": 10,
        "high": 20,
        "very_high": 50
    }
    
    MI_THRESHOLDS = {
        "A": 20,  # > 20: Very maintainable
        "B": 10,  # 10-20: Moderately maintainable
        "C": 0,   # 0-10: Difficult to maintain
        "F": -float("inf")  # < 0: Unmaintainable
    }
    
    def calculate_metrics(self, code: str) -> CodeMetrics:
        """Calculate comprehensive metrics for Python code.
        
        Args:
            code: Python source code
            
        Returns:
            CodeMetrics object with all calculated metrics
        """
        try:
            # Get raw metrics
            raw = analyze(code)
            
            # Get complexity metrics
            cc_results = cc_visit(code)
            
            # Get Halstead metrics
            try:
                halstead_result = h_visit(code)
                if halstead_result and halstead_result.total:
                    h_total = halstead_result.total
                    halstead = HalsteadMetrics(
                        volume=h_total.volume,
                        difficulty=h_total.difficulty,
                        effort=h_total.effort,
                        time=h_total.time,
                        bugs=h_total.bugs,
                        vocabulary=h_total.vocabulary,
                        length=h_total.length
                    )
                else:
                    halstead = None
            except Exception as e:
                logger.warning(f"Failed to calculate Halstead metrics: {e}")
                halstead = None
            
            # Calculate maintainability index
            mi = mi_compute(
                halstead.volume if halstead else 0,
                cc_results[0].complexity if cc_results else 1,
                raw.lloc,
                raw.comments / max(raw.loc, 1) * 100
            )
            
            # Process function metrics
            functions = []
            total_complexity = 0
            
            for func in cc_results:
                # In radon, cc_visit returns list of complexity objects
                # Filter out Class objects, keep only Function objects (including methods)
                if func.__class__.__name__ == 'Function':
                    func_metrics = self._analyze_function(func, code)
                    functions.append(func_metrics)
                    total_complexity += func.complexity
            
            avg_complexity = total_complexity / max(len(functions), 1)
            
            # Build metrics object
            metrics = CodeMetrics(
                lines_of_code=raw.loc,
                logical_lines=raw.lloc,
                source_lines=raw.sloc,
                blank_lines=raw.blank,
                comment_lines=raw.comments,
                multi_line_strings=raw.multi,
                cyclomatic_complexity=total_complexity,
                average_complexity=avg_complexity,
                complexity_rank=self._get_complexity_rank(avg_complexity),
                maintainability_index=mi,
                maintainability_grade=mi_rank(mi),
                halstead=halstead,
                functions=functions
            )
            
            # Calculate quality assessment
            self._assess_quality(metrics)
            
            # Detect anti-patterns
            self._detect_anti_patterns(metrics, code)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return minimal metrics on error
            return CodeMetrics(
                lines_of_code=len(code.splitlines()),
                logical_lines=0,
                source_lines=0,
                blank_lines=0,
                comment_lines=0,
                multi_line_strings=0,
                cyclomatic_complexity=0,
                average_complexity=0,
                complexity_rank="F",
                maintainability_index=0,
                maintainability_grade="F"
            )
    
    def calculate_file_metrics(self, file_path: str) -> CodeMetrics:
        """Calculate metrics for a Python file.
        
        Args:
            file_path: Path to Python file
            
        Returns:
            CodeMetrics object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            return self.calculate_metrics(code)
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def calculate_function_metrics(self, function_code: str) -> FunctionMetrics:
        """Calculate metrics for a single function.
        
        Args:
            function_code: Python function code
            
        Returns:
            FunctionMetrics object
        """
        try:
            # Ensure it's a valid function by parsing
            tree = ast.parse(function_code)
            if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
                raise ValueError("Input is not a valid function")
            
            func_node = tree.body[0]
            
            # Get complexity
            cc_results = cc_visit(function_code)
            if not cc_results:
                complexity = 1
                rank = "A"
            else:
                complexity = cc_results[0].complexity
                rank = cc_rank(complexity)
            
            # Count lines
            lines = function_code.splitlines()
            line_count = len(lines)
            
            # Count parameters
            param_count = len(func_node.args.args)
            
            # Calculate nesting depth
            max_depth = self._calculate_max_nesting(func_node)
            
            return FunctionMetrics(
                name=func_node.name,
                line_start=1,
                line_end=line_count,
                cyclomatic_complexity=complexity,
                complexity_rank=rank,
                parameter_count=param_count,
                lines_of_code=line_count,
                max_nesting_depth=max_depth,
                is_long=line_count > 50,
                is_complex=complexity > 10,
                has_many_params=param_count > 5,
                is_deeply_nested=max_depth > 4
            )
            
        except Exception as e:
            logger.error(f"Error analyzing function: {e}")
            raise
    
    def _analyze_function(self, func_result: Any, code: str) -> FunctionMetrics:
        """Analyze a function from radon results."""
        # Parse AST to get additional info
        param_count = 0
        max_depth = 0
        
        try:
            # Parse the entire code to find the function
            tree = ast.parse(code)
            
            # Find the function node by name and line number
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_result.name:
                    # Check if line numbers match (approximately)
                    if hasattr(node, 'lineno') and abs(node.lineno - func_result.lineno) <= 1:
                        param_count = len(node.args.args)
                        max_depth = self._calculate_max_nesting(node)
                        break
                        
        except Exception as e:
            logger.debug(f"Failed to parse function {func_result.name}: {e}")
        
        line_count = func_result.endline - func_result.lineno + 1
        
        return FunctionMetrics(
            name=func_result.name,
            line_start=func_result.lineno,
            line_end=func_result.endline,
            cyclomatic_complexity=func_result.complexity,
            complexity_rank=cc_rank(func_result.complexity),
            parameter_count=param_count,
            lines_of_code=line_count,
            max_nesting_depth=max_depth,
            is_long=line_count > 50,
            is_complex=func_result.complexity >= 10,
            has_many_params=param_count > 5,
            is_deeply_nested=max_depth > 4
        )
    
    def _calculate_max_nesting(self, node: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth in AST."""
        max_depth = depth
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)):
                # These create new nesting levels
                child_depth = depth + 1
                for grandchild in ast.iter_child_nodes(child):
                    grandchild_depth = self._calculate_max_nesting(grandchild, child_depth)
                    max_depth = max(max_depth, grandchild_depth)
        
        return max_depth
    
    def _get_complexity_rank(self, complexity: float) -> str:
        """Convert complexity score to letter grade."""
        if complexity <= 5:
            return "A"
        elif complexity <= 10:
            return "B"
        elif complexity <= 20:
            return "C"
        elif complexity <= 30:
            return "D"
        elif complexity <= 40:
            return "E"
        else:
            return "F"
    
    def _assess_quality(self, metrics: CodeMetrics) -> None:
        """Assess overall code quality and risk level."""
        # Calculate complexity score (0-100, higher is better)
        score = 100
        
        # Deduct for high complexity
        if metrics.average_complexity > 10:
            score -= min(30, (metrics.average_complexity - 10) * 2)
        elif metrics.average_complexity > 5:
            score -= min(15, (metrics.average_complexity - 5) * 2)
        elif metrics.average_complexity > 3:
            score -= min(10, (metrics.average_complexity - 3) * 2)
        
        # Deduct for poor maintainability
        if metrics.maintainability_index < 20:
            score -= 30
        elif metrics.maintainability_index < 40:
            score -= 20
        elif metrics.maintainability_index < 60:
            score -= 10
        elif metrics.maintainability_index < 70:
            score -= 5
        
        # Deduct for Halstead complexity
        if metrics.halstead:
            if metrics.halstead.difficulty > 10:
                score -= 10
            elif metrics.halstead.difficulty > 5:
                score -= 5
            elif metrics.halstead.difficulty > 2:
                score -= 2
        
        # Deduct for anti-patterns in functions
        for func in metrics.functions:
            if func.is_complex:
                score -= 5
            if func.is_long:
                score -= 3
            if func.has_many_params:
                score -= 2
            if func.is_deeply_nested:
                score -= 3
            
            # Additional penalties for really bad practices
            if func.cyclomatic_complexity > 5:
                score -= 2
            if func.parameter_count > 3:
                score -= 1
            if func.max_nesting_depth > 3:
                score -= 2
        
        metrics.complexity_score = max(0, score)
        
        # Determine risk level
        if metrics.average_complexity <= self.CC_THRESHOLDS["low"]:
            metrics.risk_level = "low"
        elif metrics.average_complexity <= self.CC_THRESHOLDS["medium"]:
            metrics.risk_level = "medium"
        elif metrics.average_complexity <= self.CC_THRESHOLDS["high"]:
            metrics.risk_level = "high"
        else:
            metrics.risk_level = "very_high"
    
    def _detect_anti_patterns(self, metrics: CodeMetrics, code: str) -> None:
        """Detect common anti-patterns in code."""
        anti_patterns = []
        
        # Check for long functions
        for func in metrics.functions:
            if func.is_long:
                anti_patterns.append({
                    "type": "long_function",
                    "name": func.name,
                    "message": f"Function '{func.name}' is too long ({func.lines_of_code} lines)",
                    "severity": "warning",
                    "line": func.line_start
                })
            
            if func.is_complex:
                anti_patterns.append({
                    "type": "complex_function", 
                    "name": func.name,
                    "message": f"Function '{func.name}' has high complexity (CC={func.cyclomatic_complexity})",
                    "severity": "error" if func.cyclomatic_complexity > 20 else "warning",
                    "line": func.line_start
                })
            
            if func.has_many_params:
                anti_patterns.append({
                    "type": "too_many_parameters",
                    "name": func.name,
                    "message": f"Function '{func.name}' has too many parameters ({func.parameter_count})",
                    "severity": "warning",
                    "line": func.line_start
                })
            
            if func.is_deeply_nested:
                anti_patterns.append({
                    "type": "deep_nesting",
                    "name": func.name,
                    "message": f"Function '{func.name}' has deep nesting (depth={func.max_nesting_depth})",
                    "severity": "warning",
                    "line": func.line_start
                })
        
        # Check file-level anti-patterns
        if metrics.lines_of_code > 500:
            anti_patterns.append({
                "type": "large_file",
                "name": "file",
                "message": f"File is too large ({metrics.lines_of_code} lines)",
                "severity": "warning",
                "line": 1
            })
        
        if metrics.comment_lines / max(metrics.lines_of_code, 1) < 0.1:
            anti_patterns.append({
                "type": "insufficient_comments",
                "name": "file",
                "message": "File has insufficient comments (< 10%)",
                "severity": "info",
                "line": 1
            })
        
        metrics.anti_patterns = anti_patterns


# Convenience functions
def calculate_metrics(code: str) -> CodeMetrics:
    """Calculate metrics for Python code."""
    calculator = MetricsCalculator()
    return calculator.calculate_metrics(code)


def calculate_file_metrics(file_path: str) -> CodeMetrics:
    """Calculate metrics for a Python file."""
    calculator = MetricsCalculator()
    return calculator.calculate_file_metrics(file_path)


def calculate_function_metrics(function_code: str) -> FunctionMetrics:
    """Calculate metrics for a single function."""
    calculator = MetricsCalculator()
    return calculator.calculate_function_metrics(function_code)