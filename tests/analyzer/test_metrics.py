"""Tests for code complexity metrics calculator."""

import pytest

from src.analyzer.metrics import (
    CodeMetrics,
    FunctionMetrics,
    MetricsCalculator,
    calculate_file_metrics,
    calculate_function_metrics,
    calculate_metrics,
)


class TestMetricsCalculator:
    """Test the MetricsCalculator class."""
    
    def test_simple_function_metrics(self):
        """Test metrics for a simple function."""
        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''
        metrics = calculate_metrics(code)
        
        assert metrics.lines_of_code == 4
        assert metrics.cyclomatic_complexity == 1
        assert metrics.complexity_rank == "A"
        assert metrics.maintainability_grade in ["A", "B"]
        assert metrics.risk_level == "low"
        assert len(metrics.functions) == 1
        
        func = metrics.functions[0]
        assert func.name == "add"
        assert func.cyclomatic_complexity == 1
        assert func.parameter_count == 2
        assert not func.is_complex
        assert not func.is_long
    
    def test_complex_function_metrics(self):
        """Test metrics for a complex function."""
        code = '''
def process_data(data, options=None):
    """Process data with various conditions."""
    if not data:
        return []
    
    results = []
    for item in data:
        if isinstance(item, dict):
            if "value" in item:
                if item["value"] > 10:
                    if options and options.get("double"):
                        results.append(item["value"] * 2)
                    else:
                        results.append(item["value"])
                else:
                    results.append(0)
            else:
                results.append(-1)
        elif isinstance(item, (int, float)):
            results.append(item)
        else:
            try:
                results.append(float(item))
            except ValueError:
                results.append(None)
    
    return results
'''
        metrics = calculate_metrics(code)
        
        assert metrics.average_complexity > 5
        assert metrics.risk_level in ["medium", "high"]
        assert len(metrics.functions) == 1
        
        func = metrics.functions[0]
        assert func.name == "process_data"
        assert func.is_complex  # High cyclomatic complexity
        assert func.is_deeply_nested  # Deep nesting
    
    def test_anti_patterns_detection(self):
        """Test detection of common anti-patterns."""
        code = '''
def very_long_function(a, b, c, d, e, f, g):
    """Function with many parameters and many lines."""
    result = 0
    # Simulate a very long function
''' + '\n'.join([f'    result += {i}' for i in range(60)]) + '''
    return result
'''
        metrics = calculate_metrics(code)
        
        assert len(metrics.anti_patterns) >= 2
        
        # Check for specific anti-patterns
        pattern_types = {p["type"] for p in metrics.anti_patterns}
        assert "long_function" in pattern_types
        assert "too_many_parameters" in pattern_types
    
    def test_function_metrics_standalone(self):
        """Test calculating metrics for a single function."""
        func_code = '''
def fibonacci(n):
    """Calculate fibonacci number."""
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
'''
        func_metrics = calculate_function_metrics(func_code)
        
        assert func_metrics.name == "fibonacci"
        assert func_metrics.cyclomatic_complexity == 2
        assert func_metrics.parameter_count == 1
        assert not func_metrics.is_long
        assert not func_metrics.has_many_params
    
    def test_maintainability_index(self):
        """Test maintainability index calculation."""
        # Well-documented, simple code
        good_code = '''
def calculate_area(radius):
    """
    Calculate the area of a circle.
    
    Args:
        radius: The radius of the circle
        
    Returns:
        The area of the circle
    """
    import math
    return math.pi * radius * radius
'''
        good_metrics = calculate_metrics(good_code)
        
        # Complex, undocumented code
        bad_code = '''
def x(a,b,c,d):
    if a>0:
        if b>0:
            if c>0:
                if d>0:
                    return a+b+c+d
                else:
                    return a+b+c
            else:
                return a+b
        else:
            return a
    else:
        return 0
'''
        bad_metrics = calculate_metrics(bad_code)
        
        assert good_metrics.maintainability_index > bad_metrics.maintainability_index
        assert good_metrics.complexity_score > bad_metrics.complexity_score
    
    def test_halstead_metrics(self):
        """Test Halstead complexity metrics."""
        code = '''
def quadratic(a, b, c, x):
    """Calculate quadratic equation."""
    return a * x * x + b * x + c
'''
        metrics = calculate_metrics(code)
        
        assert metrics.halstead is not None
        assert metrics.halstead.volume > 0
        assert metrics.halstead.difficulty > 0
        assert metrics.halstead.effort > 0
        assert metrics.halstead.vocabulary > 0
    
    def test_empty_code(self):
        """Test metrics for empty code."""
        metrics = calculate_metrics("")
        
        assert metrics.lines_of_code == 0
        assert metrics.cyclomatic_complexity == 0
        assert len(metrics.functions) == 0
    
    def test_code_with_classes(self):
        """Test metrics for code with classes."""
        code = '''
class Calculator:
    """Simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = 0
        for _ in range(b):
            result += a
        return result
'''
        metrics = calculate_metrics(code)
        
        assert len(metrics.functions) == 2
        assert any(f.name == "add" for f in metrics.functions)
        assert any(f.name == "multiply" for f in metrics.functions)
        
        # multiply should have higher complexity due to loop
        multiply_func = next(f for f in metrics.functions if f.name == "multiply")
        add_func = next(f for f in metrics.functions if f.name == "add")
        assert multiply_func.cyclomatic_complexity > add_func.cyclomatic_complexity