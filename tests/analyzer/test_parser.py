"""Tests for the tree-sitter Python parser."""

import pytest
from pathlib import Path

from src.analyzer.parser import PythonParser


class TestPythonParser:
    """Test cases for PythonParser."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance."""
        return PythonParser()

    @pytest.fixture
    def sample_code(self):
        """Sample Python code for testing."""
        return '''
import os
from typing import List, Optional
import numpy as np

def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@decorator
def complex_function(
    name: str,
    values: List[int],
    optional: Optional[str] = None
) -> dict:
    """A more complex function with decorators."""
    result = {"name": name, "sum": sum(values)}
    if optional:
        result["optional"] = optional
    return result

class Calculator:
    """A simple calculator class."""
    
    def __init__(self, name: str):
        """Initialize calculator."""
        self.name = name
        self.history = []
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers and store in history."""
        result = a + b
        self.history.append(result)
        return result
    
    @property
    def last_result(self) -> Optional[float]:
        """Get the last calculated result."""
        return self.history[-1] if self.history else None

class AdvancedCalculator(Calculator):
    """Calculator with more operations."""
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
'''

    def test_parse_code(self, parser, sample_code):
        """Test parsing code string."""
        tree = parser.parse_code(sample_code)
        assert tree is not None
        assert tree.root_node is not None

    def test_extract_functions(self, parser, sample_code):
        """Test extracting functions from code."""
        parser.parse_code(sample_code)
        functions = parser.extract_functions()
        
        assert len(functions) == 5  # 2 module-level + 3 class methods
        
        # Check module-level functions
        add_func = next(f for f in functions if f["name"] == "add")
        assert add_func["parameters"] == [
            {"name": "a", "type": "int"},
            {"name": "b", "type": "int"}
        ]
        assert add_func["return_type"] == "int"
        assert add_func["docstring"] == "Add two numbers."
        
        complex_func = next(f for f in functions if f["name"] == "complex_function")
        assert complex_func["decorators"] == ["decorator"]
        assert len(complex_func["parameters"]) == 3

    def test_extract_classes(self, parser, sample_code):
        """Test extracting classes from code."""
        parser.parse_code(sample_code)
        classes = parser.extract_classes()
        
        assert len(classes) == 2
        
        # Check Calculator class
        calc_class = next(c for c in classes if c["name"] == "Calculator")
        assert calc_class["docstring"] == "A simple calculator class."
        assert set(calc_class["methods"]) == {"__init__", "add", "last_result"}
        assert set(calc_class["attributes"]) == {"name", "history"}
        
        # Check AdvancedCalculator class
        adv_class = next(c for c in classes if c["name"] == "AdvancedCalculator")
        assert adv_class["bases"] == ["Calculator"]
        assert "multiply" in adv_class["methods"]

    def test_extract_imports(self, parser, sample_code):
        """Test extracting imports from code."""
        parser.parse_code(sample_code)
        imports = parser.extract_imports()
        
        assert len(imports) == 3
        
        # Check different import types
        os_import = next(i for i in imports if i["module"] == "os")
        assert os_import["type"] == "import_statement"
        
        typing_import = next(i for i in imports if i["module"] == "typing")
        assert typing_import["type"] == "import_from_statement"
        assert set(typing_import["names"]) == {"List", "Optional"}

    def test_get_node_location(self, parser):
        """Test getting node location."""
        code = "def test():\n    pass"
        tree = parser.parse_code(code)
        
        # Find the function definition node
        func_node = None
        for node in tree.root_node.children:
            if node.type == "function_definition":
                func_node = node
                break
        
        assert func_node is not None
        location = parser.get_node_location(func_node)
        assert location["start_line"] == 1
        assert location["end_line"] == 2

    def test_empty_code(self, parser):
        """Test parsing empty code."""
        parser.parse_code("")
        assert parser.extract_functions() == []
        assert parser.extract_classes() == []
        assert parser.extract_imports() == []

    def test_malformed_code(self, parser):
        """Test parsing malformed code (should not crash)."""
        malformed = "def broken_func(\n    return None"
        tree = parser.parse_code(malformed)
        assert tree is not None  # Parser should handle errors gracefully

    @pytest.mark.asyncio
    async def test_parse_file(self, parser, tmp_path):
        """Test parsing a file asynchronously."""
        # Create a temporary file
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    return 'world'")
        
        tree = await parser.parse_file(test_file)
        assert tree is not None
        
        functions = parser.extract_functions()
        assert len(functions) == 1
        assert functions[0]["name"] == "hello"

    @pytest.mark.asyncio
    async def test_parse_nonexistent_file(self, parser):
        """Test parsing a non-existent file."""
        with pytest.raises(FileNotFoundError):
            await parser.parse_file("/nonexistent/file.py")

    def test_complex_decorators(self, parser):
        """Test handling complex decorators."""
        code = """
@app.route("/api/test")
@requires_auth
@cache(timeout=300)
def api_endpoint():
    pass
"""
        parser.parse_code(code)
        functions = parser.extract_functions()
        assert len(functions) == 1
        assert len(functions[0]["decorators"]) == 3

    def test_type_annotations(self, parser):
        """Test handling various type annotations."""
        code = """
from typing import Dict, Union, Callable

def process(
    data: Dict[str, Union[int, str]],
    callback: Callable[[str], None]
) -> List[Dict[str, Any]]:
    pass
"""
        parser.parse_code(code)
        functions = parser.extract_functions()
        assert len(functions) == 1
        func = functions[0]
        assert func["parameters"][0]["type"] == "Dict[str, Union[int, str]]"
        assert func["parameters"][1]["type"] == "Callable[[str], None]"
        assert func["return_type"] == "List[Dict[str, Any]]"

    def test_extract_global_variables(self, parser):
        """Test extracting global variables."""
        code = """
# Configuration constants
DEBUG = True
API_KEY = "secret-key-123"
MAX_RETRIES = 3

def some_function():
    local_var = 42  # This should not be extracted
    return local_var

GLOBAL_CONFIG = {
    "host": "localhost",
    "port": 8080
}
"""
        parser.parse_code(code)
        global_vars = parser.extract_global_variables()
        
        assert len(global_vars) == 4
        
        # Check specific variables
        debug_var = next(v for v in global_vars if v["name"] == "DEBUG")
        assert debug_var["value"] == "True"
        
        api_key_var = next(v for v in global_vars if v["name"] == "API_KEY")
        assert api_key_var["value"] == '"secret-key-123"'
        
        config_var = next(v for v in global_vars if v["name"] == "GLOBAL_CONFIG")
        assert "localhost" in config_var["value"]

    def test_extract_comments(self, parser):
        """Test extracting comments."""
        code = """
# This is a module-level comment
import os

# Configuration section
DEBUG = True  # Enable debug mode

def process_data():
    # Step 1: Validate input
    pass
    # Step 2: Process data
    # This is a multi-line explanation
    pass
"""
        parser.parse_code(code)
        comments = parser.extract_comments()
        
        assert len(comments) >= 5
        
        # Check first comment
        assert any("module-level comment" in c["text"] for c in comments)
        assert any("Configuration section" in c["text"] for c in comments)
        assert any("Enable debug mode" in c["text"] for c in comments)