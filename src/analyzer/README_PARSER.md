# Tree-sitter Python Parser

The `parser.py` module provides a tree-sitter based Python code parser for extracting AST elements.

## Features

- **Functions**: Extract function definitions with parameters, return types, docstrings, and decorators
- **Classes**: Extract class definitions with methods, attributes, inheritance, and docstrings
- **Imports**: Extract import statements (both `import` and `from ... import`)
- **Global Variables**: Extract top-level variable assignments
- **Comments**: Extract all comments from the code
- **Async Support**: Parse files asynchronously with `parse_file()`

## Usage with uv

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install the project
uv pip install -e ".[dev]"
```

## Example Usage

```python
from src.analyzer.parser import PythonParser

# Create parser instance
parser = PythonParser()

# Parse code string
code = '''
import os
from typing import List

# Global config
DEBUG = True

class MyClass:
    """A sample class."""
    
    def __init__(self, name: str):
        self.name = name
    
    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, {self.name}!"

def main():
    """Main function."""
    obj = MyClass("World")
    print(obj.greet())
'''

# Parse the code
parser.parse_code(code)

# Extract elements
functions = parser.extract_functions()
classes = parser.extract_classes()
imports = parser.extract_imports()
global_vars = parser.extract_global_variables()
comments = parser.extract_comments()

# Access parsed data
for func in functions:
    print(f"Function: {func['name']} at line {func['location']['start_line']}")
    print(f"  Parameters: {func['parameters']}")
    print(f"  Docstring: {func['docstring']}")

for cls in classes:
    print(f"Class: {cls['name']}")
    print(f"  Methods: {cls['methods']}")
    print(f"  Attributes: {cls['attributes']}")
```

## Async File Parsing

```python
import asyncio
from pathlib import Path

async def analyze_file(file_path: Path):
    parser = PythonParser()
    
    # Parse file asynchronously
    await parser.parse_file(file_path)
    
    # Extract and process elements
    functions = parser.extract_functions()
    return functions

# Run async
functions = asyncio.run(analyze_file(Path("my_script.py")))
```

## Data Structures

Each extraction method returns a list of dictionaries with the following structure:

### Functions
```python
{
    "name": str,                # Function name
    "parameters": List[Dict],   # [{"name": str, "type": Optional[str]}]
    "return_type": Optional[str],
    "docstring": Optional[str],
    "decorators": List[str],
    "location": Dict,          # {"start_line": int, "end_line": int, ...}
    "text": str                # Full source text
}
```

### Classes
```python
{
    "name": str,               # Class name
    "bases": List[str],        # Base classes
    "docstring": Optional[str],
    "decorators": List[str],
    "methods": List[str],      # Method names
    "attributes": List[str],   # Instance attributes from __init__
    "location": Dict,
    "text": str
}
```

### Imports
```python
{
    "type": str,               # "import_statement" or "import_from_statement"
    "module": str,             # Module name
    "names": List[str],        # Imported names
    "location": Dict,
    "text": str
}
```

### Global Variables
```python
{
    "name": str,               # Variable name
    "value": str,              # Value as text
    "location": Dict,
    "text": str
}
```

### Comments
```python
{
    "text": str,               # Comment text (without #)
    "location": Dict,
    "raw_text": str            # Original comment with #
}
```

## Error Handling

The parser handles errors gracefully:
- Malformed code will still be parsed (tree-sitter is error-tolerant)
- File not found errors are raised with clear messages
- Empty code returns empty lists for all extractions

## Performance Notes

- The parser maintains the parsed tree internally for efficiency
- Multiple extraction methods can be called on the same parsed code
- For large files, consider using async file parsing
- Tree-sitter is very fast and can handle large codebases efficiently