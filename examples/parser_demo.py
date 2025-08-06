"""Demo script showing how to use the PythonParser."""

import asyncio
from pathlib import Path
from pprint import pprint

from src.analyzer.parser import PythonParser


async def demo_parser():
    """Demonstrate parser functionality."""
    # Create parser instance
    parser = PythonParser()
    
    # Sample code to analyze
    sample_code = '''
import os
from typing import List, Optional

# Global configuration
DEFAULT_ENCODING = "utf-8"
MAX_FILE_SIZE = 1024 * 1024  # 1MB

class FileHandler:
    """Handles file operations."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.files_processed = 0
    
    async def process_files(self, pattern: str) -> List[str]:
        """Process files matching the pattern."""
        results = []
        for file in self.base_path.glob(pattern):
            if file.is_file():
                results.append(str(file))
                self.files_processed += 1
        return results
    
    @property
    def stats(self) -> dict:
        """Get processing statistics."""
        return {"files_processed": self.files_processed}

def validate_path(path: str) -> bool:
    """Check if path exists and is accessible."""
    return Path(path).exists()
'''
    
    # Parse the code
    print("Parsing code...")
    parser.parse_code(sample_code)
    
    # Extract and display functions
    print("\n=== Functions ===")
    functions = parser.extract_functions()
    for func in functions:
        print(f"\nFunction: {func['name']}")
        print(f"  Location: lines {func['location']['start_line']}-{func['location']['end_line']}")
        print(f"  Parameters: {func['parameters']}")
        print(f"  Return type: {func['return_type']}")
        if func['docstring']:
            print(f"  Docstring: {func['docstring']}")
        if func['decorators']:
            print(f"  Decorators: {func['decorators']}")
    
    # Extract and display classes
    print("\n=== Classes ===")
    classes = parser.extract_classes()
    for cls in classes:
        print(f"\nClass: {cls['name']}")
        print(f"  Location: lines {cls['location']['start_line']}-{cls['location']['end_line']}")
        print(f"  Methods: {cls['methods']}")
        print(f"  Attributes: {cls['attributes']}")
        if cls['docstring']:
            print(f"  Docstring: {cls['docstring']}")
        if cls['bases']:
            print(f"  Inherits from: {cls['bases']}")
    
    # Extract and display imports
    print("\n=== Imports ===")
    imports = parser.extract_imports()
    for imp in imports:
        print(f"\n{imp['type']}:")
        print(f"  Module: {imp['module']}")
        if imp['names']:
            print(f"  Names: {imp['names']}")
    
    # Extract and display global variables
    print("\n=== Global Variables ===")
    global_vars = parser.extract_global_variables()
    for var in global_vars:
        print(f"\n{var['name']} = {var['value']}")
        print(f"  Location: line {var['location']['start_line']}")
    
    # Extract and display comments
    print("\n=== Comments ===")
    comments = parser.extract_comments()
    for comment in comments[:5]:  # Show first 5 comments
        print(f"\nLine {comment['location']['start_line']}: {comment['text']}")
    if len(comments) > 5:
        print(f"\n... and {len(comments) - 5} more comments")
    
    # Demo file parsing
    print("\n=== File Parsing Demo ===")
    # Parse this file itself
    this_file = Path(__file__)
    if this_file.exists():
        print(f"Parsing {this_file.name}...")
        await parser.parse_file(this_file)
        functions = parser.extract_functions()
        print(f"Found {len(functions)} functions in this file")


if __name__ == "__main__":
    asyncio.run(demo_parser())