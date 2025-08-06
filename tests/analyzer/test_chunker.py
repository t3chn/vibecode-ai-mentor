"""Tests for the code chunking module."""

import asyncio
from pathlib import Path

import pytest

from src.analyzer.chunker import ChunkStrategy, CodeChunk, CodeChunker


# Sample code for testing
SIMPLE_FUNCTION_CODE = '''
def hello_world():
    """Say hello to the world."""
    print("Hello, World!")

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
'''

CLASS_CODE = '''
import math

class Calculator:
    """A simple calculator class."""
    
    def __init__(self):
        """Initialize the calculator."""
        self.result = 0
    
    def add(self, x: float, y: float) -> float:
        """Add two numbers."""
        self.result = x + y
        return self.result
    
    def multiply(self, x: float, y: float) -> float:
        """Multiply two numbers."""
        self.result = x * y
        return self.result
    
    def divide(self, x: float, y: float) -> float:
        """Divide x by y."""
        if y == 0:
            raise ValueError("Cannot divide by zero")
        self.result = x / y
        return self.result
'''

MIXED_CODE = '''
"""Module for data processing utilities."""

import json
import csv
from typing import List, Dict, Any

# Constants
DEFAULT_ENCODING = "utf-8"
MAX_RETRIES = 3

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding=DEFAULT_ENCODING) as f:
        return json.load(f)

class DataProcessor:
    """Process various data formats."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data = []
    
    def process_csv(self, file_path: str) -> List[dict]:
        """Process CSV file."""
        with open(file_path, 'r', encoding=DEFAULT_ENCODING) as f:
            reader = csv.DictReader(f)
            self.data = list(reader)
        return self.data

def save_json(data: Any, file_path: str) -> None:
    """Save data to JSON file."""
    with open(file_path, 'w', encoding=DEFAULT_ENCODING) as f:
        json.dump(data, f, indent=2)

# Global processor instance
processor = DataProcessor({})
'''


class TestCodeChunker:
    """Test cases for CodeChunker."""
    
    def setup_method(self):
        """Set up test instance."""
        self.chunker = CodeChunker()
    
    def test_chunk_by_function(self):
        """Test function-based chunking."""
        chunks = self.chunker.chunk_code(
            SIMPLE_FUNCTION_CODE, 
            ChunkStrategy.BY_FUNCTION
        )
        
        # Should have 2 chunks for 2 functions
        assert len(chunks) == 2
        
        # Check first function
        assert chunks[0].chunk_type == "function"
        assert chunks[0].metadata["name"] == "hello_world"
        assert "def hello_world" in chunks[0].content
        assert chunks[0].token_count > 0
        
        # Check second function
        assert chunks[1].chunk_type == "function"
        assert chunks[1].metadata["name"] == "add_numbers"
        assert "def add_numbers" in chunks[1].content
    
    def test_chunk_by_class(self):
        """Test class-based chunking."""
        chunks = self.chunker.chunk_code(
            CLASS_CODE,
            ChunkStrategy.BY_CLASS
        )
        
        # Should have at least one class chunk and imports
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert len(class_chunks) >= 1
        
        # Check class chunk
        class_chunk = class_chunks[0]
        assert class_chunk.metadata["name"] == "Calculator"
        assert "class Calculator" in class_chunk.content
        assert all(method in class_chunk.metadata["methods"] 
                  for method in ["__init__", "add", "multiply", "divide"])
    
    def test_sliding_window(self):
        """Test sliding window chunking."""
        # Create a long code sample
        long_code = "\n".join([f"print('Line {i}')" for i in range(100)])
        
        chunks = self.chunker.chunk_code(
            long_code,
            ChunkStrategy.SLIDING_WINDOW
        )
        
        # Should have multiple chunks
        assert len(chunks) > 1
        
        # All chunks should be blocks
        assert all(c.chunk_type == "block" for c in chunks)
        
        # Check overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            chunk1_end = chunks[i].content.split('\n')[-5:]  # Last 5 lines
            chunk2_start = chunks[i + 1].content.split('\n')[:5]  # First 5 lines
            
            # There should be some overlap
            overlap = set(chunk1_end) & set(chunk2_start)
            assert len(overlap) > 0 or chunks[i].end_line == chunks[i + 1].start_line
    
    def test_smart_chunk(self):
        """Test smart adaptive chunking."""
        chunks = self.chunker.chunk_code(
            MIXED_CODE,
            ChunkStrategy.SMART_CHUNK
        )
        
        # Should intelligently chunk based on structure
        assert len(chunks) > 0
        
        # Should have different chunk types
        chunk_types = {c.chunk_type for c in chunks}
        assert len(chunk_types) > 1  # Multiple types
        
        # Class should be kept together if small enough
        class_chunks = [c for c in chunks if c.chunk_type == "class"]
        assert any("DataProcessor" in c.metadata.get("name", "") 
                  for c in class_chunks)
    
    def test_token_counting(self):
        """Test token counting accuracy."""
        text = "def test(): pass"
        token_count = self.chunker.count_tokens(text)
        
        # Should be a reasonable number of tokens
        assert token_count > 0
        assert token_count < 20  # Simple function shouldn't be too many tokens
    
    def test_chunk_size_limits(self):
        """Test that chunks respect size limits."""
        # Create code that would result in various chunk sizes
        chunks = self.chunker.chunk_code(
            MIXED_CODE,
            ChunkStrategy.BY_FUNCTION
        )
        
        for chunk in chunks:
            # Check token counts are within limits (with some tolerance for edge cases)
            assert chunk.token_count <= self.chunker.max_chunk_size * 1.1
            
            # Very small chunks might exist for module-level code
            if chunk.chunk_type in ["function", "class"]:
                assert chunk.token_count >= self.chunker.min_chunk_size * 0.5
    
    def test_metadata_preservation(self):
        """Test that metadata is properly preserved."""
        chunks = self.chunker.chunk_code(
            CLASS_CODE,
            ChunkStrategy.BY_CLASS
        )
        
        class_chunk = next(c for c in chunks if c.chunk_type == "class")
        
        # Check metadata
        assert "name" in class_chunk.metadata
        assert "methods" in class_chunk.metadata
        assert isinstance(class_chunk.metadata["methods"], list)
    
    def test_line_numbers(self):
        """Test that line numbers are correctly tracked."""
        chunks = self.chunker.chunk_code(
            SIMPLE_FUNCTION_CODE,
            ChunkStrategy.BY_FUNCTION
        )
        
        for chunk in chunks:
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            
            # Verify content matches line numbers
            all_lines = SIMPLE_FUNCTION_CODE.split('\n')
            chunk_lines = all_lines[chunk.start_line - 1:chunk.end_line]
            reconstructed = '\n'.join(chunk_lines)
            
            # Content should match (accounting for empty lines)
            assert chunk.content.strip() == reconstructed.strip()
    
    @pytest.mark.asyncio
    async def test_chunk_file(self, tmp_path):
        """Test chunking from file."""
        # Create a test file
        test_file = tmp_path / "test_code.py"
        test_file.write_text(MIXED_CODE)
        
        # Chunk the file
        chunks = await self.chunker.chunk_file(
            test_file,
            ChunkStrategy.SMART_CHUNK
        )
        
        assert len(chunks) > 0
        assert all(isinstance(c, CodeChunk) for c in chunks)
    
    def test_large_function_splitting(self):
        """Test splitting of functions that exceed max chunk size."""
        # Create a very large function
        large_function = '''
def process_data(data):
    """Process large amounts of data."""
    # This is a very long function with many lines
''' + '\n'.join([f'    print("Processing step {i}")' for i in range(200)])
        
        chunks = self.chunker.chunk_code(
            large_function,
            ChunkStrategy.BY_FUNCTION
        )
        
        # Large function should be split into multiple chunks
        function_chunks = [c for c in chunks 
                          if c.metadata.get("name") == "process_data"]
        
        # Should have multiple parts
        assert any("part" in c.metadata for c in function_chunks)
    
    def test_decorator_preservation(self):
        """Test that decorators are included with functions."""
        decorated_code = '''
@staticmethod
@property
def decorated_function():
    """A decorated function."""
    return "result"
'''
        
        chunks = self.chunker.chunk_code(
            decorated_code,
            ChunkStrategy.BY_FUNCTION
        )
        
        # Decorator should be included
        func_chunk = next(c for c in chunks if c.chunk_type == "function")
        assert "@staticmethod" in func_chunk.content
        assert "@property" in func_chunk.content