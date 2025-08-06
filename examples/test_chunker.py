"""Simple standalone test of the chunker without requiring full environment setup."""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer.chunker import ChunkStrategy, CodeChunk, CodeChunker
from src.analyzer.parser import PythonParser


def test_chunker():
    """Test the chunker without requiring environment setup."""
    
    # Sample code
    test_code = '''
"""Example module with different code structures."""

import os
import json
from typing import List, Dict

DEFAULT_TIMEOUT = 30

class DataProcessor:
    """Process various data formats."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data = []
    
    def process_json(self, data: str) -> Dict:
        """Process JSON data."""
        try:
            result = json.loads(data)
            self.data.append(result)
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {}
    
    def get_summary(self) -> str:
        """Get summary of processed data."""
        return f"Processed {len(self.data)} items"

def load_file(path: str) -> str:
    """Load file contents."""
    with open(path, 'r') as f:
        return f.read()

def save_file(path: str, content: str) -> None:
    """Save content to file."""
    with open(path, 'w') as f:
        f.write(content)

# Simple usage
if __name__ == "__main__":
    processor = DataProcessor({})
    data = '{"test": "value"}'
    processor.process_json(data)
'''

    chunker = CodeChunker()
    
    print("=== Chunker Test ===\n")
    
    # Test each strategy
    strategies = [
        ChunkStrategy.BY_FUNCTION,
        ChunkStrategy.BY_CLASS,
        ChunkStrategy.SMART_CHUNK,
    ]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.value} ---")
        try:
            chunks = chunker.chunk_code(test_code, strategy)
            print(f"Created {len(chunks)} chunks:")
            
            for i, chunk in enumerate(chunks, 1):
                print(f"\n  Chunk {i}:")
                print(f"    Type: {chunk.chunk_type}")
                print(f"    Lines: {chunk.start_line}-{chunk.end_line}")
                print(f"    Tokens: {chunk.token_count}")
                print(f"    Metadata: {chunk.metadata}")
                
                # Show first line of content
                first_line = chunk.content.split('\n')[0]
                print(f"    First line: {first_line[:60]}...")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # Test token counting
    print("\n\n--- Token Counting Test ---")
    sample_texts = [
        "def hello(): pass",
        "class MyClass:\n    def __init__(self):\n        pass",
        "# This is a comment\nimport os\nprint('Hello')",
    ]
    
    for text in sample_texts:
        tokens = chunker.count_tokens(text)
        print(f"Text: '{text[:30]}...' => {tokens} tokens")


if __name__ == "__main__":
    test_chunker()