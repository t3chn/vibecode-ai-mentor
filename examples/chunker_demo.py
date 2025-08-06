"""Demo script showing code chunking capabilities."""

import asyncio
from pathlib import Path

from src.analyzer.chunker import ChunkStrategy, CodeChunker


async def main():
    """Demonstrate different chunking strategies."""
    chunker = CodeChunker()
    
    # Sample code to chunk
    sample_code = '''
"""Sample module for demonstration."""

import os
import sys
from typing import List, Optional

class FileProcessor:
    """Process files with various operations."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.processed_count = 0
    
    def process_file(self, filename: str) -> bool:
        """Process a single file."""
        file_path = self.base_path / filename
        if not file_path.exists():
            return False
        
        # Process the file
        with open(file_path, 'r') as f:
            content = f.read()
            # Do something with content
            self.processed_count += 1
        
        return True
    
    def batch_process(self, filenames: List[str]) -> int:
        """Process multiple files."""
        success_count = 0
        for filename in filenames:
            if self.process_file(filename):
                success_count += 1
        return success_count

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in a directory."""
    path = Path(directory)
    return [str(f) for f in path.rglob("*.py")]

def main():
    """Main entry point."""
    processor = FileProcessor(".")
    files = find_python_files(".")
    processed = processor.batch_process(files)
    print(f"Processed {processed} files")

if __name__ == "__main__":
    main()
'''
    
    print("=== Code Chunking Demo ===\n")
    
    # Test different strategies
    strategies = [
        ChunkStrategy.BY_FUNCTION,
        ChunkStrategy.BY_CLASS,
        ChunkStrategy.SLIDING_WINDOW,
        ChunkStrategy.SMART_CHUNK
    ]
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy.value} ---")
        chunks = chunker.chunk_code(sample_code, strategy)
        
        print(f"Total chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print(f"  Type: {chunk.chunk_type}")
            print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Metadata: {chunk.metadata}")
            
            # Show first 50 chars of content
            preview = chunk.content[:50].replace('\n', '\\n')
            if len(chunk.content) > 50:
                preview += "..."
            print(f"  Preview: {preview}")
    
    # Test with a real file if available
    print("\n\n=== Testing with parser.py ===")
    parser_file = Path("src/analyzer/parser.py")
    if parser_file.exists():
        chunks = await chunker.chunk_file(parser_file, ChunkStrategy.SMART_CHUNK)
        print(f"\nTotal chunks for parser.py: {len(chunks)}")
        
        # Show statistics
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        print(f"Total tokens: {total_tokens}")
        print(f"Average tokens per chunk: {avg_tokens:.1f}")
        
        # Show chunk type distribution
        from collections import Counter
        type_counts = Counter(c.chunk_type for c in chunks)
        print("\nChunk type distribution:")
        for chunk_type, count in type_counts.items():
            print(f"  {chunk_type}: {count}")


if __name__ == "__main__":
    asyncio.run(main())