"""Debug parser chunking issue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer.chunker import ChunkStrategy, CodeChunker
from src.analyzer.parser import PythonParser


def debug_parser_chunking():
    """Debug the parser.py chunking."""
    
    # Load parser.py
    with open("src/analyzer/parser.py", 'r') as f:
        code = f.read()
    
    # Parse it
    parser = PythonParser()
    parser.parse_code(code)
    
    # Extract classes
    classes = parser.extract_classes()
    print(f"Found {len(classes)} classes:")
    for cls in classes:
        print(f"  - {cls['name']} at lines {cls['location']['start_line']}-{cls['location']['end_line']}")
        print(f"    Size: {cls['location']['end_line'] - cls['location']['start_line'] + 1} lines")
        print(f"    Methods: {len(cls['methods'])}")
    
    # Check if the class is too large
    chunker = CodeChunker()
    if classes:
        cls = classes[0]
        class_text = '\n'.join(code.split('\n')[cls['location']['start_line']-1:cls['location']['end_line']])
        tokens = chunker.count_tokens(class_text)
        print(f"\nClass '{cls['name']}' has {tokens} tokens")
        print(f"Max chunk size: {chunker.max_chunk_size}")
        print(f"Will be split: {tokens > chunker.max_chunk_size}")
    
    # Test by_class strategy directly
    print("\n--- Testing by_class strategy ---")
    chunks = chunker.chunk_code(code, ChunkStrategy.BY_CLASS)
    
    print(f"Created {len(chunks)} chunks")
    
    # Check if PythonParser class was split
    parser_chunks = [c for c in chunks if 'PythonParser' in c.metadata.get('name', '')]
    print(f"\nPythonParser chunks: {len(parser_chunks)}")
    for i, chunk in enumerate(parser_chunks):
        print(f"  Chunk {i+1}: lines {chunk.start_line}-{chunk.end_line}, "
              f"tokens: {chunk.token_count}, metadata: {chunk.metadata}")


if __name__ == "__main__":
    debug_parser_chunking()