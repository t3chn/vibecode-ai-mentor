"""Demonstrate chunking on a real project file."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer.chunker import ChunkStrategy, CodeChunker


async def chunk_parser_file():
    """Chunk the parser.py file to show real-world usage."""
    chunker = CodeChunker()
    parser_path = Path("src/analyzer/parser.py")
    
    if not parser_path.exists():
        print(f"File not found: {parser_path}")
        return
    
    print("=== Chunking parser.py ===\n")
    
    # Load the file
    with open(parser_path, 'r') as f:
        code = f.read()
    
    print(f"File size: {len(code)} characters")
    print(f"Lines: {len(code.splitlines())}")
    print(f"Total tokens: {chunker.count_tokens(code)}")
    
    # Use smart chunking
    print("\n--- Smart Chunking Strategy ---")
    chunks = chunker.chunk_code(code, ChunkStrategy.SMART_CHUNK)
    
    print(f"\nCreated {len(chunks)} chunks:")
    
    # Group chunks by type
    by_type = {}
    for chunk in chunks:
        by_type.setdefault(chunk.chunk_type, []).append(chunk)
    
    # Show summary by type
    for chunk_type, type_chunks in sorted(by_type.items()):
        print(f"\n{chunk_type.upper()} chunks ({len(type_chunks)}):")
        
        for chunk in type_chunks[:5]:  # Show first 5 of each type
            # Extract name from metadata
            name = chunk.metadata.get('name', 'N/A')
            if chunk.chunk_type == 'class':
                methods = chunk.metadata.get('methods', [])
                name = f"{name} ({len(methods)} methods)"
            
            print(f"  - Lines {chunk.start_line:3d}-{chunk.end_line:3d} "
                  f"({chunk.token_count:4d} tokens): {name}")
        
        if len(type_chunks) > 5:
            print(f"  ... and {len(type_chunks) - 5} more")
    
    # Show how chunks cover the file
    print("\n--- File Coverage ---")
    covered_lines = set()
    for chunk in chunks:
        for line in range(chunk.start_line, chunk.end_line + 1):
            covered_lines.add(line)
    
    total_lines = len(code.splitlines())
    coverage = len(covered_lines) / total_lines * 100 if total_lines > 0 else 0
    
    print(f"Lines covered: {len(covered_lines)}/{total_lines} ({coverage:.1f}%)")
    
    # Find any gaps
    all_lines = set(range(1, total_lines + 1))
    uncovered = all_lines - covered_lines
    if uncovered:
        print(f"Uncovered lines: {sorted(uncovered)[:10]}...")
    
    # Test chunk reconstruction
    print("\n--- Chunk Integrity Test ---")
    
    # Try to reconstruct specific functions
    function_chunks = [c for c in chunks if c.chunk_type == 'function']
    if function_chunks:
        test_chunk = function_chunks[0]
        print(f"\nTesting chunk: {test_chunk.metadata.get('name')} "
              f"(lines {test_chunk.start_line}-{test_chunk.end_line})")
        
        # Verify the chunk contains what we expect
        content_lines = test_chunk.content.split('\n')
        print(f"  First line: {content_lines[0]}")
        print(f"  Last line: {content_lines[-1] if content_lines else 'N/A'}")
        print(f"  Contains 'def': {'def' in test_chunk.content}")
        print(f"  Token count accurate: {chunker.count_tokens(test_chunk.content) == test_chunk.token_count}")


if __name__ == "__main__":
    asyncio.run(chunk_parser_file())