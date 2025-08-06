"""Advanced demonstration of the chunking capabilities."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analyzer.chunker import ChunkStrategy, CodeChunker


# Large example code to demonstrate different chunking scenarios
LARGE_CODE_EXAMPLE = '''
"""
Complex module demonstrating various Python patterns.
This module is designed to test different chunking strategies.
"""

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Union, TypeVar

# Constants
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
CACHE_SIZE = 1000

# Type variables
T = TypeVar('T')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Configuration:
    """Application configuration."""
    host: str = "localhost"
    port: int = 8000
    debug: bool = False
    workers: int = 4


class BaseProcessor(ABC):
    """Abstract base class for processors."""
    
    def __init__(self, config: Configuration):
        self.config = config
        self._cache: Dict[str, any] = {}
        self._initialized = False
    
    @abstractmethod
    async def process(self, data: Dict) -> Dict:
        """Process data asynchronously."""
        pass
    
    @abstractmethod
    def validate(self, data: Dict) -> bool:
        """Validate input data."""
        pass
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.info("Cache cleared")


class DataProcessor(BaseProcessor):
    """Concrete implementation of data processor."""
    
    def __init__(self, config: Configuration):
        super().__init__(config)
        self.processing_count = 0
        self.error_count = 0
    
    async def process(self, data: Dict) -> Dict:
        """Process data with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                # Validate input
                if not self.validate(data):
                    raise ValueError("Invalid data format")
                
                # Check cache
                cache_key = self._get_cache_key(data)
                if cache_key in self._cache:
                    logger.debug(f"Cache hit for {cache_key}")
                    return self._cache[cache_key]
                
                # Process data
                result = await self._do_processing(data)
                
                # Cache result
                if len(self._cache) < CACHE_SIZE:
                    self._cache[cache_key] = result
                
                self.processing_count += 1
                return result
                
            except Exception as e:
                self.error_count += 1
                logger.error(f"Processing error (attempt {attempt + 1}): {e}")
                if attempt == MAX_RETRIES - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    def validate(self, data: Dict) -> bool:
        """Validate that data has required fields."""
        required_fields = ['id', 'type', 'content']
        return all(field in data for field in required_fields)
    
    async def _do_processing(self, data: Dict) -> Dict:
        """Actual processing logic."""
        # Simulate async processing
        await asyncio.sleep(0.1)
        
        # Transform data
        result = {
            'id': data['id'],
            'processed': True,
            'type': data['type'].upper(),
            'content_length': len(str(data.get('content', ''))),
            'timestamp': asyncio.get_event_loop().time()
        }
        
        return result
    
    def _get_cache_key(self, data: Dict) -> str:
        """Generate cache key from data."""
        return f"{data.get('id')}_{data.get('type')}"
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            'processed': self.processing_count,
            'errors': self.error_count,
            'cache_size': len(self._cache)
        }


class BatchProcessor:
    """Process data in batches."""
    
    def __init__(self, processor: DataProcessor, batch_size: int = 10):
        self.processor = processor
        self.batch_size = batch_size
        self.total_processed = 0
    
    async def process_batch(self, items: List[Dict]) -> List[Dict]:
        """Process a batch of items concurrently."""
        tasks = []
        for item in items:
            task = asyncio.create_task(self.processor.process(item))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_results = []
        for result in results:
            if not isinstance(result, Exception):
                successful_results.append(result)
                self.total_processed += 1
            else:
                logger.error(f"Batch processing error: {result}")
        
        return successful_results
    
    async def process_all(self, items: List[Dict]) -> List[Dict]:
        """Process all items in batches."""
        all_results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}")
            
            batch_results = await self.process_batch(batch)
            all_results.extend(batch_results)
            
            # Small delay between batches
            if i + self.batch_size < len(items):
                await asyncio.sleep(0.5)
        
        return all_results


# Utility functions
def create_sample_data(count: int) -> List[Dict]:
    """Create sample data for testing."""
    return [
        {
            'id': f'item_{i}',
            'type': 'test' if i % 2 == 0 else 'demo',
            'content': f'Sample content for item {i}' * (i % 5 + 1)
        }
        for i in range(count)
    ]


async def run_processing_demo():
    """Demonstrate the processing pipeline."""
    # Setup
    config = Configuration(debug=True)
    processor = DataProcessor(config)
    batch_processor = BatchProcessor(processor, batch_size=5)
    
    # Create test data
    test_data = create_sample_data(20)
    
    # Process data
    logger.info("Starting batch processing...")
    results = await batch_processor.process_all(test_data)
    
    # Show results
    logger.info(f"Processed {len(results)} items successfully")
    logger.info(f"Stats: {processor.get_stats()}")
    
    return results


# Main entry point
if __name__ == "__main__":
    # Run the demo
    asyncio.run(run_processing_demo())
'''


async def demonstrate_chunking():
    """Show different chunking strategies on complex code."""
    chunker = CodeChunker()
    
    print("=== Advanced Chunking Demo ===")
    print(f"Code size: {len(LARGE_CODE_EXAMPLE)} characters")
    print(f"Total tokens: {chunker.count_tokens(LARGE_CODE_EXAMPLE)}")
    print()
    
    # Test each strategy
    for strategy in ChunkStrategy:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {strategy.value}")
        print('=' * 60)
        
        chunks = chunker.chunk_code(LARGE_CODE_EXAMPLE, strategy)
        
        # Show statistics
        total_tokens = sum(c.token_count for c in chunks)
        avg_tokens = total_tokens / len(chunks) if chunks else 0
        
        print(f"\nStatistics:")
        print(f"  Total chunks: {len(chunks)}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Average tokens per chunk: {avg_tokens:.1f}")
        print(f"  Min chunk size: {min(c.token_count for c in chunks) if chunks else 0}")
        print(f"  Max chunk size: {max(c.token_count for c in chunks) if chunks else 0}")
        
        # Show chunk distribution
        chunk_types = {}
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
        
        print(f"\nChunk type distribution:")
        for chunk_type, count in sorted(chunk_types.items()):
            print(f"  {chunk_type}: {count}")
        
        # Show some example chunks
        print(f"\nExample chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n  Chunk {i}:")
            print(f"    Type: {chunk.chunk_type}")
            print(f"    Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"    Tokens: {chunk.token_count}")
            print(f"    Metadata: {chunk.metadata}")
            
            # Show content preview
            lines = chunk.content.split('\n')
            preview = lines[0] if lines else ""
            if len(lines) > 1:
                preview += f" ... ({len(lines)} lines total)"
            print(f"    Preview: {preview[:80]}")
    
    # Test sliding window with overlap
    print(f"\n\n{'=' * 60}")
    print("Sliding Window Overlap Analysis")
    print('=' * 60)
    
    chunks = chunker.chunk_code(LARGE_CODE_EXAMPLE, ChunkStrategy.SLIDING_WINDOW)
    
    # Analyze overlap between consecutive chunks
    for i in range(len(chunks) - 1):
        chunk1_lines = set(range(chunks[i].start_line, chunks[i].end_line + 1))
        chunk2_lines = set(range(chunks[i+1].start_line, chunks[i+1].end_line + 1))
        overlap = chunk1_lines & chunk2_lines
        
        if overlap:
            print(f"\nChunks {i+1} and {i+2}:")
            print(f"  Overlap: {len(overlap)} lines (lines {min(overlap)}-{max(overlap)})")
            print(f"  Overlap percentage: {len(overlap) / len(chunk1_lines) * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(demonstrate_chunking())