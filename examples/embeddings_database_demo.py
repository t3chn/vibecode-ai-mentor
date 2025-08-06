#!/usr/bin/env python3
"""
Demo script showing integration between Gemini embeddings and TiDB database.

This script demonstrates:
1. Generating embeddings for code snippets
2. Storing embeddings in TiDB with caching
3. Performing vector similarity search
4. Using batch processing for efficiency

Usage:
    python examples/embeddings_database_demo.py

Requirements:
    - GEMINI_API_KEY environment variable set
    - TiDB connection configured in .env
    - Database schema initialized
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy.ext.asyncio import AsyncSession
from src.core.config import get_settings
from src.db.connection import get_async_session
from src.db.models import CodeSnippet, Repository, EmbeddingCache
from src.embeddings.batch import EmbeddingBatchProcessor
from src.embeddings.gemini import GeminiEmbeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_embedding_storage():
    """Demonstrate basic embedding generation and storage."""
    print("\n=== Basic Embedding Storage Demo ===")
    
    # Sample code snippets
    code_samples = [
        {
            "content": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
""",
            "file_path": "algorithms/sorting.py",
            "language": "python",
            "start_line": 1,
            "end_line": 8,
        },
        {
            "content": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",
            "file_path": "math/factorial.py", 
            "language": "python",
            "start_line": 1,
            "end_line": 4,
        },
        {
            "content": """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
""",
            "file_path": "data_structures/linked_list.py",
            "language": "python", 
            "start_line": 1,
            "end_line": 18,
        },
    ]
    
    try:
        # Initialize embeddings and database
        embeddings = GeminiEmbeddings()
        
        async with get_async_session() as session:
            # Create a demo repository
            repository = Repository(
                name="demo-algorithms",
                url="https://github.com/demo/algorithms",
                total_files=len(code_samples)
            )
            session.add(repository)
            await session.commit()  # Get the ID
            
            print(f"Created repository: {repository.name}")
            
            # Process code snippets
            snippets = []
            for sample in code_samples:
                snippet = CodeSnippet(
                    repository_id=repository.id,
                    content=sample["content"],
                    file_path=sample["file_path"],
                    language=sample["language"],
                    start_line=sample["start_line"],
                    end_line=sample["end_line"],
                )
                snippets.append(snippet)
                session.add(snippet)
            
            await session.commit()
            print(f"Created {len(snippets)} code snippets")
            
            # Generate embeddings
            print("Generating embeddings...")
            contents = [snippet.content for snippet in snippets]
            embedding_vectors = await embeddings.generate_embeddings_batch(contents)
            
            # Update snippets with embeddings
            for snippet, embedding in zip(snippets, embedding_vectors):
                snippet.embedding = embedding
            
            await session.commit()
            print("Stored embeddings in database")
            
            return repository.id
            
    except Exception as e:
        print(f"Error: {e}")
        return None


async def demo_batch_processing_with_cache():
    """Demonstrate batch processing with caching."""
    print("\n=== Batch Processing with Cache Demo ===")
    
    try:
        embeddings = GeminiEmbeddings()
        
        async with get_async_session() as session:
            # Initialize batch processor
            batch_processor = EmbeddingBatchProcessor(
                provider=embeddings,
                db_session=session,
                batch_size=2
            )
            
            # Test texts (some duplicates to test caching)
            test_texts = [
                "def hello(): print('Hello, World!')",
                "def add(a, b): return a + b",
                "def hello(): print('Hello, World!')",  # Duplicate
                "def multiply(x, y): return x * y",
                "def add(a, b): return a + b",  # Duplicate
            ]
            
            print(f"Processing {len(test_texts)} texts (with duplicates)...")
            
            # First run - no cache hits
            print("\nFirst run (cold cache):")
            embeddings_1 = await batch_processor.process_texts(test_texts)
            print(f"Generated {len(embeddings_1)} embeddings")
            
            # Second run - should have cache hits
            print("\nSecond run (warm cache):")
            embeddings_2 = await batch_processor.process_texts(test_texts)
            print(f"Generated {len(embeddings_2)} embeddings")
            
            # Verify consistency
            for i, (emb1, emb2) in enumerate(zip(embeddings_1, embeddings_2)):
                if emb1 == emb2:
                    print(f"Text {i}: Cache hit (embeddings match)")
                else:
                    print(f"Text {i}: Cache miss (embeddings differ)")
            
            # Get cache statistics
            stats = await batch_processor.get_cache_stats()
            print(f"\nCache statistics: {stats}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False


async def demo_vector_similarity_search():
    """Demonstrate vector similarity search."""
    print("\n=== Vector Similarity Search Demo ===")
    
    try:
        embeddings = GeminiEmbeddings()
        
        async with get_async_session() as session:
            # Create search query
            query_code = "def sort_array(data): return sorted(data)"
            print(f"Query: {query_code}")
            
            # Generate query embedding
            query_embedding = await embeddings.generate_embedding(query_code)
            
            # Perform similarity search (simplified SQL query)
            from sqlalchemy import text, func
            
            # TiDB vector similarity query
            query = text("""
                SELECT 
                    id,
                    file_path,
                    content,
                    VEC_COSINE_DISTANCE(embedding, :query_embedding) as similarity
                FROM code_snippets 
                WHERE embedding IS NOT NULL
                ORDER BY similarity ASC 
                LIMIT 5
            """)
            
            result = await session.execute(
                query, 
                {"query_embedding": str(query_embedding)}
            )
            
            print("\nTop 5 similar code snippets:")
            print("-" * 60)
            
            for row in result.fetchall():
                snippet_id, file_path, content, similarity = row
                print(f"File: {file_path}")
                print(f"Similarity: {similarity:.4f}")
                print(f"Code: {content.strip()[:100]}...")
                print("-" * 60)
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        print("Note: This demo requires TiDB vector search capabilities")
        return False


async def demo_cache_management():
    """Demonstrate cache management operations."""
    print("\n=== Cache Management Demo ===")
    
    try:
        embeddings = GeminiEmbeddings()
        
        async with get_async_session() as session:
            batch_processor = EmbeddingBatchProcessor(
                provider=embeddings,
                db_session=session
            )
            
            # Get initial cache stats
            initial_stats = await batch_processor.get_cache_stats()
            print(f"Initial cache stats: {initial_stats}")
            
            # Clear expired cache entries
            cleared_count = await batch_processor.clear_expired_cache()
            print(f"Cleared {cleared_count} expired cache entries")
            
            # Get updated stats
            final_stats = await batch_processor.get_cache_stats()
            print(f"Final cache stats: {final_stats}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False


async def main():
    """Run all demos."""
    print("VibeCode AI Mentor - Embeddings + Database Integration Demo")
    print("=" * 60)
    
    # Check prerequisites
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY environment variable not set")
        return
    
    settings = get_settings()
    if not settings.tidb_host:
        print("ERROR: TiDB connection not configured")
        print("Please set TiDB environment variables in .env file")
        return
    
    # Run demos
    demos = [
        ("Basic Embedding Storage", demo_basic_embedding_storage),
        ("Batch Processing with Cache", demo_batch_processing_with_cache), 
        ("Vector Similarity Search", demo_vector_similarity_search),
        ("Cache Management", demo_cache_management),
    ]
    
    results = []
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*20} {demo_name} {'='*20}")
            success = await demo_func()
            results.append((demo_name, success))
        except Exception as e:
            print(f"Demo '{demo_name}' failed with error: {e}")
            results.append((demo_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    passed = 0
    for demo_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{demo_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} demos")
    
    if passed == len(results):
        print("🎉 All demos completed successfully!")
        print("Your Gemini embeddings integration is working with TiDB!")
    else:
        print("❌ Some demos failed. Check configuration and logs.")


if __name__ == "__main__":
    asyncio.run(main())