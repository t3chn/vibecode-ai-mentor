#!/usr/bin/env python3
"""
Demo script for Gemini embeddings integration.

This script demonstrates the core functionality of the GeminiEmbeddings class
including single embedding generation, batch processing, and error handling.

Usage:
    python examples/test_gemini_embeddings.py

Requirements:
    - GEMINI_API_KEY environment variable set
    - Valid Google Gemini API key with embeddings access
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.embeddings.gemini import GeminiEmbeddings
from src.embeddings.factory import get_embedding_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_single_embedding():
    """Test generating a single embedding."""
    print("\n=== Testing Single Embedding Generation ===")
    
    # Sample code for embedding
    sample_code = """
def calculate_fibonacci(n: int) -> int:
    '''Calculate the nth Fibonacci number using recursion.
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
    '''
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""
    
    try:
        embeddings = GeminiEmbeddings()
        
        print(f"Model: {embeddings.model_name}")
        print(f"Dimensions: {embeddings.dimensions}")
        print(f"Estimated tokens: {embeddings.estimate_tokens(sample_code)}")
        
        # Generate embedding
        print("Generating embedding...")
        embedding = await embeddings.generate_embedding(sample_code)
        
        print(f"Generated embedding with {len(embedding)} dimensions")
        print(f"First 10 values: {embedding[:10]}")
        print(f"Embedding norm: {sum(x*x for x in embedding)**0.5:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


async def test_batch_embeddings():
    """Test batch embedding generation."""
    print("\n=== Testing Batch Embedding Generation ===")
    
    # Multiple code samples
    code_samples = [
        """
def binary_search(arr: list, target: int) -> int:
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
        """
def quick_sort(arr: list) -> list:
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",
        """
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        return len(self.items) == 0
""",
    ]
    
    try:
        embeddings = GeminiEmbeddings(batch_size=2)  # Small batch for demo
        
        print(f"Processing {len(code_samples)} code samples...")
        
        # Generate batch embeddings
        batch_embeddings = await embeddings.generate_embeddings_batch(code_samples)
        
        print(f"Generated {len(batch_embeddings)} embeddings")
        
        # Calculate similarities between samples
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot_product / (norm_a * norm_b)
        
        print("\nSimilarity matrix:")
        for i, emb1 in enumerate(batch_embeddings):
            for j, emb2 in enumerate(batch_embeddings):
                if i <= j:
                    sim = cosine_similarity(emb1, emb2)
                    print(f"Sample {i} vs Sample {j}: {sim:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


async def test_preprocessing():
    """Test code preprocessing functionality."""
    print("\n=== Testing Code Preprocessing ===")
    
    # Code with comments and extra whitespace
    messy_code = """
# This is a file header comment
# Author: John Doe
# Date: 2025-01-01

def complex_function(data):
    # Initialize variables
    result = []
    
    # Process each item
    for item in data:
        # Check if item is valid
        if item is not None:
            # Apply transformation
            transformed = item * 2
            result.append(transformed)  # Add to result
    
    # Return the final result
    return result

# End of file
"""
    
    try:
        embeddings = GeminiEmbeddings()
        
        print("Original code:")
        print("-" * 40)
        print(messy_code)
        
        print("\nProcessed code:")
        print("-" * 40)
        processed = embeddings.preprocess_code(messy_code)
        print(processed)
        
        print(f"\nToken reduction: {embeddings.estimate_tokens(messy_code)} -> {embeddings.estimate_tokens(processed)}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


async def test_error_handling():
    """Test error handling and fallback mechanisms."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with invalid API key
        embeddings = GeminiEmbeddings(api_key="invalid-key")
        
        print("Testing with invalid API key...")
        health = embeddings.health_check()
        print(f"Health check result: {health}")
        
        # Test with embedding manager fallback
        print("\nTesting embedding manager with fallback...")
        manager = get_embedding_manager(primary="gemini", fallback="openai")
        
        health_status = manager.health_check()
        print(f"Manager health status: {health_status}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


async def run_performance_test():
    """Run a simple performance test."""
    print("\n=== Performance Test ===")
    
    try:
        embeddings = GeminiEmbeddings(batch_size=5)
        
        # Generate test code samples
        test_samples = []
        for i in range(10):
            code = f"""
def function_{i}(x):
    '''Function number {i}.'''
    result = x * {i}
    for j in range({i + 1}):
        result += j
    return result
"""
            test_samples.append(code)
        
        print(f"Performance test with {len(test_samples)} samples...")
        
        import time
        start_time = time.time()
        
        batch_embeddings = await embeddings.generate_embeddings_batch(test_samples)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Generated {len(batch_embeddings)} embeddings in {duration:.2f}s")
        print(f"Average time per embedding: {duration/len(batch_embeddings):.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


async def main():
    """Run all tests."""
    print("VibeCode AI Mentor - Gemini Embeddings Demo")
    print("=" * 50)
    
    # Check for API key
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: GEMINI_API_KEY environment variable not set")
        print("Please set your Gemini API key:")
        print("export GEMINI_API_KEY='your-api-key-here'")
        return
    
    # Run tests
    tests = [
        ("Single Embedding", test_single_embedding),
        ("Batch Embeddings", test_batch_embeddings),
        ("Code Preprocessing", test_preprocessing),
        ("Error Handling", test_error_handling),
        ("Performance Test", run_performance_test),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = await test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<30} {status}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Gemini embeddings integration is working correctly.")
    else:
        print("❌ Some tests failed. Check the logs above for details.")


if __name__ == "__main__":
    asyncio.run(main())