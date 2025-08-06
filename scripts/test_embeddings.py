#!/usr/bin/env python3
"""
Quick test script for embeddings functionality.

Usage:
    python scripts/test_embeddings.py [--provider gemini|openai] [--text "your text"]
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.embeddings import get_embedding_manager, GeminiEmbeddings, OpenAIEmbeddings


async def test_embeddings(provider_name: str = "gemini", test_text: str = None):
    """Test embeddings with specified provider."""
    
    if test_text is None:
        test_text = """
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return "Hello, World!"
"""
    
    print(f"Testing {provider_name} embeddings...")
    print(f"Text: {test_text.strip()}")
    
    try:
        if provider_name == "gemini":
            provider = GeminiEmbeddings()
        elif provider_name == "openai":
            provider = OpenAIEmbeddings()
        else:
            # Use manager with fallback
            manager = get_embedding_manager()
            embedding = await manager.generate_embedding(test_text)
            print(f"✅ Generated embedding: {len(embedding)} dimensions")
            print(f"First 5 values: {embedding[:5]}")
            return
        
        # Test single embedding
        embedding = await provider.generate_embedding(test_text)
        print(f"✅ Generated embedding: {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        
        # Test health check
        health = provider.health_check()
        print(f"Health check: {'✅ PASS' if health else '❌ FAIL'}")
        
        # Test preprocessing
        processed = provider.preprocess_code(test_text)
        token_count = provider.estimate_tokens(processed)
        print(f"Token count: {token_count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test embeddings functionality")
    parser.add_argument(
        "--provider", 
        choices=["gemini", "openai", "manager"], 
        default="gemini",
        help="Embedding provider to test"
    )
    parser.add_argument(
        "--text",
        help="Custom text to embed (optional)"
    )
    
    args = parser.parse_args()
    
    # Check for API keys
    if args.provider == "gemini" and not os.getenv('GEMINI_API_KEY'):
        print("❌ GEMINI_API_KEY environment variable not set")
        return
    
    if args.provider == "openai" and not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY environment variable not set") 
        return
    
    # Run test
    asyncio.run(test_embeddings(args.provider, args.text))


if __name__ == "__main__":
    main()