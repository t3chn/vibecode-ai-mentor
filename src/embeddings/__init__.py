"""Embedding generation and management for code snippets.

This module handles generating vector embeddings from code using LLM APIs
(Gemini and OpenAI), chunking strategies, and batch processing for efficient
embedding generation.
"""

from src.embeddings.base import EmbeddingProvider
from src.embeddings.batch import EmbeddingBatchProcessor
from src.embeddings.factory import EmbeddingManager, get_embedding_manager
from src.embeddings.gemini import GeminiEmbeddings
from src.embeddings.openai import OpenAIEmbeddings

__all__ = [
    "EmbeddingProvider",
    "GeminiEmbeddings", 
    "OpenAIEmbeddings",
    "EmbeddingManager",
    "EmbeddingBatchProcessor",
    "get_embedding_manager",
]