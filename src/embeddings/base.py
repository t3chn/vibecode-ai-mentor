"""Base embedding interface for VibeCode AI Mentor."""

import hashlib
from abc import ABC, abstractmethod
from typing import List, Optional

import tiktoken


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model_name: str, dimensions: int = 1536):
        self.model_name = model_name
        self.dimensions = dimensions
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(self._tokenizer.encode(text))
    
    def preprocess_code(self, code: str) -> str:
        """Preprocess code for better embeddings."""
        # Remove comments and excessive whitespace
        lines = []
        for line in code.split('\n'):
            stripped = line.strip()
            # Skip empty lines and comments
            if stripped and not stripped.startswith('#'):
                lines.append(stripped)
        
        return '\n'.join(lines)
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """Normalize embedding for cosine similarity."""
        import numpy as np
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return (np.array(embedding) / norm).tolist()
    
    def create_content_hash(self, text: str) -> str:
        """Create hash for caching."""
        return hashlib.sha256(f"{self.model_name}:{text}".encode()).hexdigest()