"""Google Gemini embeddings integration for VibeCode AI Mentor."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import google.generativeai as genai
from google.api_core import retry
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import (
    retry as tenacity_retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.config import get_settings
from src.db.models import EmbeddingCache
from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class GeminiEmbeddings(EmbeddingProvider):
    """Google Gemini embeddings provider with caching and error handling."""
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 100):
        """Initialize Gemini embeddings client.
        
        Args:
            api_key: Gemini API key (defaults to config)
            batch_size: Number of texts to process in one batch
        """
        settings = get_settings()
        super().__init__(model_name="text-embedding-004", dimensions=1536)
        
        self.api_key = api_key or settings.gemini_api_key
        self.batch_size = batch_size
        self.cache_ttl = settings.cache_ttl
        self.enable_cache = settings.enable_cache
        
        # Configure Gemini client
        genai.configure(api_key=self.api_key)
        self.client = genai
        
        # Rate limiting
        self._request_times: List[float] = []
        self._max_requests_per_minute = 60  # Conservative limit
        
        logger.info(f"Initialized Gemini embeddings with model {self.model_name}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            1536-dimensional embedding vector
            
        Raises:
            Exception: If API call fails after retries
        """
        embeddings = await self.generate_embeddings_batch([text])
        return embeddings[0]
    
    @tenacity_retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
    )
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and retries.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache first if enabled
        if self.enable_cache:
            cached_results = await self._get_cached_embeddings(texts)
            if cached_results:
                logger.info(f"Found {len(cached_results)} cached embeddings")
                return cached_results
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting
            await self._enforce_rate_limit()
        
        # Cache results if enabled
        if self.enable_cache and all_embeddings:
            await self._cache_embeddings(texts, all_embeddings)
        
        return all_embeddings
    
    async def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts."""
        # Preprocess texts
        processed_texts = [self.preprocess_code(text) for text in texts]
        
        try:
            # Make API call
            logger.debug(f"Generating embeddings for batch of {len(texts)} texts")
            
            # Use embed_content for batch processing
            embeddings = []
            for text in processed_texts:
                # Validate token count
                token_count = self.estimate_tokens(text)
                if token_count > 8192:  # Gemini model limit
                    logger.warning(f"Text too long ({token_count} tokens), truncating")
                    text = self._truncate_text(text, 8000)  # Leave buffer
                
                result = await asyncio.to_thread(
                    self.client.embed_content,
                    model=f"models/{self.model_name}",
                    content=text,
                    task_type="code",  # Optimize for code content
                )
                
                if hasattr(result, 'embedding') and result.embedding:
                    embedding = self.normalize_embedding(result.embedding)
                    embeddings.append(embedding)
                else:
                    logger.error(f"No embedding returned for text: {text[:100]}...")
                    # Return zero vector as fallback
                    embeddings.append([0.0] * self.dimensions)
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero vectors for failed requests
            return [[0.0] * self.dimensions for _ in texts]
    
    def _truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        truncated_tokens = tokens[:max_tokens]
        return self._tokenizer.decode(truncated_tokens)
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting to avoid hitting API limits."""
        import time
        
        now = time.time()
        # Remove requests older than 1 minute
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self._max_requests_per_minute:
            sleep_time = 60 - (now - self._request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
                await asyncio.sleep(sleep_time)
        
        self._request_times.append(now)
    
    async def _get_cached_embeddings(
        self, texts: List[str]
    ) -> Optional[List[List[float]]]:
        """Retrieve cached embeddings if available."""
        # This would need database session - for now return None
        # TODO: Implement with proper database session injection
        return None
    
    async def _cache_embeddings(
        self, texts: List[str], embeddings: List[List[float]]
    ) -> None:
        """Cache embeddings for future use."""
        # This would need database session - for now skip
        # TODO: Implement with proper database session injection
        pass
    
    def health_check(self) -> bool:
        """Check if the Gemini API is accessible."""
        try:
            # Simple test call
            result = self.client.embed_content(
                model=f"models/{self.model_name}",
                content="test",
            )
            return hasattr(result, 'embedding') and result.embedding is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Convenience function for testing
async def test_gemini_embeddings() -> None:
    """Test function for Gemini embeddings."""
    embeddings = GeminiEmbeddings()
    
    test_code = """
def calculate_complexity(code: str) -> float:
    '''Calculate cyclomatic complexity of code.'''
    # Simple complexity calculation
    complexity = 1
    for line in code.split('\\n'):
        if any(keyword in line for keyword in ['if', 'elif', 'for', 'while']):
            complexity += 1
    return complexity
"""
    
    try:
        # Test single embedding
        embedding = await embeddings.generate_embedding(test_code)
        print(f"Generated embedding with {len(embedding)} dimensions")
        
        # Test batch embeddings
        batch_embeddings = await embeddings.generate_embeddings_batch([test_code, test_code])
        print(f"Generated {len(batch_embeddings)} batch embeddings")
        
        # Test health check
        is_healthy = embeddings.health_check()
        print(f"API health check: {'PASS' if is_healthy else 'FAIL'}")
        
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_gemini_embeddings())