"""OpenAI embeddings fallback provider for VibeCode AI Mentor."""

import asyncio
import logging
from typing import List, Optional

import openai
from tenacity import (
    retry as tenacity_retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.core.config import get_settings
from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embeddings provider as fallback option."""
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 100):
        """Initialize OpenAI embeddings client.
        
        Args:
            api_key: OpenAI API key (defaults to config)
            batch_size: Number of texts to process in one batch
        """
        settings = get_settings()
        super().__init__(model_name="text-embedding-3-small", dimensions=1536)
        
        self.api_key = api_key or settings.openai_api_key
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
            
        self.batch_size = batch_size
        self.client = openai.AsyncOpenAI(api_key=self.api_key)
        
        # Rate limiting
        self._request_times: List[float] = []
        self._max_requests_per_minute = 100  # OpenAI's default limit
        
        logger.info(f"Initialized OpenAI embeddings with model {self.model_name}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            1536-dimensional embedding vector
        """
        embeddings = await self.generate_embeddings_batch([text])
        return embeddings[0]
    
    @tenacity_retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIError)),
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
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Rate limiting
            await self._enforce_rate_limit()
        
        return all_embeddings
    
    async def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """Process a single batch of texts."""
        # Preprocess texts
        processed_texts = [self.preprocess_code(text) for text in texts]
        
        try:
            logger.debug(f"Generating embeddings for batch of {len(texts)} texts")
            
            # Validate token counts
            valid_texts = []
            for text in processed_texts:
                token_count = self.estimate_tokens(text)
                if token_count > 8000:  # OpenAI model limit
                    logger.warning(f"Text too long ({token_count} tokens), truncating")
                    text = self._truncate_text(text, 8000)
                valid_texts.append(text)
            
            # Make API call
            response = await self.client.embeddings.create(
                model=self.model_name,
                input=valid_texts,
                encoding_format="float"
            )
            
            # Extract embeddings
            embeddings = []
            for data in response.data:
                embedding = self.normalize_embedding(data.embedding)
                embeddings.append(embedding)
            
            logger.info(f"Successfully generated {len(embeddings)} embeddings")
            return embeddings
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating embeddings: {e}")
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
    
    def health_check(self) -> bool:
        """Check if the OpenAI API is accessible."""
        try:
            # Simple synchronous test call
            sync_client = openai.OpenAI(api_key=self.api_key)
            response = sync_client.embeddings.create(
                model=self.model_name,
                input="test"
            )
            return len(response.data) > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Convenience function for testing
async def test_openai_embeddings() -> None:
    """Test function for OpenAI embeddings."""
    try:
        embeddings = OpenAIEmbeddings()
        
        test_code = """
def fibonacci(n: int) -> int:
    '''Calculate fibonacci number recursively.'''
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
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
    asyncio.run(test_openai_embeddings())