"""Embedding provider factory and manager."""

import logging
from typing import Optional

from src.core.config import get_settings
from src.embeddings.base import EmbeddingProvider
from src.embeddings.gemini import GeminiEmbeddings
from src.embeddings.openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingProviderFactory:
    """Factory for creating embedding providers with fallback support."""
    
    @staticmethod
    def create_provider(
        provider_name: str = "gemini",
        api_key: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> EmbeddingProvider:
        """Create embedding provider instance.
        
        Args:
            provider_name: Provider name ("gemini" or "openai")
            api_key: Override API key
            batch_size: Override batch size
            
        Returns:
            Embedding provider instance
            
        Raises:
            ValueError: If provider name is invalid
        """
        settings = get_settings()
        batch_size = batch_size or settings.batch_size
        
        if provider_name.lower() == "gemini":
            return GeminiEmbeddings(api_key=api_key, batch_size=batch_size)
        elif provider_name.lower() == "openai":
            return OpenAIEmbeddings(api_key=api_key, batch_size=batch_size)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    @staticmethod
    def create_with_fallback(
        primary_provider: str = "gemini",
        fallback_provider: str = "openai",
        batch_size: Optional[int] = None,
    ) -> "EmbeddingManager":
        """Create embedding manager with fallback support.
        
        Args:
            primary_provider: Primary provider name
            fallback_provider: Fallback provider name
            batch_size: Override batch size
            
        Returns:
            EmbeddingManager with fallback configured
        """
        primary = EmbeddingProviderFactory.create_provider(
            primary_provider, batch_size=batch_size
        )
        
        fallback = None
        try:
            fallback = EmbeddingProviderFactory.create_provider(
                fallback_provider, batch_size=batch_size
            )
        except Exception as e:
            logger.warning(f"Failed to create fallback provider {fallback_provider}: {e}")
        
        return EmbeddingManager(primary_provider=primary, fallback_provider=fallback)


class EmbeddingManager:
    """Manages embedding providers with automatic fallback."""
    
    def __init__(
        self,
        primary_provider: EmbeddingProvider,
        fallback_provider: Optional[EmbeddingProvider] = None,
    ):
        """Initialize embedding manager.
        
        Args:
            primary_provider: Primary embedding provider
            fallback_provider: Optional fallback provider
        """
        self.primary_provider = primary_provider
        self.fallback_provider = fallback_provider
        
        logger.info(
            f"Initialized embedding manager with primary: {primary_provider.model_name}"
            + (f", fallback: {fallback_provider.model_name}" if fallback_provider else "")
        )
    
    async def generate_embedding(self, text: str, use_fallback: bool = True) -> list[float]:
        """Generate embedding with automatic fallback.
        
        Args:
            text: Text to embed
            use_fallback: Whether to use fallback on failure
            
        Returns:
            Embedding vector
        """
        # Try primary provider
        try:
            return await self.primary_provider.generate_embedding(text)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
            
            # Try fallback if available and enabled
            if use_fallback and self.fallback_provider:
                try:
                    logger.info("Using fallback provider")
                    return await self.fallback_provider.generate_embedding(text)
                except Exception as fallback_e:
                    logger.error(f"Fallback provider also failed: {fallback_e}")
            
            # Return zero vector as last resort
            logger.error("All providers failed, returning zero vector")
            return [0.0] * self.primary_provider.dimensions
    
    async def generate_embeddings_batch(
        self, texts: list[str], use_fallback: bool = True
    ) -> list[list[float]]:
        """Generate embeddings batch with automatic fallback.
        
        Args:
            texts: List of texts to embed
            use_fallback: Whether to use fallback on failure
            
        Returns:
            List of embedding vectors
        """
        # Try primary provider
        try:
            return await self.primary_provider.generate_embeddings_batch(texts)
        except Exception as e:
            logger.warning(f"Primary provider failed: {e}")
            
            # Try fallback if available and enabled
            if use_fallback and self.fallback_provider:
                try:
                    logger.info("Using fallback provider for batch")
                    return await self.fallback_provider.generate_embeddings_batch(texts)
                except Exception as fallback_e:
                    logger.error(f"Fallback provider also failed: {fallback_e}")
            
            # Return zero vectors as last resort
            logger.error("All providers failed, returning zero vectors")
            return [[0.0] * self.primary_provider.dimensions for _ in texts]
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using primary provider."""
        return self.primary_provider.estimate_tokens(text)
    
    def preprocess_code(self, code: str) -> str:
        """Preprocess code using primary provider."""
        return self.primary_provider.preprocess_code(code)
    
    def health_check(self) -> dict[str, bool]:
        """Check health of all providers.
        
        Returns:
            Dictionary with health status of each provider
        """
        health = {"primary": False, "fallback": None}
        
        try:
            health["primary"] = self.primary_provider.health_check()
        except Exception as e:
            logger.error(f"Primary provider health check failed: {e}")
        
        if self.fallback_provider:
            try:
                health["fallback"] = self.fallback_provider.health_check()
            except Exception as e:
                logger.error(f"Fallback provider health check failed: {e}")
        
        return health
    
    @property
    def dimensions(self) -> int:
        """Get embedding dimensions."""
        return self.primary_provider.dimensions
    
    @property
    def model_name(self) -> str:
        """Get primary model name."""
        return self.primary_provider.model_name


# Convenience function to get default embedding manager
def get_embedding_manager(
    primary: str = "gemini", fallback: str = "openai"
) -> EmbeddingManager:
    """Get default embedding manager with fallback.
    
    Args:
        primary: Primary provider name
        fallback: Fallback provider name
        
    Returns:
        Configured EmbeddingManager
    """
    return EmbeddingProviderFactory.create_with_fallback(
        primary_provider=primary, fallback_provider=fallback
    )