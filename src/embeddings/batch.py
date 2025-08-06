"""Batch processing utilities for embeddings generation."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.db.models import EmbeddingCache
from src.embeddings.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingBatchProcessor:
    """Efficient batch processing for embeddings with caching."""
    
    def __init__(
        self,
        provider: EmbeddingProvider,
        db_session: AsyncSession,
        batch_size: Optional[int] = None,
    ):
        """Initialize batch processor.
        
        Args:
            provider: Embedding provider instance
            db_session: Database session for caching
            batch_size: Override default batch size
        """
        self.provider = provider
        self.db_session = db_session
        self.batch_size = batch_size or get_settings().batch_size
        self.cache_ttl = get_settings().cache_ttl
        self.enable_cache = get_settings().enable_cache
        
        logger.info(f"Initialized batch processor with batch size {self.batch_size}")
    
    async def process_texts(
        self, texts: List[str], use_cache: bool = True
    ) -> List[List[float]]:
        """Process multiple texts with intelligent caching.
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use caching
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        logger.info(f"Processing {len(texts)} texts for embeddings")
        
        # Check cache for existing embeddings
        cache_results = {}
        uncached_texts = []
        uncached_indices = []
        
        if use_cache and self.enable_cache:
            cache_results, uncached_texts, uncached_indices = await self._check_cache(texts)
            logger.info(
                f"Cache hit: {len(cache_results)}/{len(texts)} "
                f"({len(cache_results)/len(texts)*100:.1f}%)"
            )
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            new_embeddings = await self._generate_embeddings_with_retry(uncached_texts)
            
            # Cache new embeddings
            if use_cache and self.enable_cache and new_embeddings:
                await self._cache_new_embeddings(uncached_texts, new_embeddings)
        
        # Combine cached and new results
        final_embeddings = [None] * len(texts)
        
        # Fill cached results
        for i, text in enumerate(texts):
            text_hash = self.provider.create_content_hash(text)
            if text_hash in cache_results:
                final_embeddings[i] = cache_results[text_hash]
        
        # Fill new results
        for i, embedding_idx in enumerate(uncached_indices):
            if i < len(new_embeddings):
                final_embeddings[embedding_idx] = new_embeddings[i]
        
        # Fill any remaining None values with zero vectors
        for i in range(len(final_embeddings)):
            if final_embeddings[i] is None:
                logger.warning(f"Missing embedding for text {i}, using zero vector")
                final_embeddings[i] = [0.0] * self.provider.dimensions
        
        logger.info(f"Successfully processed all {len(final_embeddings)} embeddings")
        return final_embeddings
    
    async def _check_cache(
        self, texts: List[str]
    ) -> Tuple[Dict[str, List[float]], List[str], List[int]]:
        """Check cache for existing embeddings.
        
        Returns:
            Tuple of (cached_results, uncached_texts, uncached_indices)
        """
        text_hashes = [self.provider.create_content_hash(text) for text in texts]
        
        # Query cache
        query = select(EmbeddingCache).where(
            and_(
                EmbeddingCache.content_hash.in_(text_hashes),
                EmbeddingCache.model_name == self.provider.model_name,
                EmbeddingCache.expires_at > datetime.utcnow(),
            )
        )
        
        result = await self.db_session.execute(query)
        cached_entries = result.scalars().all()
        
        # Build cache results dictionary
        cache_results = {}
        for entry in cached_entries:
            cache_results[entry.content_hash] = entry.embedding
        
        # Identify uncached texts
        uncached_texts = []
        uncached_indices = []
        for i, (text, text_hash) in enumerate(zip(texts, text_hashes)):
            if text_hash not in cache_results:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        return cache_results, uncached_texts, uncached_indices
    
    async def _generate_embeddings_with_retry(
        self, texts: List[str]
    ) -> List[List[float]]:
        """Generate embeddings with error handling and batching."""
        all_embeddings = []
        
        # Process in smaller batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                batch_embeddings = await self.provider.generate_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
                
                logger.debug(f"Processed batch {i//self.batch_size + 1}, "
                           f"total embeddings: {len(all_embeddings)}")
                
                # Add small delay between batches to be respectful to API
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to process batch starting at {i}: {e}")
                # Add zero vectors for failed batch
                batch_embeddings = [[0.0] * self.provider.dimensions for _ in batch]
                all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _cache_new_embeddings(
        self, texts: List[str], embeddings: List[List[float]]
    ) -> None:
        """Cache new embeddings in database."""
        if len(texts) != len(embeddings):
            logger.error("Mismatch between texts and embeddings count")
            return
        
        try:
            cache_entries = []
            expires_at = datetime.utcnow() + timedelta(seconds=self.cache_ttl)
            
            for text, embedding in zip(texts, embeddings):
                text_hash = self.provider.create_content_hash(text)
                
                cache_entry = EmbeddingCache(
                    content_hash=text_hash,
                    embedding=embedding,
                    model_name=self.provider.model_name,
                    expires_at=expires_at,
                )
                cache_entries.append(cache_entry)
            
            # Bulk insert
            self.db_session.add_all(cache_entries)
            await self.db_session.commit()
            
            logger.info(f"Cached {len(cache_entries)} new embeddings")
            
        except Exception as e:
            logger.error(f"Failed to cache embeddings: {e}")
            await self.db_session.rollback()
    
    async def clear_expired_cache(self) -> int:
        """Clear expired cache entries.
        
        Returns:
            Number of entries cleared
        """
        try:
            from sqlalchemy import delete
            
            # Delete expired entries
            delete_query = delete(EmbeddingCache).where(
                EmbeddingCache.expires_at < datetime.utcnow()
            )
            
            result = await self.db_session.execute(delete_query)
            await self.db_session.commit()
            
            cleared_count = result.rowcount or 0
            logger.info(f"Cleared {cleared_count} expired cache entries")
            
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            await self.db_session.rollback()
            return 0
    
    async def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            from sqlalchemy import func
            
            # Count total entries
            total_query = select(func.count(EmbeddingCache.id))
            total_result = await self.db_session.execute(total_query)
            total_count = total_result.scalar() or 0
            
            # Count expired entries
            expired_query = select(func.count(EmbeddingCache.id)).where(
                EmbeddingCache.expires_at < datetime.utcnow()
            )
            expired_result = await self.db_session.execute(expired_query)
            expired_count = expired_result.scalar() or 0
            
            # Count by model
            model_query = select(
                EmbeddingCache.model_name, func.count(EmbeddingCache.id)
            ).group_by(EmbeddingCache.model_name)
            model_result = await self.db_session.execute(model_query)
            model_counts = dict(model_result.fetchall())
            
            return {
                "total_entries": total_count,
                "expired_entries": expired_count,
                "active_entries": total_count - expired_count,
                "model_counts": model_counts,
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}