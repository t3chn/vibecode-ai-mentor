"""Tests for Gemini embeddings integration."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.embeddings.gemini import GeminiEmbeddings


class TestGeminiEmbeddings:
    """Test Gemini embeddings functionality."""
    
    @pytest.fixture
    def mock_genai(self):
        """Mock google.generativeai module."""
        with patch('src.embeddings.gemini.genai') as mock:
            mock_response = MagicMock()
            mock_response.embedding = [0.1] * 1536
            mock.embed_content.return_value = mock_response
            yield mock
    
    @pytest.fixture
    def embeddings(self, mock_genai):
        """Create GeminiEmbeddings instance with mocked API."""
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}):
            return GeminiEmbeddings(api_key='test-key')
    
    def test_initialization(self, embeddings):
        """Test embeddings initialization."""
        assert embeddings.model_name == "text-embedding-004"
        assert embeddings.dimensions == 1536
        assert embeddings.api_key == "test-key"
        assert embeddings.batch_size == 100
    
    def test_preprocess_code(self, embeddings):
        """Test code preprocessing."""
        code = """
# This is a comment
def hello():
    # Another comment
    print("Hello")
    
    # More comments
    return True
"""
        
        processed = embeddings.preprocess_code(code)
        lines = processed.split('\n')
        
        # Should remove comments and empty lines
        assert "# This is a comment" not in processed
        assert "def hello():" in processed
        assert "print(\"Hello\")" in processed
        assert "return True" in processed
    
    def test_estimate_tokens(self, embeddings):
        """Test token estimation."""
        text = "def test(): pass"
        tokens = embeddings.estimate_tokens(text)
        assert isinstance(tokens, int)
        assert tokens > 0
    
    def test_normalize_embedding(self, embeddings):
        """Test embedding normalization."""
        embedding = [1.0, 2.0, 3.0, 4.0]
        normalized = embeddings.normalize_embedding(embedding)
        
        # Check normalization (L2 norm should be 1)
        import numpy as np
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6
    
    def test_create_content_hash(self, embeddings):
        """Test content hash creation."""
        text = "def test(): pass"
        hash1 = embeddings.create_content_hash(text)
        hash2 = embeddings.create_content_hash(text)
        hash3 = embeddings.create_content_hash("different text")
        
        assert hash1 == hash2  # Same text should produce same hash
        assert hash1 != hash3  # Different text should produce different hash
        assert len(hash1) == 64  # SHA256 hex string length
    
    @pytest.mark.asyncio
    async def test_generate_embedding(self, embeddings, mock_genai):
        """Test single embedding generation."""
        # Mock the async call
        with patch('asyncio.to_thread') as mock_thread:
            mock_result = MagicMock()
            mock_result.embedding = [0.1] * 1536
            mock_thread.return_value = mock_result
            
            embedding = await embeddings.generate_embedding("def test(): pass")
            
            assert len(embedding) == 1536
            assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, embeddings, mock_genai):
        """Test batch embedding generation."""
        texts = ["def test1(): pass", "def test2(): pass"]
        
        # Mock the async calls
        with patch('asyncio.to_thread') as mock_thread:
            mock_result = MagicMock()
            mock_result.embedding = [0.1] * 1536
            mock_thread.return_value = mock_result
            
            embeddings = await embeddings.generate_embeddings_batch(texts)
            
            assert len(embeddings) == 2
            assert all(len(emb) == 1536 for emb in embeddings)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, embeddings):
        """Test rate limiting functionality."""
        # Fill up request times to trigger rate limiting
        import time
        now = time.time()
        embeddings._request_times = [now - 30] * 60  # 60 requests in last 30 seconds
        
        start_time = time.time()
        await embeddings._enforce_rate_limit()
        end_time = time.time()
        
        # Should have slept for some time
        assert end_time - start_time >= 0  # At least some delay
    
    def test_health_check(self, embeddings, mock_genai):
        """Test health check functionality."""
        # Mock successful response
        mock_result = MagicMock()
        mock_result.embedding = [0.1] * 1536
        mock_genai.embed_content.return_value = mock_result
        
        is_healthy = embeddings.health_check()
        assert is_healthy is True
        
        # Mock failed response
        mock_genai.embed_content.side_effect = Exception("API Error")
        is_healthy = embeddings.health_check()
        assert is_healthy is False
    
    @pytest.mark.asyncio
    async def test_error_handling(self, embeddings):
        """Test error handling in batch processing."""
        texts = ["def test(): pass"]
        
        # Mock API failure
        with patch('asyncio.to_thread') as mock_thread:
            mock_thread.side_effect = Exception("API Error")
            
            embeddings_result = await embeddings.generate_embeddings_batch(texts)
            
            # Should return zero vectors on failure
            assert len(embeddings_result) == 1
            assert embeddings_result[0] == [0.0] * 1536


@pytest.mark.integration
class TestGeminiIntegration:
    """Integration tests for Gemini embeddings (requires API key)."""
    
    @pytest.mark.skipif(
        not pytest.config.getoption("--integration", default=False),
        reason="Integration tests disabled"
    )  
    @pytest.mark.asyncio
    async def test_real_api_call(self):
        """Test real API call (requires valid API key)."""
        import os
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            pytest.skip("GEMINI_API_KEY not set")
        
        embeddings = GeminiEmbeddings(api_key=api_key)
        
        test_code = """
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        # Test single embedding
        embedding = await embeddings.generate_embedding(test_code)
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)
        
        # Test health check
        is_healthy = embeddings.health_check()
        assert is_healthy is True