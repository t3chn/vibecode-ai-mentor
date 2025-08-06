"""Tests for LLM client functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json

from src.generator.llm_client import (
    LLMClient, LLMProvider, RecommendationLevel,
    Recommendation, RefactorSuggestion, AntiPatternFix,
    generate_code_recommendations
)
from src.services.analysis import SnippetAnalysis, FileAnalysis
from src.api.models import RecommendationType


@pytest.fixture
def mock_settings():
    """Mock settings with API keys."""
    with patch('src.generator.llm_client.get_settings') as mock_get_settings:
        settings = MagicMock()
        settings.gemini_api_key = "test_gemini_key"
        settings.openai_api_key = "test_openai_key"
        mock_get_settings.return_value = settings
        yield settings


@pytest.fixture
def sample_snippet_analysis():
    """Sample snippet analysis for testing."""
    analysis = SnippetAnalysis(
        snippet_id="test-123",
        language="python"
    )
    analysis.metrics = {
        "lines_of_code": 25,
        "cyclomatic_complexity": 8,
        "maintainability_index": 65.5,
        "complexity_score": 7.2,
        "risk_level": "medium"
    }
    analysis.ast_elements = {
        "functions": [{"name": "test_function", "line_start": 5}],
        "classes": [],
        "imports": [{"name": "os", "line": 1}]
    }
    analysis.issues = [
        {
            "type": "complexity",
            "severity": "warning",
            "message": "Function is too complex",
            "line": 5
        }
    ]
    return analysis


@pytest.fixture
def sample_file_analysis():
    """Sample file analysis for testing."""
    analysis = FileAnalysis(
        file_path="/test/sample.py",
        language="python",
        status="success"
    )
    analysis.metrics = {
        "lines_of_code": 150,
        "cyclomatic_complexity": 15,
        "average_complexity": 3.2,
        "maintainability_index": 78.0,
        "functions_count": 8,
        "anti_patterns": [
            {
                "type": "long_method",
                "severity": "warning",
                "message": "Method is too long",
                "line": 25
            }
        ]
    }
    analysis.functions = [
        {"name": "main", "line_start": 10},
        {"name": "helper", "line_start": 50}
    ]
    analysis.classes = [{"name": "TestClass", "line_start": 80}]
    return analysis


class TestLLMClient:
    """Test LLM client functionality."""
    
    def test_initialization(self, mock_settings):
        """Test LLM client initialization."""
        with patch('src.generator.llm_client.genai') as mock_genai, \
             patch('src.generator.llm_client.AsyncOpenAI') as mock_openai:
            
            # Mock successful initialization
            mock_genai.GenerativeModel.return_value = MagicMock()
            mock_openai.return_value = MagicMock()
            
            client = LLMClient()
            
            assert client.primary_provider == LLMProvider.GEMINI
            assert client.fallback_provider == LLMProvider.OPENAI
            assert client.max_tokens == 4000
            assert client.temperature == 0.3
    
    def test_initialization_no_api_keys(self):
        """Test initialization failure with no API keys."""
        with patch('src.generator.llm_client.get_settings') as mock_get_settings:
            settings = MagicMock()
            settings.gemini_api_key = None
            settings.openai_api_key = None
            mock_get_settings.return_value = settings
            
            with pytest.raises(ValueError, match="No LLM providers available"):
                LLMClient()
    
    @pytest.mark.asyncio
    async def test_generate_with_gemini(self, mock_settings):
        """Test generation with Gemini provider."""
        with patch('src.generator.llm_client.genai') as mock_genai, \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            # Mock Gemini response
            mock_response = MagicMock()
            mock_response.text = "Test response from Gemini"
            mock_model = MagicMock()
            mock_model.generate_content = MagicMock(return_value=mock_response)
            mock_genai.GenerativeModel.return_value = mock_model
            
            client = LLMClient()
            
            with patch('asyncio.to_thread', return_value=mock_response):
                response = await client._generate_with_gemini("test prompt")
                
            assert response == "Test response from Gemini"
    
    @pytest.mark.asyncio
    async def test_generate_with_openai(self, mock_settings):
        """Test generation with OpenAI provider."""
        with patch('src.generator.llm_client.genai'), \
             patch('src.generator.llm_client.AsyncOpenAI') as mock_openai_class:
            
            # Mock OpenAI response
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test response from OpenAI"
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            client = LLMClient()
            response = await client._generate_with_openai("test prompt")
            
            assert response == "Test response from OpenAI"
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback(self, mock_settings):
        """Test fallback to secondary provider when primary fails."""
        with patch('src.generator.llm_client.genai') as mock_genai, \
             patch('src.generator.llm_client.AsyncOpenAI') as mock_openai_class:
            
            # Mock Gemini to fail
            mock_genai.GenerativeModel.return_value = None
            
            # Mock OpenAI to succeed
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Fallback response"
            
            mock_client = AsyncMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            client = LLMClient()
            client.gemini_model = None  # Simulate Gemini failure
            
            response = await client._generate_with_fallback("test prompt")
            
            assert response == "Fallback response"
    
    @pytest.mark.asyncio
    async def test_generate_recommendations_snippet(self, mock_settings, sample_snippet_analysis):
        """Test recommendation generation for code snippet."""
        with patch('src.generator.llm_client.genai') as mock_genai, \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            # Mock LLM response with valid JSON
            mock_response_text = '''[
                {
                    "type": "refactoring",
                    "severity": "warning",
                    "line_start": 5,
                    "line_end": 15,
                    "message": "Function is too complex, consider breaking it down",
                    "suggestion": "Extract helper methods",
                    "explanation": "Complex functions are harder to maintain and test",
                    "confidence": 0.85,
                    "code_before": "def complex_function():\\n    # complex logic",
                    "code_after": "def simple_function():\\n    helper1()\\n    helper2()"
                }
            ]'''
            
            mock_response = MagicMock()
            mock_response.text = mock_response_text
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            
            client = LLMClient()
            
            with patch.object(client, '_generate_with_fallback', return_value=mock_response_text):
                recommendations = await client.generate_recommendations(
                    sample_snippet_analysis,
                    level=RecommendationLevel.DETAILED
                )
            
            assert len(recommendations) == 1
            rec = recommendations[0]
            assert rec.type == RecommendationType.REFACTORING
            assert rec.severity == "warning"
            assert rec.line_start == 5
            assert rec.confidence == 0.85
            assert "complex" in rec.message.lower()
    
    @pytest.mark.asyncio
    async def test_generate_refactoring_suggestions(self, mock_settings):
        """Test refactoring suggestion generation."""
        with patch('src.generator.llm_client.genai'), \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            mock_response_text = '''[
                {
                    "refactor_type": "extract_method",
                    "description": "Extract complex logic into separate method",
                    "code_before": "def big_function():\\n    # lots of code",
                    "code_after": "def big_function():\\n    extracted_method()\\n\\ndef extracted_method():\\n    # extracted logic",
                    "confidence": 0.9,
                    "benefits": ["Improved readability", "Better testability"],
                    "risks": ["May introduce coupling"],
                    "effort_level": "medium"
                }
            ]'''
            
            client = LLMClient()
            
            with patch.object(client, '_generate_with_fallback', return_value=mock_response_text):
                suggestions = await client.generate_refactoring_suggestions(
                    "def complex_function(): pass",
                    {"cyclomatic_complexity": 10}
                )
            
            assert len(suggestions) == 1
            sug = suggestions[0]
            assert sug.refactor_type == "extract_method"
            assert sug.confidence == 0.9
            assert "readability" in sug.benefits[0].lower()
    
    @pytest.mark.asyncio
    async def test_generate_code_explanation(self, mock_settings):
        """Test code explanation generation."""
        with patch('src.generator.llm_client.genai'), \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            mock_explanation = """This function implements a binary search algorithm.
            It takes a sorted list and a target value, then efficiently finds
            the target by repeatedly dividing the search space in half."""
            
            client = LLMClient()
            
            with patch.object(client, '_generate_with_fallback', return_value=mock_explanation):
                explanation = await client.generate_code_explanation(
                    "def binary_search(arr, target): pass"
                )
            
            assert "binary search" in explanation.lower()
            assert "algorithm" in explanation.lower()
    
    @pytest.mark.asyncio
    async def test_generate_anti_pattern_fixes(self, mock_settings):
        """Test anti-pattern fix generation."""
        with patch('src.generator.llm_client.genai'), \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            mock_response_text = '''{
                "pattern_type": "god_object",
                "pattern_description": "Class with too many responsibilities",
                "fix_description": "Split into multiple smaller classes",
                "code_before": "class GodClass:\\n    # too much code",
                "code_after": "class SpecificClass1:\\n    pass\\n\\nclass SpecificClass2:\\n    pass",
                "step_by_step": ["Identify responsibilities", "Extract classes", "Refactor usage"],
                "related_patterns": ["single_responsibility_principle"],
                "confidence": 0.8
            }'''
            
            client = LLMClient()
            
            with patch.object(client, '_generate_with_fallback', return_value=mock_response_text):
                fixes = await client.generate_anti_pattern_fixes(
                    [{"type": "god_object", "message": "Class is too large"}],
                    "class LargeClass: pass"
                )
            
            assert len(fixes) == 1
            fix = fixes[0]
            assert fix.pattern_type == "god_object"
            assert fix.confidence == 0.8
            assert len(fix.step_by_step) == 3
    
    def test_extract_json_from_response(self, mock_settings):
        """Test JSON extraction from LLM response."""
        with patch('src.generator.llm_client.genai'), \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            client = LLMClient()
            
            # Test with JSON array
            response1 = 'Here is the analysis: [{"type": "test", "value": 123}] as requested.'
            json1 = client._extract_json_from_response(response1)
            assert json1 == '[{"type": "test", "value": 123}]'
            
            # Test with JSON object
            response2 = 'Analysis results: {"status": "success", "data": [1,2,3]} - done!'
            json2 = client._extract_json_from_response(response2)
            assert json2 == '{"status": "success", "data": [1,2,3]}'
    
    def test_parse_recommendations_invalid_json(self, mock_settings):
        """Test recommendation parsing with invalid JSON."""
        with patch('src.generator.llm_client.genai'), \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            client = LLMClient()
            
            # Test with invalid JSON
            recommendations = client._parse_recommendations("invalid json", {})
            assert recommendations == []
    
    def test_health_check(self, mock_settings):
        """Test health check functionality."""
        with patch('src.generator.llm_client.genai') as mock_genai, \
             patch('src.generator.llm_client.AsyncOpenAI') as mock_openai:
            
            mock_genai.GenerativeModel.return_value = MagicMock()
            mock_openai.return_value = MagicMock()
            
            client = LLMClient()
            health = client.health_check()
            
            assert health["status"] == "healthy"
            assert health["primary_provider"] == "gemini"
            assert health["fallback_provider"] == "openai"
            assert health["providers"]["gemini"]["available"] is True
            assert health["providers"]["openai"]["available"] is True
            assert health["available_providers"] == 2
    
    def test_format_helper_methods(self, mock_settings, sample_file_analysis):
        """Test formatting helper methods."""
        with patch('src.generator.llm_client.genai'), \
             patch('src.generator.llm_client.AsyncOpenAI'):
            
            client = LLMClient()
            
            # Test function formatting
            functions = [
                {"name": "func1", "line_start": 10},
                {"name": "func2", "line_start": 20}
            ]
            formatted = client._format_functions_summary(functions)
            assert "func1 (line 10)" in formatted
            assert "func2 (line 20)" in formatted
            
            # Test empty list
            empty_formatted = client._format_functions_summary([])
            assert empty_formatted == "No functions found."
            
            # Test anti-pattern formatting
            anti_patterns = [
                {"type": "long_method", "severity": "warning", "message": "Method too long"}
            ]
            formatted_patterns = client._format_anti_patterns(anti_patterns)
            assert "long_method (warning)" in formatted_patterns


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.asyncio
    async def test_generate_code_recommendations(self, mock_settings, sample_snippet_analysis):
        """Test convenience function for generating recommendations."""
        with patch('src.generator.llm_client.LLMClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client.generate_recommendations.return_value = [
                Recommendation(
                    type=RecommendationType.STYLE,
                    severity="info",
                    line_start=1,
                    line_end=1,
                    message="Test recommendation",
                    explanation="Test explanation",
                    confidence=0.8
                )
            ]
            mock_client_class.return_value = mock_client
            
            recommendations = await generate_code_recommendations(sample_snippet_analysis)
            
            assert len(recommendations) == 1
            assert recommendations[0].type == RecommendationType.STYLE
            mock_client.generate_recommendations.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])