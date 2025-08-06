"""LLM-powered code recommendation generation module.

This module provides comprehensive LLM integration for generating actionable
code recommendations, refactoring suggestions, and anti-pattern fixes.
"""

from .llm_client import (
    LLMClient,
    LLMProvider,
    RecommendationLevel,
    Recommendation,
    RefactorSuggestion,
    AntiPatternFix,
    generate_code_recommendations,
    generate_refactoring,
    explain_code,
    fix_anti_patterns
)

from .recommendation_service import (
    RecommendationService,
    recommendation_service,
    analyze_code_snippet,
    analyze_file,
    explain_code_snippet
)

__all__ = [
    # Core classes
    "LLMClient",
    "RecommendationService",
    
    # Enums
    "LLMProvider",
    "RecommendationLevel",
    
    # Data classes
    "Recommendation",
    "RefactorSuggestion", 
    "AntiPatternFix",
    
    # Convenience functions
    "generate_code_recommendations",
    "generate_refactoring",
    "explain_code",
    "fix_anti_patterns",
    "analyze_code_snippet",
    "analyze_file",
    "explain_code_snippet",
    
    # Service instances
    "recommendation_service"
]