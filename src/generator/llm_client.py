"""LLM client for generating code recommendations using Gemini and OpenAI.

This module provides a comprehensive LLM client that generates actionable 
code recommendations, refactoring suggestions, and anti-pattern fixes.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import google.generativeai as genai
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.core.config import get_settings
from src.services.analysis import FileAnalysis, SnippetAnalysis
from src.search.vector_search import CodeMatch
from src.api.models import CodeRecommendation, RecommendationType

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Available LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"


class RecommendationLevel(str, Enum):
    """Recommendation detail levels."""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"


@dataclass
class Recommendation:
    """Individual code recommendation."""
    type: RecommendationType
    severity: str  # info, warning, error
    line_start: int
    line_end: int
    message: str
    suggestion: Optional[str] = None
    explanation: str = ""
    confidence: float = 0.0
    code_before: Optional[str] = None
    code_after: Optional[str] = None
    related_patterns: List[str] = field(default_factory=list)


@dataclass
class RefactorSuggestion:
    """Refactoring suggestion with before/after code."""
    refactor_type: str
    description: str
    code_before: str
    code_after: str
    confidence: float
    benefits: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    effort_level: str = "medium"  # low, medium, high


@dataclass
class AntiPatternFix:
    """Anti-pattern fix with detailed guidance."""
    pattern_type: str
    pattern_description: str
    fix_description: str
    code_before: str
    code_after: str
    step_by_step: List[str] = field(default_factory=list)
    related_patterns: List[str] = field(default_factory=list)
    confidence: float = 0.0


class LLMClient:
    """Multi-provider LLM client for code recommendations."""
    
    def __init__(self, primary_provider: LLMProvider = LLMProvider.GEMINI):
        """Initialize LLM client.
        
        Args:
            primary_provider: Primary LLM provider to use
        """
        self.settings = get_settings()
        self.primary_provider = primary_provider
        self.fallback_provider = LLMProvider.OPENAI if primary_provider == LLMProvider.GEMINI else LLMProvider.GEMINI
        
        # Initialize providers
        self._setup_providers()
        
        # Generation settings
        self.max_tokens = 4000
        self.temperature = 0.3  # Lower for more consistent recommendations
        self.top_p = 0.9
        
    def _setup_providers(self):
        """Setup LLM providers based on available API keys."""
        # Gemini setup
        if self.settings.gemini_api_key:
            try:
                genai.configure(api_key=self.settings.gemini_api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
            
        # OpenAI setup  
        if self.settings.openai_api_key:
            try:
                self.openai_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
                logger.info("OpenAI provider initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI: {e}")
                self.openai_client = None
        else:
            self.openai_client = None
            
        # Validate at least one provider is available
        if not self.gemini_model and not self.openai_client:
            raise ValueError("No LLM providers available. Please configure API keys.")
    
    async def generate_recommendations(
        self,
        code_analysis: Union[FileAnalysis, SnippetAnalysis],
        similar_patterns: Optional[List[CodeMatch]] = None,
        level: RecommendationLevel = RecommendationLevel.DETAILED
    ) -> List[Recommendation]:
        """Generate code quality recommendations.
        
        Args:
            code_analysis: Analysis results from AnalysisService
            similar_patterns: Similar code patterns from vector search
            level: Detail level for recommendations
            
        Returns:
            List of actionable recommendations
        """
        logger.info(f"Generating recommendations for {type(code_analysis).__name__}")
        
        # Build context from analysis
        context = self._build_analysis_context(code_analysis, similar_patterns)
        
        # Generate recommendations based on analysis type
        if isinstance(code_analysis, FileAnalysis):
            return await self._generate_file_recommendations(context, level)
        else:
            return await self._generate_snippet_recommendations(context, level)
    
    async def generate_refactoring_suggestions(
        self,
        code_snippet: str,
        metrics: Dict[str, Any],
        similar_patterns: Optional[List[CodeMatch]] = None
    ) -> List[RefactorSuggestion]:
        """Generate refactoring suggestions for complex code.
        
        Args:
            code_snippet: Code to refactor
            metrics: Code metrics from MetricsCalculator
            similar_patterns: Similar code examples
            
        Returns:
            List of refactoring suggestions
        """
        logger.info("Generating refactoring suggestions")
        
        # Build refactoring prompt
        prompt = self._build_refactoring_prompt(code_snippet, metrics, similar_patterns)
        
        # Generate suggestions
        response = await self._generate_with_fallback(prompt)
        
        # Parse and validate suggestions
        return self._parse_refactoring_suggestions(response)
    
    async def generate_code_explanation(
        self,
        code_snippet: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate clear explanation of code functionality.
        
        Args:
            code_snippet: Code to explain
            context: Optional context from analysis
            
        Returns:
            Human-readable code explanation
        """
        logger.info("Generating code explanation")
        
        prompt = self._build_explanation_prompt(code_snippet, context)
        response = await self._generate_with_fallback(prompt)
        
        return response.strip()
    
    async def generate_anti_pattern_fixes(
        self,
        anti_patterns: List[Dict[str, Any]],
        code_context: str,
        similar_patterns: Optional[List[CodeMatch]] = None
    ) -> List[AntiPatternFix]:
        """Generate fixes for detected anti-patterns.
        
        Args:
            anti_patterns: Detected anti-patterns from analysis
            code_context: Full code context
            similar_patterns: Similar code examples
            
        Returns:
            List of anti-pattern fixes with examples
        """
        logger.info(f"Generating fixes for {len(anti_patterns)} anti-patterns")
        
        fixes = []
        for pattern in anti_patterns:
            fix = await self._generate_single_anti_pattern_fix(
                pattern, code_context, similar_patterns
            )
            if fix:
                fixes.append(fix)
        
        return fixes
    
    async def _generate_with_fallback(
        self,
        prompt: str,
        provider: Optional[LLMProvider] = None
    ) -> str:
        """Generate response with automatic fallback to secondary provider.
        
        Args:
            prompt: Generation prompt
            provider: Specific provider to use (None for primary)
            
        Returns:
            Generated response text
        """
        target_provider = provider or self.primary_provider
        
        try:
            if target_provider == LLMProvider.GEMINI and self.gemini_model:
                return await self._generate_with_gemini(prompt)
            elif target_provider == LLMProvider.OPENAI and self.openai_client:
                return await self._generate_with_openai(prompt)
            else:
                raise ValueError(f"Provider {target_provider} not available")
                
        except Exception as e:
            logger.warning(f"Primary provider {target_provider} failed: {e}")
            
            # Try fallback provider
            if target_provider != self.fallback_provider:
                logger.info(f"Falling back to {self.fallback_provider}")
                return await self._generate_with_fallback(prompt, self.fallback_provider)
            else:
                raise Exception(f"All providers failed. Last error: {e}")
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Gemini."""
        try:
            response = await asyncio.to_thread(
                self.gemini_model.generate_content,
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_output_tokens=self.max_tokens,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise
    
    async def _generate_with_openai(self, prompt: str) -> str:
        """Generate response using OpenAI."""
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    def _build_analysis_context(
        self,
        analysis: Union[FileAnalysis, SnippetAnalysis],
        similar_patterns: Optional[List[CodeMatch]] = None
    ) -> Dict[str, Any]:
        """Build context dictionary from analysis results."""
        context = {
            "analysis_type": type(analysis).__name__,
            "language": analysis.language if hasattr(analysis, 'language') else "python",
        }
        
        # Add metrics if available
        if hasattr(analysis, 'metrics') and analysis.metrics:
            context["metrics"] = analysis.metrics
            
        # Add AST elements
        if hasattr(analysis, 'functions'):
            context["functions"] = analysis.functions
        if hasattr(analysis, 'classes'):
            context["classes"] = analysis.classes
        if hasattr(analysis, 'ast_elements'):
            context["ast_elements"] = analysis.ast_elements
            
        # Add issues if present
        if hasattr(analysis, 'issues'):
            context["issues"] = analysis.issues
            
        # Add similar patterns context
        if similar_patterns:
            context["similar_patterns"] = [
                {
                    "content": pattern.content[:200] + "..." if len(pattern.content) > 200 else pattern.content,
                    "file_path": pattern.file_path,
                    "similarity_score": 1.0 - pattern.similarity_score,
                    "repository_name": getattr(pattern, 'repository_name', 'unknown')
                }
                for pattern in similar_patterns[:5]  # Limit to top 5
            ]
        
        return context
    
    async def _generate_file_recommendations(
        self,
        context: Dict[str, Any],
        level: RecommendationLevel
    ) -> List[Recommendation]:
        """Generate recommendations for file analysis."""
        prompt = self._build_file_recommendations_prompt(context, level)
        response = await self._generate_with_fallback(prompt)
        
        return self._parse_recommendations(response, context)
    
    async def _generate_snippet_recommendations(
        self,
        context: Dict[str, Any],
        level: RecommendationLevel
    ) -> List[Recommendation]:
        """Generate recommendations for snippet analysis."""
        prompt = self._build_snippet_recommendations_prompt(context, level)
        response = await self._generate_with_fallback(prompt)
        
        return self._parse_recommendations(response, context)
    
    def _build_file_recommendations_prompt(
        self,
        context: Dict[str, Any],
        level: RecommendationLevel
    ) -> str:
        """Build prompt for file-level recommendations."""
        metrics = context.get("metrics", {})
        functions = context.get("functions", [])
        classes = context.get("classes", [])
        similar_patterns = context.get("similar_patterns", [])
        
        prompt = f"""Analyze the following Python code file and provide {level.value} recommendations for improvement.

## File Analysis Results:
- Total lines of code: {metrics.get('lines_of_code', 'N/A')}
- Cyclomatic complexity: {metrics.get('cyclomatic_complexity', 'N/A')}
- Maintainability index: {metrics.get('maintainability_index', 'N/A')}
- Functions count: {len(functions)}
- Classes count: {len(classes)}
- Risk level: {metrics.get('risk_level', 'N/A')}

## Functions Found:
{self._format_functions_summary(functions)}

## Classes Found:
{self._format_classes_summary(classes)}

## Anti-patterns Detected:
{self._format_anti_patterns(metrics.get('anti_patterns', []))}

## Similar Code Patterns:
{self._format_similar_patterns(similar_patterns)}

Please provide recommendations as a JSON array with the following structure:
[
  {{
    "type": "performance|security|style|best_practice|refactoring|bug_fix",
    "severity": "info|warning|error",
    "line_start": number,
    "line_end": number,
    "message": "Clear, actionable recommendation",
    "suggestion": "Specific code suggestion or null",
    "explanation": "Detailed explanation of why this matters",
    "confidence": 0.0-1.0,
    "code_before": "Current problematic code or null",
    "code_after": "Improved code example or null"
  }}
]

Focus on:
1. Code quality improvements based on metrics
2. Performance optimizations
3. Best practices from similar patterns
4. Security considerations
5. Maintainability enhancements

Provide specific, actionable recommendations with high confidence scores for clear issues."""

        return prompt
    
    def _build_snippet_recommendations_prompt(
        self,
        context: Dict[str, Any],
        level: RecommendationLevel
    ) -> str:
        """Build prompt for snippet-level recommendations."""
        metrics = context.get("metrics", {})
        ast_elements = context.get("ast_elements", {})
        issues = context.get("issues", [])
        similar_patterns = context.get("similar_patterns", [])
        
        prompt = f"""Analyze the following Python code snippet and provide {level.value} recommendations.

## Code Metrics:
- Lines of code: {metrics.get('lines_of_code', 'N/A')}
- Cyclomatic complexity: {metrics.get('cyclomatic_complexity', 'N/A')}
- Maintainability index: {metrics.get('maintainability_index', 'N/A')}
- Risk level: {metrics.get('risk_level', 'N/A')}

## AST Elements:
{self._format_ast_elements(ast_elements)}

## Issues Found:
{self._format_issues(issues)}

## Similar Patterns:
{self._format_similar_patterns(similar_patterns)}

Provide recommendations as JSON array:
[
  {{
    "type": "performance|security|style|best_practice|refactoring|bug_fix",
    "severity": "info|warning|error", 
    "line_start": number,
    "line_end": number,
    "message": "Clear recommendation",
    "suggestion": "Code suggestion or null",
    "explanation": "Why this matters",
    "confidence": 0.0-1.0,
    "code_before": "Current code or null",
    "code_after": "Improved code or null"
  }}
]

Focus on immediate improvements that can be applied to this specific snippet."""

        return prompt
    
    def _build_refactoring_prompt(
        self,
        code_snippet: str,
        metrics: Dict[str, Any],
        similar_patterns: Optional[List[CodeMatch]] = None
    ) -> str:
        """Build prompt for refactoring suggestions."""
        complexity = metrics.get('cyclomatic_complexity', 0)
        maintainability = metrics.get('maintainability_index', 100)
        
        prompt = f"""Analyze this Python code and suggest specific refactoring improvements.

## Code to Refactor:
```python
{code_snippet}
```

## Current Metrics:
- Cyclomatic complexity: {complexity}
- Maintainability index: {maintainability}
- Lines of code: {metrics.get('lines_of_code', 'N/A')}

## Similar Code Examples:
{self._format_similar_patterns(similar_patterns or [])}

Provide refactoring suggestions as JSON array:
[
  {{
    "refactor_type": "extract_method|reduce_complexity|improve_naming|remove_duplication|etc",
    "description": "What to refactor and why",
    "code_before": "Current problematic code",
    "code_after": "Refactored improved code",
    "confidence": 0.0-1.0,
    "benefits": ["List of specific benefits"],
    "risks": ["Potential risks or considerations"],
    "effort_level": "low|medium|high"
  }}
]

Focus on:
1. Reducing complexity and improving readability
2. Extracting reusable functions
3. Improving variable and function names
4. Eliminating code duplication
5. Following Python best practices"""

        return prompt
    
    def _build_explanation_prompt(
        self,
        code_snippet: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build prompt for code explanation."""
        context_info = ""
        if context:
            context_info = f"""
## Additional Context:
- Metrics: {context.get('metrics', {})}
- Purpose: {context.get('purpose', 'Unknown')}
"""

        prompt = f"""Explain what this Python code does in clear, human-readable language.

## Code to Explain:
```python
{code_snippet}
```
{context_info}

Provide a clear explanation that covers:
1. What the code does (main functionality)
2. How it works (key logic and flow)
3. Any notable patterns or techniques used
4. Potential issues or improvements

Write for someone who understands programming but may not be familiar with this specific code."""

        return prompt
    
    async def _generate_single_anti_pattern_fix(
        self,
        anti_pattern: Dict[str, Any],
        code_context: str,
        similar_patterns: Optional[List[CodeMatch]] = None
    ) -> Optional[AntiPatternFix]:
        """Generate fix for a single anti-pattern."""
        pattern_type = anti_pattern.get('type', 'unknown')
        pattern_message = anti_pattern.get('message', 'No description')
        
        prompt = f"""Generate a fix for this anti-pattern in Python code.

## Anti-Pattern Detected:
- Type: {pattern_type}
- Description: {pattern_message}
- Severity: {anti_pattern.get('severity', 'warning')}

## Code Context:
```python
{code_context}
```

## Similar Code Examples:
{self._format_similar_patterns(similar_patterns or [])}

Provide the fix as JSON:
{{
  "pattern_type": "{pattern_type}",
  "pattern_description": "Clear description of the anti-pattern",
  "fix_description": "How to fix it",
  "code_before": "Problematic code example",
  "code_after": "Fixed code example",
  "step_by_step": ["Step 1", "Step 2", "etc"],
  "related_patterns": ["Related patterns to watch for"],
  "confidence": 0.0-1.0
}}

Focus on providing a practical, actionable fix with clear before/after examples."""

        try:
            response = await self._generate_with_fallback(prompt)
            return self._parse_anti_pattern_fix(response, pattern_type)
        except Exception as e:
            logger.error(f"Failed to generate fix for {pattern_type}: {e}")
            return None
    
    def _parse_recommendations(
        self,
        response: str,
        context: Dict[str, Any]
    ) -> List[Recommendation]:
        """Parse LLM response into Recommendation objects."""
        try:
            # Extract JSON from response
            json_str = self._extract_json_from_response(response)
            recommendations_data = json.loads(json_str)
            
            recommendations = []
            for rec_data in recommendations_data:
                try:
                    recommendation = Recommendation(
                        type=RecommendationType(rec_data.get('type', 'best_practice')),
                        severity=rec_data.get('severity', 'info'),
                        line_start=rec_data.get('line_start', 1),
                        line_end=rec_data.get('line_end', 1),
                        message=rec_data.get('message', ''),
                        suggestion=rec_data.get('suggestion'),
                        explanation=rec_data.get('explanation', ''),
                        confidence=float(rec_data.get('confidence', 0.5)),
                        code_before=rec_data.get('code_before'),
                        code_after=rec_data.get('code_after'),
                        related_patterns=rec_data.get('related_patterns', [])
                    )
                    recommendations.append(recommendation)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid recommendation: {e}")
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Failed to parse recommendations: {e}")
            # Return empty list rather than failing completely
            return []
    
    def _parse_refactoring_suggestions(self, response: str) -> List[RefactorSuggestion]:
        """Parse refactoring suggestions from LLM response."""
        try:
            json_str = self._extract_json_from_response(response)
            suggestions_data = json.loads(json_str)
            
            suggestions = []
            for sug_data in suggestions_data:
                try:
                    suggestion = RefactorSuggestion(
                        refactor_type=sug_data.get('refactor_type', 'general'),
                        description=sug_data.get('description', ''),
                        code_before=sug_data.get('code_before', ''),
                        code_after=sug_data.get('code_after', ''),
                        confidence=float(sug_data.get('confidence', 0.5)),
                        benefits=sug_data.get('benefits', []),
                        risks=sug_data.get('risks', []),
                        effort_level=sug_data.get('effort_level', 'medium')
                    )
                    suggestions.append(suggestion)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping invalid refactoring suggestion: {e}")
                    
            return suggestions
            
        except Exception as e:
            logger.error(f"Failed to parse refactoring suggestions: {e}")
            return []
    
    def _parse_anti_pattern_fix(self, response: str, pattern_type: str) -> Optional[AntiPatternFix]:
        """Parse anti-pattern fix from LLM response."""
        try:
            json_str = self._extract_json_from_response(response)
            fix_data = json.loads(json_str)
            
            return AntiPatternFix(
                pattern_type=fix_data.get('pattern_type', pattern_type),
                pattern_description=fix_data.get('pattern_description', ''),
                fix_description=fix_data.get('fix_description', ''),
                code_before=fix_data.get('code_before', ''),
                code_after=fix_data.get('code_after', ''),
                step_by_step=fix_data.get('step_by_step', []),
                related_patterns=fix_data.get('related_patterns', []),
                confidence=float(fix_data.get('confidence', 0.5))
            )
            
        except Exception as e:
            logger.error(f"Failed to parse anti-pattern fix: {e}")
            return None
    
    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON from LLM response that may contain additional text."""
        # Look for JSON array or object
        start_markers = ['[', '{']
        end_markers = [']', '}']
        
        for start_marker, end_marker in zip(start_markers, end_markers):
            start_idx = response.find(start_marker)
            if start_idx != -1:
                # Find matching end marker
                bracket_count = 0
                for i, char in enumerate(response[start_idx:], start_idx):
                    if char == start_marker:
                        bracket_count += 1
                    elif char == end_marker:
                        bracket_count -= 1
                        if bracket_count == 0:
                            return response[start_idx:i+1]
        
        # If no JSON found, return original response and let parser handle error
        return response
    
    def _format_functions_summary(self, functions: List[Dict]) -> str:
        """Format functions summary for prompt."""
        if not functions:
            return "No functions found."
        
        summary = []
        for func in functions[:10]:  # Limit to first 10
            name = func.get('name', 'unknown')
            line_start = func.get('line_start', 'N/A')
            summary.append(f"- {name} (line {line_start})")
        
        if len(functions) > 10:
            summary.append(f"... and {len(functions) - 10} more functions")
        
        return "\n".join(summary)
    
    def _format_classes_summary(self, classes: List[Dict]) -> str:
        """Format classes summary for prompt."""
        if not classes:
            return "No classes found."
        
        summary = []
        for cls in classes[:5]:  # Limit to first 5
            name = cls.get('name', 'unknown')
            line_start = cls.get('line_start', 'N/A')
            summary.append(f"- {name} (line {line_start})")
        
        if len(classes) > 5:
            summary.append(f"... and {len(classes) - 5} more classes")
        
        return "\n".join(summary)
    
    def _format_anti_patterns(self, anti_patterns: List[Dict]) -> str:
        """Format anti-patterns for prompt."""
        if not anti_patterns:
            return "No anti-patterns detected."
        
        summary = []
        for pattern in anti_patterns:
            pattern_type = pattern.get('type', 'unknown')
            severity = pattern.get('severity', 'info')
            message = pattern.get('message', 'No description')
            summary.append(f"- {pattern_type} ({severity}): {message}")
        
        return "\n".join(summary)
    
    def _format_similar_patterns(self, patterns: List[Dict]) -> str:
        """Format similar patterns for prompt."""
        if not patterns:
            return "No similar patterns found."
        
        summary = []
        for pattern in patterns:
            file_path = pattern.get('file_path', 'unknown')
            similarity = pattern.get('similarity_score', 0)
            content_preview = pattern.get('content', '')[:100] + "..."
            summary.append(f"- {file_path} (similarity: {similarity:.2f}): {content_preview}")
        
        return "\n".join(summary)
    
    def _format_ast_elements(self, ast_elements: Dict) -> str:
        """Format AST elements for prompt."""
        if not ast_elements:
            return "No AST elements found."
        
        summary = []
        for element_type, elements in ast_elements.items():
            if elements:
                count = len(elements) if isinstance(elements, list) else 1
                summary.append(f"- {element_type}: {count}")
        
        return "\n".join(summary) if summary else "No significant AST elements."
    
    def _format_issues(self, issues: List[Dict]) -> str:
        """Format issues for prompt."""
        if not issues:
            return "No issues found."
        
        summary = []
        for issue in issues:
            issue_type = issue.get('type', 'unknown')
            severity = issue.get('severity', 'info')
            message = issue.get('message', 'No description')
            line = issue.get('line', 'N/A')
            summary.append(f"- {issue_type} ({severity}) line {line}: {message}")
        
        return "\n".join(summary)
    
    def health_check(self) -> Dict[str, Any]:
        """Check health status of LLM providers.
        
        Returns:
            Health status dictionary
        """
        status = {
            "primary_provider": self.primary_provider.value,
            "fallback_provider": self.fallback_provider.value,
            "providers": {
                "gemini": {
                    "available": self.gemini_model is not None,
                    "model": "gemini-1.5-flash" if self.gemini_model else None
                },
                "openai": {
                    "available": self.openai_client is not None,
                    "model": "gpt-4o-mini" if self.openai_client else None
                }
            },
            "settings": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        }
        
        # Overall health
        healthy_providers = sum(1 for p in status["providers"].values() if p["available"])
        status["status"] = "healthy" if healthy_providers >= 1 else "unhealthy"
        status["available_providers"] = healthy_providers
        
        return status


# Convenience functions for easy integration
async def generate_code_recommendations(
    analysis: Union[FileAnalysis, SnippetAnalysis],
    similar_patterns: Optional[List[CodeMatch]] = None,
    level: RecommendationLevel = RecommendationLevel.DETAILED
) -> List[Recommendation]:
    """Generate code recommendations using default LLM client."""
    client = LLMClient()
    return await client.generate_recommendations(analysis, similar_patterns, level)


async def generate_refactoring(
    code: str,
    metrics: Dict[str, Any],
    similar_patterns: Optional[List[CodeMatch]] = None
) -> List[RefactorSuggestion]:
    """Generate refactoring suggestions using default LLM client."""
    client = LLMClient()
    return await client.generate_refactoring_suggestions(code, metrics, similar_patterns)


async def explain_code(code: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Explain code using default LLM client."""
    client = LLMClient()
    return await client.generate_code_explanation(code, context)


async def fix_anti_patterns(
    anti_patterns: List[Dict[str, Any]],
    code_context: str,
    similar_patterns: Optional[List[CodeMatch]] = None
) -> List[AntiPatternFix]:
    """Fix anti-patterns using default LLM client."""
    client = LLMClient()
    return await client.generate_anti_pattern_fixes(anti_patterns, code_context, similar_patterns)