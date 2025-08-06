"""Recommendation service that integrates analysis, search, and LLM generation.

This service orchestrates the complete recommendation pipeline:
1. Code analysis and metrics calculation
2. Vector search for similar patterns
3. LLM-powered recommendation generation
4. Result aggregation and storage
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.generator.llm_client import (
    LLMClient, Recommendation, RefactorSuggestion, AntiPatternFix,
    RecommendationLevel, generate_code_recommendations
)
from src.services.analysis import AnalysisService, FileAnalysis, SnippetAnalysis
from src.search.service import SearchServiceManager
from src.search.vector_search import CodeMatch
from src.api.models import CodeRecommendation, RecommendationType, AnalysisStatus
from src.db.connection import get_async_session
from src.core.config import get_settings

logger = logging.getLogger(__name__)


class RecommendationService:
    """Orchestrates the complete recommendation generation pipeline."""
    
    def __init__(self):
        """Initialize recommendation service."""
        self.settings = get_settings()
        self.analysis_service = AnalysisService()
        self.search_manager = SearchServiceManager()
        self.llm_client = LLMClient()
        
        # Cache for analysis results
        self._analysis_cache: Dict[str, Any] = {}
        
    async def analyze_and_recommend(
        self,
        code: str,
        filename: str = "snippet.py",
        language: str = "python",
        find_similar: bool = True,
        level: RecommendationLevel = RecommendationLevel.DETAILED,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Complete analysis and recommendation pipeline for code snippet.
        
        Args:
            code: Source code to analyze
            filename: Name of the file (for context)
            language: Programming language
            find_similar: Whether to search for similar patterns
            level: Detail level for recommendations
            session: Optional database session
            
        Returns:
            Complete analysis and recommendation results
        """
        analysis_id = str(uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting analysis {analysis_id} for {filename}")
        
        try:
            # Step 1: Analyze code
            logger.debug("Step 1: Analyzing code structure and metrics")
            snippet_analysis = self.analysis_service.analyze_code_snippet(code, language)
            
            # Step 2: Find similar patterns (if requested)
            similar_patterns = []
            if find_similar:
                logger.debug("Step 2: Searching for similar code patterns")
                try:
                    similar_results = await self.search_manager.find_similar_code(
                        code_snippet=code,
                        language=language,
                        threshold=0.7,
                        limit=10
                    )
                    similar_patterns = similar_results.get("similar_examples", [])
                    # Convert to CodeMatch-like objects for compatibility
                    similar_patterns = [self._dict_to_code_match(sp) for sp in similar_patterns]
                except Exception as e:
                    logger.warning(f"Similar pattern search failed: {e}")
            
            # Step 3: Generate recommendations
            logger.debug("Step 3: Generating LLM recommendations")
            recommendations = await self.llm_client.generate_recommendations(
                snippet_analysis, similar_patterns, level
            )
            
            # Step 4: Generate refactoring suggestions for complex code
            refactoring_suggestions = []
            if snippet_analysis.metrics and snippet_analysis.metrics.get('cyclomatic_complexity', 0) > 5:
                logger.debug("Step 4: Generating refactoring suggestions")
                try:
                    refactoring_suggestions = await self.llm_client.generate_refactoring_suggestions(
                        code, snippet_analysis.metrics, similar_patterns
                    )
                except Exception as e:
                    logger.warning(f"Refactoring generation failed: {e}")
            
            # Step 5: Generate anti-pattern fixes
            anti_pattern_fixes = []
            if snippet_analysis.metrics and snippet_analysis.metrics.get('anti_patterns'):
                logger.debug("Step 5: Generating anti-pattern fixes")
                try:
                    anti_pattern_fixes = await self.llm_client.generate_anti_pattern_fixes(
                        snippet_analysis.metrics['anti_patterns'], code, similar_patterns
                    )
                except Exception as e:
                    logger.warning(f"Anti-pattern fix generation failed: {e}")
            
            # Step 6: Calculate overall score
            overall_score = self._calculate_quality_score(snippet_analysis, recommendations)
            
            # Step 7: Generate summary
            summary = self._generate_analysis_summary(
                snippet_analysis, recommendations, refactoring_suggestions, anti_pattern_fixes
            )
            
            # Prepare final result
            result = {
                "analysis_id": analysis_id,
                "status": AnalysisStatus.COMPLETED,
                "filename": filename,
                "language": language,
                "analyzed_at": start_time.isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                
                # Analysis results
                "analysis": {
                    "metrics": snippet_analysis.metrics,
                    "ast_elements": snippet_analysis.ast_elements,
                    "issues": snippet_analysis.issues,
                    "analysis_time_ms": snippet_analysis.analysis_time_ms
                },
                
                # Recommendations
                "recommendations": [self._recommendation_to_dict(rec) for rec in recommendations],
                "refactoring_suggestions": [self._refactoring_to_dict(ref) for ref in refactoring_suggestions],
                "anti_pattern_fixes": [self._anti_pattern_fix_to_dict(fix) for fix in anti_pattern_fixes],
                
                # Similar patterns
                "similar_patterns": [
                    {
                        "file_path": sp.file_path,
                        "similarity_score": 1.0 - sp.similarity_score,
                        "content_preview": sp.content[:200] + "..." if len(sp.content) > 200 else sp.content,
                        "repository_name": getattr(sp, 'repository_name', 'unknown')
                    }
                    for sp in similar_patterns[:5]
                ],
                
                # Summary
                "summary": summary,
                "overall_score": overall_score,
                "recommendation_count": len(recommendations),
                "refactoring_count": len(refactoring_suggestions),
                "anti_pattern_count": len(anti_pattern_fixes)
            }
            
            logger.info(f"Analysis {analysis_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Analysis {analysis_id} failed: {e}", exc_info=True)
            return {
                "analysis_id": analysis_id,
                "status": AnalysisStatus.FAILED,
                "error": str(e),
                "analyzed_at": start_time.isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    async def analyze_file_comprehensive(
        self,
        file_path: str,
        find_similar: bool = True,
        level: RecommendationLevel = RecommendationLevel.DETAILED,
        session: Optional[AsyncSession] = None
    ) -> Dict[str, Any]:
        """Comprehensive analysis and recommendations for a file.
        
        Args:
            file_path: Path to the file to analyze
            find_similar: Whether to search for similar patterns
            level: Detail level for recommendations
            session: Optional database session
            
        Returns:
            Complete file analysis and recommendations
        """
        analysis_id = str(uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Starting file analysis {analysis_id} for {file_path}")
        
        try:
            # Step 1: Analyze file
            file_analysis = await self.analysis_service.analyze_file(file_path)
            
            if file_analysis.status != "success":
                return {
                    "analysis_id": analysis_id,
                    "status": AnalysisStatus.FAILED,
                    "error": file_analysis.error_message,
                    "file_path": file_path
                }
            
            # Step 2: Find similar patterns for each significant code chunk
            all_similar_patterns = []
            if find_similar and file_analysis.chunks:
                logger.debug("Searching for similar patterns in code chunks")
                for chunk in file_analysis.chunks[:5]:  # Limit to first 5 chunks
                    try:
                        similar_results = await self.search_manager.find_similar_code(
                            code_snippet=chunk["content"],
                            language=file_analysis.language,
                            threshold=0.7,
                            limit=5
                        )
                        chunk_patterns = similar_results.get("similar_examples", [])
                        all_similar_patterns.extend([self._dict_to_code_match(sp) for sp in chunk_patterns])
                    except Exception as e:
                        logger.warning(f"Similar pattern search failed for chunk: {e}")
            
            # Remove duplicates and limit
            unique_patterns = self._deduplicate_patterns(all_similar_patterns)[:10]
            
            # Step 3: Generate recommendations
            recommendations = await self.llm_client.generate_recommendations(
                file_analysis, unique_patterns, level
            )
            
            # Step 4: Generate refactoring suggestions for complex functions
            refactoring_suggestions = []
            if file_analysis.metrics and file_analysis.metrics.get('function_metrics'):
                complex_functions = [
                    f for f in file_analysis.metrics['function_metrics'] 
                    if f.get('complexity', 0) > 5 or f.get('is_complex', False)
                ]
                
                for func in complex_functions[:3]:  # Limit to top 3 complex functions
                    try:
                        # Extract function code (simplified - would need AST parsing for precision)
                        func_code = f"# Complex function: {func['name']}\n# Complexity: {func.get('complexity', 'N/A')}"
                        func_refactoring = await self.llm_client.generate_refactoring_suggestions(
                            func_code, {"cyclomatic_complexity": func.get('complexity', 0)}, unique_patterns
                        )
                        refactoring_suggestions.extend(func_refactoring)
                    except Exception as e:
                        logger.warning(f"Refactoring generation failed for function {func['name']}: {e}")
            
            # Step 5: Generate anti-pattern fixes
            anti_pattern_fixes = []
            if file_analysis.metrics and file_analysis.metrics.get('anti_patterns'):
                try:
                    # Read file content for context
                    with open(file_path, 'r', encoding='utf-8') as f:
                        file_content = f.read()
                    
                    anti_pattern_fixes = await self.llm_client.generate_anti_pattern_fixes(
                        file_analysis.metrics['anti_patterns'], file_content, unique_patterns
                    )
                except Exception as e:
                    logger.warning(f"Anti-pattern fix generation failed: {e}")
            
            # Step 6: Calculate scores and generate summary
            overall_score = self._calculate_quality_score(file_analysis, recommendations)
            summary = self._generate_file_analysis_summary(
                file_analysis, recommendations, refactoring_suggestions, anti_pattern_fixes
            )
            
            result = {
                "analysis_id": analysis_id,
                "status": AnalysisStatus.COMPLETED,
                "file_path": file_analysis.file_path,
                "language": file_analysis.language,
                "analyzed_at": start_time.isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                
                # File analysis results
                "analysis": {
                    "metrics": file_analysis.metrics,
                    "functions": file_analysis.functions,
                    "classes": file_analysis.classes,
                    "imports": file_analysis.imports,
                    "chunks": len(file_analysis.chunks),
                    "analysis_time_ms": file_analysis.analysis_time_ms,
                    "warnings": file_analysis.warnings
                },
                
                # Recommendations
                "recommendations": [self._recommendation_to_dict(rec) for rec in recommendations],
                "refactoring_suggestions": [self._refactoring_to_dict(ref) for ref in refactoring_suggestions],
                "anti_pattern_fixes": [self._anti_pattern_fix_to_dict(fix) for fix in anti_pattern_fixes],
                
                # Context
                "similar_patterns": [
                    {
                        "file_path": sp.file_path,
                        "similarity_score": 1.0 - sp.similarity_score,
                        "content_preview": sp.content[:200] + "..." if len(sp.content) > 200 else sp.content,
                        "repository_name": getattr(sp, 'repository_name', 'unknown')
                    }
                    for sp in unique_patterns[:5]
                ],
                
                # Summary
                "summary": summary,
                "overall_score": overall_score,
                "recommendation_count": len(recommendations),
                "refactoring_count": len(refactoring_suggestions),
                "anti_pattern_count": len(anti_pattern_fixes)
            }
            
            logger.info(f"File analysis {analysis_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"File analysis {analysis_id} failed: {e}", exc_info=True)
            return {
                "analysis_id": analysis_id,
                "status": AnalysisStatus.FAILED,
                "error": str(e),
                "file_path": file_path,
                "analyzed_at": start_time.isoformat(),
                "processing_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    async def get_code_explanation(
        self,
        code: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate detailed code explanation.
        
        Args:
            code: Code to explain
            context: Optional context information
            
        Returns:
            Code explanation with metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Quick analysis for context
            snippet_analysis = self.analysis_service.analyze_code_snippet(code)
            
            # Enhance context with analysis results
            enhanced_context = context or {}
            enhanced_context.update({
                "metrics": snippet_analysis.metrics,
                "ast_elements": snippet_analysis.ast_elements,
                "issues": snippet_analysis.issues
            })
            
            # Generate explanation
            explanation = await self.llm_client.generate_code_explanation(code, enhanced_context)
            
            return {
                "explanation": explanation,
                "code_length": len(code),
                "language": snippet_analysis.language,
                "complexity": snippet_analysis.metrics.get('cyclomatic_complexity', 0) if snippet_analysis.metrics else 0,
                "analysis_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
                "has_issues": len(snippet_analysis.issues) > 0 if snippet_analysis.issues else False
            }
            
        except Exception as e:
            logger.error(f"Code explanation failed: {e}")
            return {
                "error": str(e),
                "analysis_time_ms": (datetime.utcnow() - start_time).total_seconds() * 1000
            }
    
    def _dict_to_code_match(self, pattern_dict: Dict[str, Any]) -> CodeMatch:
        """Convert dictionary to CodeMatch-like object for compatibility."""
        # Create a simple object that mimics CodeMatch interface
        class SimpleCodeMatch:
            def __init__(self, data):
                self.file_path = data.get('file_path', '')
                self.content = data.get('content', '')
                self.similarity_score = 1.0 - data.get('similarity_score', 0.0)  # Convert to distance
                self.repository_name = data.get('repository_name', 'unknown')
                self.start_line = data.get('start_line', 1)
                self.end_line = data.get('end_line', 1)
                self.language = data.get('language', 'python')
        
        return SimpleCodeMatch(pattern_dict)
    
    def _deduplicate_patterns(self, patterns: List[Any]) -> List[Any]:
        """Remove duplicate patterns based on file path and content similarity."""
        seen_files = set()
        unique_patterns = []
        
        for pattern in patterns:
            file_key = pattern.file_path
            if file_key not in seen_files:
                seen_files.add(file_key)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _calculate_quality_score(
        self,
        analysis: Any,
        recommendations: List[Recommendation]
    ) -> int:
        """Calculate overall code quality score (0-100)."""
        base_score = 100
        
        # Deduct points based on issues
        if hasattr(analysis, 'issues') and analysis.issues:
            for issue in analysis.issues:
                severity = issue.get('severity', 'info')
                if severity == 'error':
                    base_score -= 15
                elif severity == 'warning':
                    base_score -= 10
                else:
                    base_score -= 5
        
        # Deduct points based on metrics
        if hasattr(analysis, 'metrics') and analysis.metrics:
            complexity = analysis.metrics.get('cyclomatic_complexity', 0)
            if complexity > 10:
                base_score -= (complexity - 10) * 2
            
            maintainability = analysis.metrics.get('maintainability_index', 100)
            if maintainability < 50:
                base_score -= (50 - maintainability) // 2
        
        # Deduct points based on recommendation severity
        for rec in recommendations:
            if rec.severity == 'error':
                base_score -= 10
            elif rec.severity == 'warning':
                base_score -= 5
            else:
                base_score -= 2
        
        return max(0, min(100, base_score))
    
    def _generate_analysis_summary(
        self,
        analysis: SnippetAnalysis,
        recommendations: List[Recommendation],
        refactoring_suggestions: List[RefactorSuggestion],
        anti_pattern_fixes: List[AntiPatternFix]
    ) -> str:
        """Generate human-readable analysis summary."""
        parts = []
        
        if analysis.metrics:
            complexity = analysis.metrics.get('cyclomatic_complexity', 0)
            maintainability = analysis.metrics.get('maintainability_index', 100)
            
            if complexity <= 5 and maintainability >= 80:
                parts.append("Code quality is good with low complexity and high maintainability.")
            elif complexity > 10:
                parts.append(f"Code has high complexity ({complexity}) and may be difficult to maintain.")
            elif maintainability < 50:
                parts.append(f"Code has low maintainability index ({maintainability:.1f}) and needs improvement.")
            else:
                parts.append("Code quality is acceptable but has room for improvement.")
        
        if recommendations:
            high_priority = sum(1 for r in recommendations if r.severity in ['error', 'warning'])
            if high_priority > 0:
                parts.append(f"Found {high_priority} high-priority issues that should be addressed.")
        
        if refactoring_suggestions:
            parts.append(f"Identified {len(refactoring_suggestions)} refactoring opportunities.")
        
        if anti_pattern_fixes:
            parts.append(f"Detected {len(anti_pattern_fixes)} anti-patterns with specific fixes.")
        
        if not parts:
            parts.append("No significant issues found. Code appears to follow good practices.")
        
        return " ".join(parts)
    
    def _generate_file_analysis_summary(
        self,
        analysis: FileAnalysis,
        recommendations: List[Recommendation],
        refactoring_suggestions: List[RefactorSuggestion],
        anti_pattern_fixes: List[AntiPatternFix]
    ) -> str:
        """Generate summary for file analysis."""
        parts = []
        
        if analysis.metrics:
            loc = analysis.metrics.get('lines_of_code', 0)
            complexity = analysis.metrics.get('average_complexity', 0)
            functions = analysis.metrics.get('functions_count', 0)
            
            parts.append(f"File contains {loc} lines of code with {functions} functions.")
            
            if complexity > 5:
                parts.append(f"Average complexity is {complexity:.1f}, which may indicate complex logic.")
            
            risk_level = analysis.metrics.get('risk_level', 'low')
            if risk_level in ['high', 'very_high']:
                parts.append(f"File has {risk_level} risk level and requires attention.")
        
        # Add recommendation summary
        if recommendations:
            error_count = sum(1 for r in recommendations if r.severity == 'error')
            warning_count = sum(1 for r in recommendations if r.severity == 'warning')
            
            if error_count > 0:
                parts.append(f"Found {error_count} critical issues.")
            if warning_count > 0:
                parts.append(f"Found {warning_count} warnings.")
        
        if refactoring_suggestions:
            parts.append(f"Identified {len(refactoring_suggestions)} refactoring opportunities.")
        
        return " ".join(parts) if parts else "File analysis completed successfully."
    
    def _recommendation_to_dict(self, rec: Recommendation) -> Dict[str, Any]:
        """Convert Recommendation to dictionary."""
        return {
            "type": rec.type.value,
            "severity": rec.severity,
            "line_start": rec.line_start,
            "line_end": rec.line_end,
            "message": rec.message,
            "suggestion": rec.suggestion,
            "explanation": rec.explanation,
            "confidence": rec.confidence,
            "code_before": rec.code_before,
            "code_after": rec.code_after,
            "related_patterns": rec.related_patterns
        }
    
    def _refactoring_to_dict(self, ref: RefactorSuggestion) -> Dict[str, Any]:
        """Convert RefactorSuggestion to dictionary."""
        return {
            "refactor_type": ref.refactor_type,
            "description": ref.description,
            "code_before": ref.code_before,
            "code_after": ref.code_after,
            "confidence": ref.confidence,
            "benefits": ref.benefits,
            "risks": ref.risks,
            "effort_level": ref.effort_level
        }
    
    def _anti_pattern_fix_to_dict(self, fix: AntiPatternFix) -> Dict[str, Any]:
        """Convert AntiPatternFix to dictionary."""
        return {
            "pattern_type": fix.pattern_type,
            "pattern_description": fix.pattern_description,
            "fix_description": fix.fix_description,
            "code_before": fix.code_before,
            "code_after": fix.code_after,
            "step_by_step": fix.step_by_step,
            "related_patterns": fix.related_patterns,
            "confidence": fix.confidence
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of recommendation service components."""
        try:
            # Check LLM client health
            llm_health = self.llm_client.health_check()
            
            # Check search service health  
            search_health = await self.search_manager.get_service_health()
            
            # Test analysis service
            analysis_test = True
            try:
                test_analysis = self.analysis_service.analyze_code_snippet("print('test')")
                analysis_test = test_analysis is not None
            except Exception as e:
                logger.warning(f"Analysis service test failed: {e}")
                analysis_test = False
            
            return {
                "status": "healthy" if all([
                    llm_health.get("status") == "healthy",
                    search_health.get("status") == "healthy", 
                    analysis_test
                ]) else "degraded",
                "components": {
                    "llm_client": llm_health,
                    "search_service": search_health,
                    "analysis_service": {
                        "status": "healthy" if analysis_test else "unhealthy",
                        "test_passed": analysis_test
                    }
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }


# Singleton instance for easy access
recommendation_service = RecommendationService()


# Convenience functions
async def analyze_code_snippet(
    code: str,
    filename: str = "snippet.py",
    find_similar: bool = True,
    level: RecommendationLevel = RecommendationLevel.DETAILED
) -> Dict[str, Any]:
    """Analyze code snippet and generate recommendations."""
    return await recommendation_service.analyze_and_recommend(
        code, filename, find_similar=find_similar, level=level
    )


async def analyze_file(
    file_path: str,
    find_similar: bool = True,
    level: RecommendationLevel = RecommendationLevel.DETAILED
) -> Dict[str, Any]:
    """Analyze file and generate comprehensive recommendations."""
    return await recommendation_service.analyze_file_comprehensive(
        file_path, find_similar=find_similar, level=level
    )


async def explain_code_snippet(
    code: str,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Get explanation for code snippet."""
    return await recommendation_service.get_code_explanation(code, context)