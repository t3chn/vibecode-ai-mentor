#!/usr/bin/env python3
"""Example of integrating LLM client with API endpoints.

This shows how the recommendation service can be integrated into
the FastAPI application for real-time code analysis.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from generator.recommendation_service import RecommendationService, RecommendationLevel
from api.models import AnalyzeRequest, CodeRecommendation, RecommendationResponse


# Extended request models for LLM integration
class EnhancedAnalyzeRequest(AnalyzeRequest):
    """Enhanced analyze request with LLM options."""
    find_similar_patterns: bool = True
    recommendation_level: str = "detailed"  # basic, detailed, comprehensive
    include_refactoring: bool = True
    include_explanations: bool = False


class ExplainCodeRequest(BaseModel):
    """Request for code explanation."""
    code: str
    language: str = "python"
    context: Optional[Dict[str, Any]] = None


class RefactoringRequest(BaseModel):
    """Request for refactoring suggestions."""
    code: str
    language: str = "python"
    focus_areas: Optional[List[str]] = None  # complexity, readability, performance


# Example FastAPI integration
def create_enhanced_app() -> FastAPI:
    """Create FastAPI app with LLM-enhanced endpoints."""
    app = FastAPI(
        title="VibeCode AI Mentor - Enhanced API",
        description="Code analysis with LLM-powered recommendations",
        version="1.0.0"
    )
    
    # Initialize services
    recommendation_service = RecommendationService()
    
    @app.post("/analyze/enhanced", response_model=Dict[str, Any])
    async def analyze_code_enhanced(request: EnhancedAnalyzeRequest):
        """Enhanced code analysis with LLM recommendations."""
        try:
            # Map level string to enum
            level_map = {
                "basic": RecommendationLevel.BASIC,
                "detailed": RecommendationLevel.DETAILED,
                "comprehensive": RecommendationLevel.COMPREHENSIVE
            }
            level = level_map.get(request.recommendation_level, RecommendationLevel.DETAILED)
            
            # Perform comprehensive analysis
            result = await recommendation_service.analyze_and_recommend(
                code=request.content,
                filename=request.filename,
                language=request.language,
                find_similar=request.find_similar_patterns,
                level=level
            )
            
            # Transform to API response format
            recommendations = []
            for rec in result.get('recommendations', []):
                recommendations.append(CodeRecommendation(
                    type=rec['type'],
                    severity=rec['severity'],
                    line_start=rec['line_start'],
                    line_end=rec['line_end'],
                    message=rec['message'],
                    suggestion=rec.get('suggestion'),
                    explanation=rec['explanation'],
                    confidence=rec['confidence']
                ))
            
            # Include additional insights if requested
            enhanced_result = {
                "analysis_id": result['analysis_id'],
                "status": result['status'],
                "recommendations": [rec.dict() for rec in recommendations],
                "summary": result['summary'],
                "score": result['overall_score'],
                "analyzed_at": result['analyzed_at'],
                "processing_time_ms": result['processing_time_ms'],
                
                # Enhanced features
                "metrics": result['analysis']['metrics'],
                "recommendation_count": result['recommendation_count']
            }
            
            # Add refactoring suggestions if requested
            if request.include_refactoring and result.get('refactoring_suggestions'):
                enhanced_result["refactoring_suggestions"] = result['refactoring_suggestions']
            
            # Add anti-pattern fixes
            if result.get('anti_pattern_fixes'):
                enhanced_result["anti_pattern_fixes"] = result['anti_pattern_fixes']
            
            # Add similar patterns context
            if result.get('similar_patterns'):
                enhanced_result["similar_patterns"] = result['similar_patterns']
            
            return enhanced_result
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    @app.post("/explain", response_model=Dict[str, Any])
    async def explain_code(request: ExplainCodeRequest):
        """Generate detailed code explanation."""
        try:
            result = await recommendation_service.get_code_explanation(
                code=request.code,
                context=request.context
            )
            
            if 'error' in result:
                raise HTTPException(status_code=500, detail=result['error'])
            
            return {
                "explanation": result['explanation'],
                "metadata": {
                    "code_length": result['code_length'],
                    "language": result['language'],
                    "complexity": result['complexity'],
                    "has_issues": result['has_issues'],
                    "analysis_time_ms": result['analysis_time_ms']
                }
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
    
    @app.post("/refactor", response_model=Dict[str, Any])
    async def suggest_refactoring(request: RefactoringRequest):
        """Generate refactoring suggestions."""
        try:
            # Quick analysis for metrics
            from services.analysis import AnalysisService
            analysis_service = AnalysisService()
            snippet_analysis = analysis_service.analyze_code_snippet(request.code, request.language)
            
            if not snippet_analysis.metrics:
                raise HTTPException(status_code=400, detail="Unable to analyze code metrics")
            
            # Generate refactoring suggestions
            from generator.llm_client import LLMClient
            llm_client = LLMClient()
            
            suggestions = await llm_client.generate_refactoring_suggestions(
                code_snippet=request.code,
                metrics=snippet_analysis.metrics
            )
            
            return {
                "refactoring_suggestions": [
                    {
                        "refactor_type": sug.refactor_type,
                        "description": sug.description,
                        "code_before": sug.code_before,
                        "code_after": sug.code_after,
                        "confidence": sug.confidence,
                        "benefits": sug.benefits,
                        "risks": sug.risks,
                        "effort_level": sug.effort_level
                    }
                    for sug in suggestions
                ],
                "original_metrics": snippet_analysis.metrics,
                "suggestions_count": len(suggestions)
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Refactoring analysis failed: {str(e)}")
    
    @app.get("/health/llm", response_model=Dict[str, Any])
    async def llm_health_check():
        """Check health of LLM services."""
        try:
            health = await recommendation_service.health_check()
            return health
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")
    
    return app


async def demo_api_integration():
    """Demonstrate API integration with sample requests."""
    print("🌐 API Integration Demo")
    print("=" * 50)
    
    # Sample code for testing
    sample_code = '''
def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    total = 0
    for num in numbers:
        total += num
    return total / len(numbers)

def find_max(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
'''
    
    # Create service instance
    service = RecommendationService()
    
    print("\n📝 Testing enhanced analysis...")
    
    # Test enhanced analysis
    result = await service.analyze_and_recommend(
        code=sample_code,
        filename="test_functions.py",
        find_similar=False,
        level=RecommendationLevel.DETAILED
    )
    
    print(f"✅ Analysis completed: {result['status']}")
    print(f"📊 Overall Score: {result['overall_score']}/100")
    print(f"💡 Recommendations: {result['recommendation_count']}")
    
    # Show sample recommendations
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"\n  {i}. [{rec['type'].upper()}] {rec['message'][:60]}...")
    
    print("\n📖 Testing code explanation...")
    
    # Test code explanation
    explanation_result = await service.get_code_explanation(
        code="def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        context={"purpose": "Mathematical sequence generation"}
    )
    
    if 'explanation' in explanation_result:
        print(f"✅ Explanation generated ({len(explanation_result['explanation'])} chars)")
        print(f"🔍 Complexity: {explanation_result['complexity']}")
    else:
        print(f"❌ Explanation failed: {explanation_result.get('error', 'Unknown error')}")
    
    print("\n🏥 Testing health check...")
    
    # Test health check
    health = await service.health_check()
    print(f"✅ Service Status: {health['status'].upper()}")
    
    llm_health = health['components']['llm_client']
    print(f"🤖 LLM Providers: {llm_health['available_providers']}")
    
    print("\n" + "=" * 50)


def demo_request_examples():
    """Show example API request formats."""
    print("\n📄 Example API Requests")
    print("=" * 50)
    
    print("\n1. Enhanced Analysis Request:")
    print("""
POST /analyze/enhanced
{
  "filename": "example.py",
  "content": "def hello(): return 'world'",
  "language": "python",
  "find_similar_patterns": true,
  "recommendation_level": "detailed",
  "include_refactoring": true,
  "include_explanations": false
}
""")
    
    print("\n2. Code Explanation Request:")
    print("""
POST /explain
{
  "code": "def binary_search(arr, target): ...",
  "language": "python",
  "context": {
    "purpose": "Search algorithm",
    "difficulty": "intermediate"
  }
}
""")
    
    print("\n3. Refactoring Request:")
    print("""
POST /refactor
{
  "code": "def complex_function(): ...",
  "language": "python",
  "focus_areas": ["complexity", "readability"]
}
""")
    
    print("\n4. Health Check Request:")
    print("""
GET /health/llm
""")
    
    print("\n" + "=" * 50)


async def main():
    """Main demo function."""
    print("🚀 VibeCode AI Mentor - API Integration Demo")
    print("=" * 60)
    print("This demo shows how to integrate the LLM client with FastAPI")
    print("endpoints for real-time code analysis and recommendations.")
    print("=" * 60)
    
    try:
        # Run API integration demo
        await demo_api_integration()
        
        # Show example request formats
        demo_request_examples()
        
        print("\n🎉 API Integration Demo completed!")
        print("\nIntegration Features:")
        print("  ✅ Enhanced analysis endpoint with LLM recommendations")
        print("  ✅ Code explanation endpoint")
        print("  ✅ Refactoring suggestions endpoint")
        print("  ✅ LLM health monitoring endpoint")
        print("  ✅ Flexible request/response models")
        print("  ✅ Error handling and validation")
        
        print("\nTo use in production:")
        print("  1. Add these endpoints to your main FastAPI app")
        print("  2. Configure proper authentication and rate limiting")
        print("  3. Set up monitoring and logging")
        print("  4. Test with your specific code patterns")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have proper environment setup and API keys")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)