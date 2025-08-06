#!/usr/bin/env python3
"""Demonstration of LLM client functionality for code recommendations.

This script shows how to use the LLM client to generate various types of
recommendations and analyses for Python code.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator.llm_client import LLMClient, RecommendationLevel
from generator.recommendation_service import RecommendationService
from services.analysis import AnalysisService


# Sample code with various issues for demonstration
SAMPLE_CODE_COMPLEX = '''
def process_user_data(users, filters, options):
    result = []
    for user in users:
        if user['age'] > 18:
            if user['active'] == True:
                if 'premium' in filters:
                    if user['subscription'] == 'premium':
                        if options.get('include_inactive', False) or user['last_login'] > 30:
                            processed_user = {}
                            processed_user['id'] = user['id']
                            processed_user['name'] = user['name']
                            processed_user['email'] = user['email']
                            processed_user['score'] = user['score'] * 1.5 if user['score'] > 100 else user['score']
                            result.append(processed_user)
                else:
                    processed_user = {}
                    processed_user['id'] = user['id']
                    processed_user['name'] = user['name']
                    processed_user['email'] = user['email']
                    result.append(processed_user)
    return result

def calculate_stats(data):
    # Magic numbers everywhere
    score = 0
    for item in data:
        if item > 50:
            score += item * 0.75
        elif item > 25:
            score += item * 0.5
        else:
            score += item * 0.25
    
    return score / 100 if score > 1000 else score
'''

SAMPLE_CODE_SIMPLE = '''
def greet_user(name):
    """Greet a user by name."""
    if not name:
        return "Hello, stranger!"
    return f"Hello, {name}!"

def add_numbers(a, b):
    """Add two numbers together."""
    return a + b
'''

SAMPLE_CODE_WITH_ISSUES = '''
import os
import sys

class DataProcessor:
    def __init__(self):
        self.data = []
        self.results = []
        self.cache = {}
        self.config = {}
        self.settings = {}
        self.handlers = {}
        
    def process(self, input_data, mode="default"):
        # God object with too many responsibilities
        if mode == "fast":
            for item in input_data:
                self.data.append(item)
                result = item * 2  # Magic number
                self.results.append(result)
                if result > 100:  # Another magic number
                    self.cache[item] = result
        elif mode == "slow":
            for item in input_data:
                self.data.append(item)
                result = item * 3  # Magic number
                if result in self.cache:
                    self.results.append(self.cache[result])
                else:
                    processed = self.complex_calculation(result)
                    self.results.append(processed)
                    self.cache[result] = processed
        return self.results
    
    def complex_calculation(self, value):
        # Duplicate logic that should be extracted
        if value > 1000:
            return value * 0.9
        elif value > 500:
            return value * 0.95
        else:
            return value
            
    def another_complex_calculation(self, value):
        # More duplicate logic
        if value > 1000:
            return value * 0.85
        elif value > 500:
            return value * 0.92
        else:
            return value * 1.1
'''


async def demo_basic_analysis():
    """Demonstrate basic code analysis and recommendations."""
    print("🔍 Basic Code Analysis Demo")
    print("=" * 50)
    
    service = RecommendationService()
    
    # Analyze simple code
    print("\n📝 Analyzing simple code...")
    result = await service.analyze_and_recommend(
        SAMPLE_CODE_SIMPLE,
        filename="simple_example.py",
        find_similar=False,
        level=RecommendationLevel.BASIC
    )
    
    print(f"Status: {result['status']}")
    print(f"Overall Score: {result['overall_score']}/100")
    print(f"Summary: {result['summary']}")
    print(f"Recommendations: {len(result['recommendations'])}")
    
    for i, rec in enumerate(result['recommendations'][:3], 1):
        print(f"\n  {i}. [{rec['severity'].upper()}] {rec['message']}")
        if rec['suggestion']:
            print(f"     💡 {rec['suggestion']}")
    
    print("\n" + "=" * 50)


async def demo_complex_analysis():
    """Demonstrate analysis of complex code with issues."""
    print("\n🔧 Complex Code Analysis Demo")
    print("=" * 50)
    
    service = RecommendationService()
    
    # Analyze complex code
    print("\n📝 Analyzing complex code with issues...")
    result = await service.analyze_and_recommend(
        SAMPLE_CODE_COMPLEX,
        filename="complex_example.py",
        find_similar=False,
        level=RecommendationLevel.DETAILED
    )
    
    print(f"Status: {result['status']}")
    print(f"Overall Score: {result['overall_score']}/100")
    print(f"Processing Time: {result['processing_time_ms']:.1f}ms")
    
    # Show metrics
    metrics = result['analysis']['metrics']
    print(f"\n📊 Code Metrics:")
    print(f"  Lines of Code: {metrics.get('lines_of_code', 'N/A')}")
    print(f"  Cyclomatic Complexity: {metrics.get('cyclomatic_complexity', 'N/A')}")
    print(f"  Maintainability Index: {metrics.get('maintainability_index', 'N/A'):.1f}")
    print(f"  Risk Level: {metrics.get('risk_level', 'N/A')}")
    
    # Show recommendations
    print(f"\n💡 Recommendations ({len(result['recommendations'])}):")
    for i, rec in enumerate(result['recommendations'][:5], 1):
        print(f"\n  {i}. [{rec['type'].upper()}] [{rec['severity'].upper()}] Line {rec['line_start']}-{rec['line_end']}")
        print(f"     {rec['message']}")
        if rec['explanation']:
            print(f"     📖 {rec['explanation'][:100]}...")
    
    # Show refactoring suggestions
    if result['refactoring_suggestions']:
        print(f"\n🔄 Refactoring Suggestions ({len(result['refactoring_suggestions'])}):")
        for i, ref in enumerate(result['refactoring_suggestions'][:3], 1):
            print(f"\n  {i}. {ref['refactor_type'].title()}: {ref['description']}")
            print(f"     Effort: {ref['effort_level']} | Confidence: {ref['confidence']:.2f}")
            if ref['benefits']:
                print(f"     Benefits: {', '.join(ref['benefits'][:2])}")
    
    print("\n" + "=" * 50)


async def demo_anti_pattern_detection():
    """Demonstrate anti-pattern detection and fixes."""
    print("\n⚠️  Anti-Pattern Detection Demo")
    print("=" * 50)
    
    service = RecommendationService()
    
    # Analyze code with anti-patterns
    print("\n📝 Analyzing code with anti-patterns...")
    result = await service.analyze_and_recommend(
        SAMPLE_CODE_WITH_ISSUES,
        filename="issues_example.py",
        find_similar=False,
        level=RecommendationLevel.COMPREHENSIVE
    )
    
    print(f"Status: {result['status']}")
    print(f"Overall Score: {result['overall_score']}/100")
    
    # Show anti-pattern fixes
    if result['anti_pattern_fixes']:
        print(f"\n🛠️  Anti-Pattern Fixes ({len(result['anti_pattern_fixes'])}):")
        for i, fix in enumerate(result['anti_pattern_fixes'][:3], 1):
            print(f"\n  {i}. {fix['pattern_type'].title()}")
            print(f"     Problem: {fix['pattern_description']}")
            print(f"     Solution: {fix['fix_description']}")
            print(f"     Confidence: {fix['confidence']:.2f}")
            
            if fix['step_by_step']:
                print(f"     Steps:")
                for step_num, step in enumerate(fix['step_by_step'][:3], 1):
                    print(f"       {step_num}. {step}")
    
    print("\n" + "=" * 50)


async def demo_code_explanation():
    """Demonstrate code explanation functionality."""
    print("\n📚 Code Explanation Demo")
    print("=" * 50)
    
    service = RecommendationService()
    
    # Get explanation for complex function
    complex_function = '''
def binary_search(arr, target, low=0, high=None):
    """Binary search implementation with recursion."""
    if high is None:
        high = len(arr) - 1
    
    if low > high:
        return -1
    
    mid = (low + high) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binary_search(arr, target, low, mid - 1)
    else:
        return binary_search(arr, target, mid + 1, high)
'''
    
    print("\n📝 Explaining binary search function...")
    result = await service.get_code_explanation(
        complex_function,
        context={"purpose": "Search algorithm implementation"}
    )
    
    if 'explanation' in result:
        print(f"\n📖 Explanation:")
        print(f"{result['explanation']}")
        print(f"\n📊 Metadata:")
        print(f"  Code Length: {result['code_length']} characters")
        print(f"  Complexity: {result['complexity']}")
        print(f"  Has Issues: {result['has_issues']}")
        print(f"  Analysis Time: {result['analysis_time_ms']:.1f}ms")
    else:
        print(f"❌ Failed to generate explanation: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)


async def demo_health_check():
    """Demonstrate health check functionality."""
    print("\n🏥 Health Check Demo")
    print("=" * 50)
    
    service = RecommendationService()
    
    print("\n📝 Checking service health...")
    health = await service.health_check()
    
    print(f"Overall Status: {health['status'].upper()}")
    
    # LLM Client Health
    llm_health = health['components']['llm_client']
    print(f"\n🤖 LLM Client:")
    print(f"  Status: {llm_health['status']}")
    print(f"  Primary Provider: {llm_health['primary_provider']}")
    print(f"  Available Providers: {llm_health['available_providers']}")
    
    for provider, info in llm_health['providers'].items():
        status_icon = "✅" if info['available'] else "❌"
        print(f"  {status_icon} {provider.title()}: {info['model'] or 'Not available'}")
    
    # Search Service Health
    search_health = health['components']['search_service']
    print(f"\n🔍 Search Service:")
    print(f"  Status: {search_health['status']}")
    
    # Analysis Service Health
    analysis_health = health['components']['analysis_service']
    print(f"\n📊 Analysis Service:")
    print(f"  Status: {analysis_health['status']}")
    print(f"  Test Passed: {analysis_health['test_passed']}")
    
    print("\n" + "=" * 50)


async def demo_file_analysis():
    """Demonstrate file analysis if a Python file is available."""
    print("\n📁 File Analysis Demo")
    print("=" * 50)
    
    # Look for Python files to analyze
    current_dir = Path(__file__).parent
    src_dir = current_dir.parent / "src"
    
    # Find a suitable file to analyze
    test_files = [
        src_dir / "analyzer" / "metrics.py",
        src_dir / "services" / "analysis.py",
        current_dir / "parser_demo.py"
    ]
    
    target_file = None
    for file_path in test_files:
        if file_path.exists():
            target_file = file_path
            break
    
    if not target_file:
        print("⚠️  No suitable Python files found for analysis")
        return
    
    service = RecommendationService()
    
    print(f"\n📝 Analyzing file: {target_file.name}")
    result = await service.analyze_file_comprehensive(
        str(target_file),
        find_similar=False,  # Skip similar search for demo
        level=RecommendationLevel.BASIC
    )
    
    if result['status'] == 'completed':
        print(f"✅ Analysis completed successfully")
        print(f"Overall Score: {result['overall_score']}/100")
        
        analysis = result['analysis']
        print(f"\n📊 File Metrics:")
        if analysis['metrics']:
            metrics = analysis['metrics']
            print(f"  Lines of Code: {metrics.get('lines_of_code', 'N/A')}")
            print(f"  Functions: {len(analysis['functions'])}")
            print(f"  Classes: {len(analysis['classes'])}")
            print(f"  Average Complexity: {metrics.get('average_complexity', 'N/A'):.2f}")
        
        print(f"\n💡 Top Recommendations:")
        for i, rec in enumerate(result['recommendations'][:3], 1):
            print(f"  {i}. [{rec['severity'].upper()}] {rec['message'][:80]}...")
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 50)


async def main():
    """Run all demonstration scenarios."""
    print("🚀 VibeCode AI Mentor - LLM Client Demo")
    print("=" * 60)
    print("This demo shows the LLM client capabilities for code analysis")
    print("and recommendation generation.")
    print("=" * 60)
    
    try:
        # Run demonstrations
        await demo_basic_analysis()
        await demo_complex_analysis()
        await demo_anti_pattern_detection()
        await demo_code_explanation()
        await demo_file_analysis()
        await demo_health_check()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("  ✅ Code analysis and metrics calculation")
        print("  ✅ LLM-powered recommendation generation")
        print("  ✅ Refactoring suggestions")
        print("  ✅ Anti-pattern detection and fixes")
        print("  ✅ Code explanation generation")
        print("  ✅ File-level comprehensive analysis")
        print("  ✅ Health monitoring and fallback systems")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Make sure you have:")
        print("  - Proper environment setup (.env file)")
        print("  - Valid API keys (GEMINI_API_KEY or OPENAI_API_KEY)")
        print("  - All dependencies installed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)