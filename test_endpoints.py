#!/usr/bin/env python3
"""Simple test script to verify API endpoints are working."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.api.app import app
from fastapi.testclient import TestClient


def test_api_endpoints():
    """Test basic API endpoint functionality."""
    client = TestClient(app)
    
    # Test health endpoint
    print("Testing health endpoint...")
    response = client.get("/api/v1/health")
    print(f"Health check status: {response.status_code}")
    if response.status_code == 200:
        health_data = response.json()
        print(f"Service status: {health_data.get('status', 'unknown')}")
    else:
        print(f"Health check failed: {response.text}")
    
    # Test analyze endpoint
    print("\nTesting analyze endpoint...")
    analyze_request = {
        "filename": "test.py",
        "content": "def hello_world():\n    print('Hello, World!')\n    return True",
        "language": "python"
    }
    
    response = client.post(
        "/api/v1/analyze",
        json=analyze_request,
        headers={"X-API-Key": "sk-test-key-for-testing"}
    )
    print(f"Analyze status: {response.status_code}")
    if response.status_code == 202:
        analyze_data = response.json()
        analysis_id = analyze_data.get("analysis_id")
        print(f"Analysis ID: {analysis_id}")
        print(f"Status: {analyze_data.get('status')}")
        
        # Test getting analysis status
        if analysis_id:
            print(f"\nTesting analysis status endpoint...")
            status_response = client.get(
                f"/api/v1/analysis/{analysis_id}/status",
                headers={"X-API-Key": "sk-test-key-for-testing"}
            )
            print(f"Status check: {status_response.status_code}")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"Analysis status: {status_data.get('status')}")
    else:
        print(f"Analyze failed: {response.text}")
    
    # Test search endpoint
    print("\nTesting search endpoint...")
    search_request = {
        "query": "hello world function",
        "language": "python",
        "limit": 10
    }
    
    response = client.post(
        "/api/v1/search",
        json=search_request,
        headers={"X-API-Key": "sk-test-key-for-testing"}
    )
    print(f"Search status: {response.status_code}")
    if response.status_code == 200:
        search_data = response.json()
        print(f"Search results: {search_data.get('total_count', 0)} found")
        print(f"Search time: {search_data.get('search_time_ms', 0):.2f}ms")
    else:
        print(f"Search failed: {response.text}")
    
    # Test repositories list endpoint
    print("\nTesting repositories list endpoint...")
    response = client.get(
        "/api/v1/repositories",
        headers={"X-API-Key": "sk-test-key-for-testing"}
    )
    print(f"Repositories list status: {response.status_code}")
    if response.status_code == 200:
        repos_data = response.json()
        print(f"Found {len(repos_data)} repositories")
    else:
        print(f"Repositories list failed: {response.text}")
    
    print("\nAPI endpoint tests completed!")


if __name__ == "__main__":
    test_api_endpoints()