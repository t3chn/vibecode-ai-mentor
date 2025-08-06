# QA Tester Tasks - VibeCode AI Mentor

## 🎯 Role Overview
You are the QA tester responsible for ensuring quality, preparing demo scenarios, validating recommendations, and making sure the VibeCode AI Mentor performs flawlessly during the hackathon presentation.

## 📋 Primary Responsibilities

### 1. API Testing
- Test all REST endpoints thoroughly
- Validate request/response schemas
- Check error handling and edge cases
- Test WebSocket connections
- Measure response times

### 2. Quality Validation
- Verify code analysis accuracy
- Validate recommendation quality
- Test pattern matching precision
- Check similarity search relevance
- Ensure consistent results

### 3. Demo Preparation
- Create compelling demo scenarios
- Prepare test repositories
- Script demo walkthrough
- Test presentation flow
- Create backup plans

### 4. Performance Testing
- Load test API endpoints
- Measure indexing speed
- Test concurrent users
- Monitor resource usage
- Identify bottlenecks

## 🛠 Testing Requirements

### Test Categories
```yaml
Functional Testing:
  - Code parsing accuracy
  - Embedding generation
  - Vector search results
  - Recommendation quality
  - API functionality

Performance Testing:
  - Response time < 500ms
  - Concurrent users: 100+
  - Indexing speed: 1000 lines/sec
  - Memory usage < 512MB

Integration Testing:
  - TiDB connectivity
  - LLM API integration
  - End-to-end workflows
  - Error recovery

Demo Testing:
  - Scenario walkthroughs
  - Edge case handling
  - Visual presentation
  - Timing and flow
```

### Test Data Sets
```python
# Repositories for testing
test_repos = {
    "small": "https://github.com/requests/requests-oauthlib",  # ~2K lines
    "medium": "https://github.com/pallets/click",              # ~15K lines
    "large": "https://github.com/encode/httpx",                # ~30K lines
}

# Code samples with known issues
anti_patterns = [
    "mutable_defaults.py",
    "god_function.py",
    "deep_nesting.py",
    "bare_except.py"
]

# High-quality examples
best_practices = [
    "clean_architecture.py",
    "solid_principles.py",
    "async_patterns.py",
    "error_handling.py"
]
```

## 📁 Files You Own

```
tests/
├── api/
│   ├── test_endpoints.py
│   ├── test_websocket.py
│   └── test_schemas.py
├── integration/
│   ├── test_analysis_flow.py
│   ├── test_indexing.py
│   └── test_recommendations.py
├── performance/
│   ├── load_test.py
│   ├── benchmark.py
│   └── profiling.py
├── quality/
│   ├── test_accuracy.py
│   ├── test_patterns.py
│   └── validation.py
├── fixtures/
│   ├── code_samples/
│   ├── expected_results/
│   └── test_repos/
└── demo/
    ├── demo_script.md
    ├── scenarios/
    └── backup_data/
```

## 🔧 Testing Tools

### API Testing
```python
# Tools
import pytest
import httpx
import asyncio
from locust import HttpUser, task

# Example test
async def test_analyze_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/analyze",
            json={"code": "def hello(): pass"}
        )
        assert response.status_code == 200
        assert "recommendations" in response.json()
```

### Demo Scenarios
```markdown
## Scenario 1: Live Code Analysis
1. Open popular repository (httpx)
2. Select problematic file
3. Run analysis
4. Show identified issues
5. Display recommendations
6. Apply improvements
7. Re-analyze to show fixes

## Scenario 2: Pattern Recognition
1. Upload code with anti-pattern
2. System identifies similar bad patterns
3. Show best practice alternatives
4. Generate refactoring suggestions
5. Demonstrate improvement metrics
```

## 🔄 Integration Points

### With Backend Developer
- Report API bugs immediately
- Validate endpoint contracts
- Test error scenarios
- Verify response formats

### With ML Engineer
- Validate recommendation quality
- Test edge cases for embeddings
- Verify search accuracy
- Check LLM responses

### With DevOps Engineer
- Test in different environments
- Verify deployment process
- Check monitoring alerts
- Validate backups

## 📊 Success Metrics

### Quality Standards
- Zero critical bugs in demo
- All happy paths working
- < 2% false positive rate
- 95% test coverage
- Demo runs smoothly

### Performance Benchmarks
- API p95 latency < 500ms
- Indexing: 1K lines/sec
- Search results < 200ms
- 100 concurrent users
- No memory leaks

## 🚀 Testing Commands

```bash
# Run all tests
pytest tests/ -v

# API tests only
pytest tests/api/ -v

# Performance tests
locust -f tests/performance/load_test.py --host=http://localhost:8000

# Coverage report
pytest --cov=src --cov-report=html

# Demo dry run
python -m demo.run_scenarios
```

## ⏰ Timeline

### Phase 1 (Hours 14-26)
- [ ] Setup test framework
- [ ] Create test fixtures
- [ ] Write API tests
- [ ] Initial smoke tests

### Phase 2 (Hours 26-42)
- [ ] Integration testing
- [ ] Quality validation
- [ ] Performance testing
- [ ] Bug reporting

### Phase 3 (Hours 42-48)
- [ ] Demo preparation
- [ ] Final testing
- [ ] Scenario practice
- [ ] Backup creation

## 🎯 Definition of Done

- [ ] All API endpoints tested
- [ ] Performance benchmarks met
- [ ] Demo scenarios rehearsed
- [ ] No critical bugs
- [ ] Test report generated
- [ ] Backup demo ready

## 🎪 Demo Checklist

### Pre-Demo
- [ ] All scenarios tested 3x
- [ ] Backup data prepared
- [ ] Timing rehearsed
- [ ] Edge cases handled
- [ ] Internet backup plan

### Demo Must-Haves
- [ ] Fast, responsive UI
- [ ] Clear value demonstration
- [ ] No errors or crashes
- [ ] Smooth transitions
- [ ] Impressive metrics

### Post-Demo
- [ ] Collect feedback
- [ ] Document issues
- [ ] Save demo recording
- [ ] Archive test data
- [ ] Update test cases

## 🐛 Bug Report Template

```markdown
### Bug Title
[Clear, concise description]

### Severity
Critical | High | Medium | Low

### Steps to Reproduce
1. 
2. 
3. 

### Expected Result
[What should happen]

### Actual Result
[What actually happens]

### Environment
- Endpoint: 
- Payload:
- Response:

### Screenshots/Logs
[Attach if applicable]
```