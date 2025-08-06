# Backend Developer Tasks - VibeCode AI Mentor

## 🎯 Role Overview
You are the backend developer responsible for building the core application logic, API endpoints, and database integrations for the VibeCode AI Mentor hackathon project.

## 📋 Primary Responsibilities

### 1. Code Analysis Engine
- Implement Python code parser using tree-sitter
- Extract AST nodes (functions, classes, methods)
- Calculate complexity metrics (cyclomatic, cognitive)
- Design efficient chunking strategy for embeddings
- Handle edge cases and malformed code

### 2. FastAPI Application
- Setup async FastAPI project structure
- Implement RESTful endpoints:
  ```python
  POST /api/v1/analyze      # Analyze single file
  POST /api/v1/index        # Index entire repository
  GET  /api/v1/search       # Search similar patterns
  GET  /api/v1/health       # Health check
  WS   /api/v1/ws/progress  # Real-time progress updates
  ```
- Add request validation with Pydantic models
- Implement proper error handling and logging
- Setup CORS for web clients

### 3. Database Integration
- Create async TiDB client with connection pooling
- Implement efficient bulk insert operations
- Design database queries for vector similarity search
- Add transaction support for consistency
- Implement query result caching

### 4. Business Logic
- Create service layer for code analysis
- Implement pattern matching algorithms
- Build recommendation aggregation logic
- Add rate limiting and quota management
- Design modular, testable architecture

## 🛠 Technical Requirements

### Dependencies to Use
```python
# Core
fastapi >= 0.104.0
uvicorn[standard]
pydantic >= 2.0
python-multipart

# Database
pymysql
aiomysql
sqlalchemy >= 2.0

# Code Analysis
tree-sitter
tree-sitter-python

# Utilities
httpx
tenacity
python-jose[cryptography]
```

### Code Standards
- Use Python 3.13 features (pattern matching, improved typing)
- Follow async/await patterns throughout
- Implement proper dependency injection
- Write type hints for all functions
- Keep functions under 20 lines
- Test coverage > 80%

## 📁 Files You Own

```
src/
├── core/
│   ├── __init__.py
│   ├── config.py         # Settings management
│   ├── models.py         # Pydantic models
│   └── exceptions.py     # Custom exceptions
├── analyzer/
│   ├── __init__.py
│   ├── parser.py         # Tree-sitter integration
│   ├── metrics.py        # Code metrics calculation
│   └── chunker.py        # Smart chunking logic
├── api/
│   ├── __init__.py
│   ├── app.py           # FastAPI app setup
│   ├── routes.py        # API endpoints
│   ├── schemas.py       # Request/response schemas
│   └── websocket.py     # WebSocket handlers
└── services/
    ├── __init__.py
    ├── analysis.py      # Analysis orchestration
    ├── indexing.py      # Repository indexing
    └── search.py        # Search logic
```

## 🔄 Integration Points

### With ML Engineer
- Provide code chunks for embedding generation
- Receive embedding vectors for storage
- Pass search queries and get results

### With DevOps Engineer
- Use provided TiDB connection string
- Follow Docker best practices
- Implement health check endpoints

### With QA Tester
- Provide API documentation
- Support test data generation
- Fix reported bugs quickly

## 📊 Success Metrics

### Performance Targets
- API response time < 500ms (p95)
- Handle 100+ concurrent requests
- Index 1000 lines/second
- Memory usage < 512MB

### Quality Standards
- Zero critical bugs
- All endpoints documented
- Comprehensive error messages
- Clean, maintainable code

## 🚀 Quick Start Commands

```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run development server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest tests/ -v --cov=src

# Format code
ruff format src/
ruff check src/ --fix
```

## ⏰ Timeline

### Phase 1 (Hours 6-14)
- [ ] Setup FastAPI project
- [ ] Implement tree-sitter parser
- [ ] Create basic API endpoints
- [ ] Add database models

### Phase 2 (Hours 14-26)
- [ ] Complete all endpoints
- [ ] Add WebSocket support
- [ ] Implement caching
- [ ] Optimize performance

### Phase 3 (Hours 26-42)
- [ ] Integration testing
- [ ] Bug fixes
- [ ] Documentation
- [ ] Performance tuning

## 🎯 Definition of Done

- [ ] All endpoints working with < 500ms response
- [ ] 80%+ test coverage achieved
- [ ] API documentation complete
- [ ] No Ruff errors or warnings
- [ ] Integration tests passing
- [ ] Ready for demo