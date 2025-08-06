# VibeCode AI Mentor - Project Context

## 🎯 Project Goal
Build an AI-powered code quality analysis tool using TiDB Cloud Vector Search for the TiDB Future App Hackathon 2025.

## 🏆 Hackathon Requirements
- **Mandatory**: Use TiDB Cloud with Vector Search
- **Timeline**: 48-72 hours to build MVP
- **Demo**: Working prototype with impressive presentation

## 🚀 Core Concept
The system analyzes source code, finds similar patterns and anti-patterns through vector search in TiDB, and generates improvement recommendations using LLM.

## 🛠 Technology Stack

### Core
- **Python 3.13** - pattern matching, improved async support
- **TiDB Cloud** - vector database with SQL support
- **FastAPI** - async web framework
- **Ruff** - linting and formatting

### AI/ML
- **Google Gemini 1.5 Flash** - primary LLM (8K tokens, $0.0001/1K)
- **OpenAI API** - fallback for embeddings
- **tree-sitter** - universal AST parser

### Infrastructure
- **Docker** - containerization
- **httpx** - async HTTP client
- **Rich** - beautiful CLI output

## 📊 TiDB Database Schema

```sql
-- Code snippets with vector indexes
CREATE TABLE code_snippets (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    content TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    language VARCHAR(50) NOT NULL,
    complexity_score INT,
    repo_name VARCHAR(200),
    VECTOR INDEX idx_embedding (embedding)
);

-- Best practice patterns
CREATE TABLE best_practices (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    pattern_name VARCHAR(200) NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    example_code TEXT,
    severity ENUM('info', 'warning', 'error'),
    VECTOR INDEX idx_pattern_embedding (embedding)
);
```

## 🎯 MVP Key Metrics

### Performance
- Indexing: < 1 sec per 1000 lines of code
- Analysis: < 5 sec per file
- Vector search: < 500ms

### Quality
- Recommendation accuracy: > 85%
- Test coverage: > 80% for critical path
- Analysis cost: < $0.01 per file

### Scalability
- Support repositories up to 10K files
- Batch processing up to 100 files in parallel
- Result caching in TiDB

## 🔍 TiDB Unique Features

1. **Hybrid search** - SQL + vectors in one query
2. **ACID transactions** - data consistency
3. **Serverless** - auto-scaling
4. **Free tier** - sufficient for demo

## 📁 Project Structure

```
vibecode/
├── src/
│   ├── core/           # Configuration, models
│   ├── db/             # TiDB client and queries
│   ├── analyzer/       # Code parsing and analysis
│   ├── embeddings/     # Vector generation
│   ├── search/         # Vector search
│   ├── generator/      # LLM recommendations
│   └── api/            # FastAPI endpoints
├── cli/                # CLI interface
├── tests/              # Tests
└── docker/             # Containerization
```

## 🏃‍♂️ Critical Path

1. **TiDB Setup** → Database schema with vector indexes
2. **Code Parser** → AST parsing for Python code
3. **Embeddings** → Vectorization via Gemini API
4. **Vector Search** → Find similar patterns
5. **LLM Generation** → Improvement recommendations
6. **Demo** → Impressive presentation

## 👥 Team Roles

- **backend-developer** - Core logic, API, integrations
- **ml-engineer** - Embeddings, vector search, prompts
- **devops-engineer** - TiDB, Docker, deployment, monitoring
- **qa-tester** - Testing, demo, quality validation

## 🎪 Demo Scenarios

1. **Live analysis** - Clone popular project, find issues
2. **Pattern comparison** - Show anti-patterns and best practices
3. **Improvement metrics** - Before/after visualization

## ⚡ Quick Start

```bash
# Clone repository
git clone https://github.com/team/vibecode
cd vibecode

# Setup environment
cp .env.example .env
# Add TIDB_CONNECTION_STRING and API keys

# Start services
docker-compose up -d

# Index repository
python -m cli index ./path/to/repo

# Analyze file
python -m cli analyze ./file.py
```

## 🔑 Keys to Victory

1. **Show TiDB power** - vector + SQL search
2. **Real value** - finds actual problems
3. **Easy to use** - 5 minutes to results
4. **Impressive demo** - visualization and metrics