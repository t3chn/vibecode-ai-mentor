# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VibeCode AI Mentor is a hackathon project for TiDB Future App Hackathon 2025. It's an AI-powered code quality analysis tool that uses TiDB Cloud Vector Search to find similar code patterns and generate improvement recommendations via LLMs.

## Critical Commands

### Development Setup
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh  # On Unix/macOS
# Or: pip install uv  # Alternative installation method

# Create and activate virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (uv automatically manages virtual environment)
uv sync  # Install all dependencies including dev dependencies
# Or install just production dependencies: uv sync --no-dev

# For editable installation (development mode)
uv pip install -e .
uv pip install -e ".[dev]"  # Include development dependencies

# Setup environment
cp .env.example .env
# Edit .env to add: TIDB_HOST, TIDB_USER, TIDB_PASSWORD, GEMINI_API_KEY, OPENAI_API_KEY
```

### Running the Application
```bash
# Start API server (development) - using uv run
uv run uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Alternative: activate venv first, then run normally
source .venv/bin/activate
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Start with Docker
docker-compose -f docker/docker-compose.yml up -d

# CLI commands (after installation)
uv run vibecode index ./path/to/repo  # Index a repository
uv run vibecode analyze ./file.py      # Analyze a single file
uv run vibecode search "query"         # Search for similar patterns

# Or with activated venv:
vibecode index ./path/to/repo
vibecode analyze ./file.py
vibecode search "query"
```

### Code Quality
```bash
# Linting and formatting with uv
uv run ruff format src/ cli/ tests/  # Format code
uv run ruff check src/ cli/ tests/ --fix  # Lint and auto-fix

# Type checking
uv run mypy src/ cli/

# Testing
uv run pytest tests/ -v  # Run all tests
uv run pytest tests/api/test_endpoints.py::test_analyze_endpoint -v  # Run single test
uv run pytest --cov=src --cov-report=html  # Generate coverage report

# Performance testing
uv run locust -f tests/performance/load_test.py --host=http://localhost:8000

# Alternative: activate venv and run without uv prefix
source .venv/bin/activate
ruff format src/ cli/ tests/
ruff check src/ cli/ tests/ --fix
mypy src/ cli/
pytest tests/ -v
```

### Database Setup
```bash
# Initialize TiDB schema
uv run python scripts/setup_db.py

# Run migrations
uv run python scripts/migrate.py

# Alternative: with activated venv
source .venv/bin/activate
python scripts/setup_db.py
python scripts/migrate.py
```

## UV Workflow & Best Practices

### Why UV?
- **Speed**: 10-100x faster than pip for dependency resolution and installation
- **Reliability**: Deterministic dependency resolution with lockfile support
- **Simplicity**: Single tool for virtual environments and package management
- **Compatibility**: Full compatibility with pip and existing Python packaging standards

### Key UV Commands
```bash
# Quick start (all-in-one)
uv sync                    # Install all dependencies from pyproject.toml
uv sync --no-dev          # Install only production dependencies
uv sync --frozen          # Install from uv.lock without updating

# Development workflow
uv add fastapi            # Add new dependency
uv add --dev pytest       # Add development dependency
uv remove old-package     # Remove dependency
uv lock                   # Generate/update uv.lock file

# Running commands
uv run <command>          # Run command in project environment
uv run --no-sync <cmd>    # Skip sync, run command faster

# Environment management
uv venv                   # Create virtual environment (.venv)
uv venv --python 3.13     # Create with specific Python version
```

### Performance Tips
- Use `uv run --no-sync` for repeated commands if dependencies haven't changed
- Run `uv lock` after adding dependencies to ensure reproducible builds
- Use `uv sync --frozen` in CI/CD for faster, deterministic installs
- Cache uv installs in Docker with proper layer ordering

### Docker Integration
The project's Dockerfile is optimized for uv:
```dockerfile
# Uses uv's official Docker image for fast installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Optimized environment variables
ENV UV_COMPILE_BYTECODE=1  # Pre-compile for faster startup
ENV UV_LINK_MODE=copy      # Copy packages for better caching

# Fast, deterministic installs
RUN uv sync --frozen --no-dev
```

### CI/CD Integration
For GitHub Actions or similar CI systems:
```yaml
- name: Set up uv
  uses: astral-sh/setup-uv@v1
  with:
    version: "latest"

- name: Install dependencies
  run: uv sync --frozen

- name: Run tests
  run: uv run pytest
```

### Migration from pip to uv
For existing developers, use the migration script:
```bash
# Run the automated migration script
python scripts/migrate_to_uv.py

# Or manual migration:
uv venv                    # Create new .venv
uv sync                    # Install all dependencies
uv lock                    # Generate lockfile
```

### Migration Notes
- **pyproject.toml**: Primary dependency specification (use this)
- **requirements.txt**: Legacy file - can be removed after migration
- **uv.lock**: Generated lockfile for reproducible installs (commit to git)
- **Old venv**: Remove old `venv/`, `env/` directories after migration
- Dependencies are now managed via `uv add/remove` instead of manual editing

## Architecture & Key Design Decisions

### Agentic Workflow Architecture
The project follows a multi-agent design pattern with 4 specialized roles:
- **backend-developer**: Handles core logic, FastAPI endpoints, TiDB integration
- **ml-engineer**: Manages embeddings, vector search, LLM prompts
- **devops-engineer**: Configures TiDB Cloud, Docker, deployment
- **qa-tester**: Validates quality, prepares demos

### TiDB Vector Search Integration
- Uses `VECTOR(1536)` columns for storing embeddings
- Implements hybrid search combining vector similarity with SQL filters
- Example query pattern:
```sql
SELECT *, VEC_COSINE_DISTANCE(embedding, ?) as similarity 
FROM code_snippets 
WHERE language = ? AND similarity < 0.3
ORDER BY similarity ASC LIMIT 10;
```

### Code Analysis Pipeline
1. **Parsing**: tree-sitter extracts AST nodes from Python code
2. **Chunking**: Smart splitting into 512-2048 token chunks
3. **Embedding**: Gemini API generates vectors (OpenAI as fallback)
4. **Storage**: Bulk insert into TiDB with vector indexes
5. **Search**: Vector similarity search with metadata filtering
6. **Generation**: LLM creates recommendations from retrieved patterns

### API Design
- Async FastAPI with Pydantic validation
- WebSocket support for real-time progress updates
- Key endpoints: `/analyze`, `/index`, `/search`, `/recommendations`
- Response time target: < 500ms (p95)

## Performance Considerations

- **Batch Processing**: Process embeddings in batches of 100
- **Connection Pooling**: Use aiomysql for async database operations
- **Caching**: Cache results in TiDB with TTL
- **Rate Limiting**: Implement per-API key limits for LLM calls
- **Chunk Optimization**: Balance between context (larger chunks) and precision (smaller chunks)

## Hackathon-Specific Notes

### Timeline Constraints (48-72 hours)
- Phase 1 (6h): Infrastructure setup
- Phase 2 (20h): Core functionality  
- Phase 3 (16h): AI integration
- Phase 4 (8h): Polish & demo

### Demo Scenarios
1. Live analysis of popular repositories (Django, FastAPI)
2. Anti-pattern detection with best practice suggestions
3. Before/after metrics visualization

### Success Metrics
- Indexing: < 1 sec per 1000 lines
- Analysis: < 5 sec per file
- Vector search: < 500ms
- Recommendation accuracy: > 85%

## Working with Agent Task Files
Each agent has specific responsibilities documented in:
- `BACKEND_TASKS.md`: API, core logic, database integration
- `ML_TASKS.md`: Embeddings, LLM, vector search
- `DEVOPS_TASKS.md`: Infrastructure, deployment, monitoring
- `QA_TASKS.md`: Testing, demo preparation, validation

When implementing features, refer to the appropriate task file for detailed requirements and success criteria.