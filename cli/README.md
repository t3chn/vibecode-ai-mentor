# VibeCode AI Mentor CLI

Command-line interface for VibeCode AI Mentor - an AI-powered code quality analysis tool using TiDB Cloud Vector Search.

## Installation

```bash
# Install from source
git clone <repository>
cd vibecode-ai-mentor
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

## Quick Start

1. **Set up the environment:**
   ```bash
   # Interactive setup (recommended)
   vibecode setup
   
   # Or set environment variables
   export TIDB_HOST="your-tidb-host"
   export TIDB_USER="your-username"
   export TIDB_PASSWORD="your-password"
   export GEMINI_API_KEY="your-gemini-key"
   ```

2. **Start the API server:**
   ```bash
   vibecode serve
   ```

3. **Index a repository:**
   ```bash
   vibecode index ./my-project
   ```

4. **Analyze code:**
   ```bash
   vibecode analyze ./my-file.py
   ```

5. **Search for patterns:**
   ```bash
   vibecode search "async function with error handling"
   ```

## Commands

### `vibecode setup`

Interactive setup wizard to configure database connections and API keys.

```bash
vibecode setup                    # Interactive setup
vibecode setup --non-interactive  # Use environment variables
vibecode setup --create-tables    # Also create database tables
```

**Options:**
- `--interactive` / `--non-interactive`: Choose setup mode
- `--config-file FILE`: Save configuration to specific file
- `--skip-db-test`: Skip database connection test
- `--create-tables`: Create database schema

### `vibecode analyze`

Analyze code files or directories for quality metrics and patterns.

```bash
vibecode analyze ./file.py
vibecode analyze ./project --recursive
vibecode analyze ./src -i "*.py" -e "*test*"
```

**Options:**
- `--recursive`, `-r`: Recursively analyze directories
- `--include-pattern`, `-i`: File patterns to include (e.g., '*.py')
- `--exclude-pattern`, `-e`: File patterns to exclude (e.g., '*test*')
- `--language`, `-l`: Programming language (default: python)
- `--metrics-only`: Only calculate metrics, skip chunking
- `--detailed`, `-d`: Show detailed analysis
- `--output-file`, `-o`: Save results to file

**Examples:**
```bash
# Analyze single file
vibecode analyze ./app.py

# Analyze project recursively
vibecode analyze ./my-project --recursive

# Filter by patterns
vibecode analyze ./src -i "*.py" -e "*test*" -e "*__pycache__*"

# Detailed analysis with JSON output
vibecode analyze ./app.py --detailed --json

# Save results to file
vibecode analyze ./project -r -o analysis.json
```

### `vibecode index`

Index a repository for vector search and AI analysis.

```bash
vibecode index ./my-project
vibecode index ./src --name "MyApp Core"
vibecode index ./repo --batch-size 100
```

**Options:**
- `--name`, `-n`: Repository name (default: directory name)
- `--include-pattern`, `-i`: File patterns to include
- `--exclude-pattern`, `-e`: File patterns to exclude  
- `--batch-size`, `-b`: Batch size for embedding generation (default: 50)
- `--force`, `-f`: Force re-indexing if repository exists
- `--dry-run`: Show what would be indexed without doing it
- `--skip-embeddings`: Index code without generating embeddings

**Examples:**
```bash
# Index repository
vibecode index ./my-project

# Custom name and patterns
vibecode index ./src --name "Core Services" -i "*.py" -e "*test*"

# Dry run to see what would be indexed
vibecode index ./project --dry-run

# Force re-index with larger batches
vibecode index ./project --force --batch-size 100
```

### `vibecode search`

Search for similar code patterns using vector similarity.

```bash
vibecode search "async function with error handling"
vibecode search "class with multiple methods" -n 10
vibecode search "database connection" --repository myapp
```

**Options:**
- `--language`, `-l`: Programming language filter (default: python)
- `--limit`, `-n`: Maximum results to return (default: 10)
- `--similarity-threshold`, `-t`: Minimum similarity (0.0-1.0, default: 0.7)
- `--repository`, `-r`: Filter by repository name
- `--file-pattern`, `-f`: Filter by file pattern
- `--chunk-type`, `-c`: Type of code chunks (function, class, block, all)
- `--show-code`, `-s`: Show code snippets in results
- `--detailed`, `-d`: Show detailed match information
- `--output-file`, `-o`: Save results to file

**Examples:**
```bash
# Natural language search
vibecode search "function that calculates total price"

# Code pattern search
vibecode search "def calculate_total(" --show-code

# Filtered search
vibecode search "unit test" --file-pattern "*test*" --limit 5

# High similarity threshold
vibecode search "error handling" -t 0.9 --detailed
```

### `vibecode serve`

Start the FastAPI server for the REST API.

```bash
vibecode serve
vibecode serve --host localhost --port 8080
vibecode serve --reload --workers 1
```

**Options:**
- `--host`, `-h`: Host to bind server (default: 0.0.0.0)
- `--port`, `-p`: Port to bind server (default: 8000)
- `--workers`, `-w`: Number of worker processes (default: 1)
- `--reload`: Enable auto-reload on code changes
- `--log-level`: Log level (debug, info, warning, error, critical)
- `--access-log` / `--no-access-log`: Enable/disable access logging

**Examples:**
```bash
# Start server with defaults
vibecode serve

# Development server with reload
vibecode serve --reload --log-level debug

# Production server
vibecode serve --workers 4 --no-access-log
```

### `vibecode status`

Check the status of the running API server.

```bash
vibecode status
vibecode status --host localhost --port 8080
```

## Configuration

### Configuration Files

VibeCode looks for configuration files in the following order:

1. File specified with `--config` option
2. `~/.vibecode/config.yaml`
3. `~/.vibecode/config.yml` 
4. `./.vibecode.yaml`
5. `./.vibecode.yml`

### Environment Variables

All settings can be configured via environment variables:

```bash
# Database
export TIDB_HOST="gateway01.us-west-2.prod.aws.tidbcloud.com"
export TIDB_PORT="4000"
export TIDB_USER="your_username"
export TIDB_PASSWORD="your_password"
export TIDB_DATABASE="vibecode"

# API Keys
export GEMINI_API_KEY="your_gemini_api_key"
export OPENAI_API_KEY="your_openai_api_key"

# Application
export VIBECODE_API_HOST="localhost"
export VIBECODE_API_PORT="8000"
export VIBECODE_DEFAULT_LANGUAGE="python"
export VIBECODE_OUTPUT_FORMAT="table"
```

### Example Configuration File

```yaml
# ~/.vibecode/config.yaml
tidb_host: "gateway01.us-west-2.prod.aws.tidbcloud.com"
tidb_port: 4000
tidb_user: "your_username"
tidb_password: "your_password"
tidb_database: "vibecode"

api_host: "localhost"
api_port: 8000

gemini_api_key: "your_gemini_api_key"
openai_api_key: "your_openai_api_key"

default_language: "python"
output_format: "table"
batch_size: 100

include_patterns:
  - "**/*.py"
  - "**/*.js"
  - "**/*.ts"

exclude_patterns:
  - "**/__pycache__/**"
  - "**/.*"
  - "**/node_modules/**"
  - "**/venv/**"
  - "**/.git/**"
```

## Global Options

These options are available for all commands:

- `--config`, `-c FILE`: Path to configuration file
- `--verbose`, `-v`: Enable verbose output
- `--quiet`, `-q`: Suppress output except errors
- `--json`: Output results in JSON format
- `--version`: Show version information
- `--help`: Show help message

## Output Formats

VibeCode supports multiple output formats:

- **table** (default): Human-readable tables with colors
- **json**: Machine-readable JSON format
- **yaml**: YAML format for configuration-like output

Set the default format in configuration or use `--json` flag.

## Examples

### Complete Workflow

```bash
# 1. Set up the environment
vibecode setup

# 2. Start the API server (in another terminal)
vibecode serve --reload

# 3. Index your project
vibecode index ./my-project --name "My Project"

# 4. Analyze specific files
vibecode analyze ./src/main.py --detailed

# 5. Search for patterns
vibecode search "async function with database query" --show-code

# 6. Check server status
vibecode status
```

### Batch Processing

```bash
# Index multiple repositories
for repo in proj1 proj2 proj3; do
    vibecode index ./$repo --batch-size 100
done

# Analyze multiple files with JSON output
vibecode analyze ./src --recursive --json > analysis.json
```

### Integration with CI/CD

```bash
# Non-interactive setup in CI
export TIDB_HOST="..."
export TIDB_USER="..."
export TIDB_PASSWORD="..."
export GEMINI_API_KEY="..."

vibecode setup --non-interactive
vibecode analyze ./src --recursive --json > code-analysis.json
```

## Troubleshooting

### Common Issues

1. **Database connection failed**
   ```bash
   # Check configuration
   vibecode setup --non-interactive
   
   # Test connection
   vibecode setup --skip-db-test
   ```

2. **API keys not working**
   ```bash
   # Verify keys are set
   echo $GEMINI_API_KEY
   
   # Reconfigure
   vibecode setup --interactive
   ```

3. **Server not starting**
   ```bash
   # Check if port is in use
   vibecode serve --port 8001
   
   # Check logs
   vibecode serve --log-level debug
   ```

### Debug Mode

Enable verbose output for debugging:

```bash
vibecode --verbose analyze ./file.py
vibecode --verbose --json search "pattern" > debug.json
```

## API Integration

The CLI can work alongside the REST API:

```bash
# Start server
vibecode serve &

# Use CLI while server is running
vibecode analyze ./file.py
vibecode search "pattern"

# Server provides web interface at http://localhost:8000/docs
```