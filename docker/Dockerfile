# Multi-stage build for optimized image size
FROM python:3.13-slim as builder

# Install build dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy project files for dependency resolution
COPY pyproject.toml .
COPY uv.toml .
COPY README.md .

# Install Python dependencies using uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
RUN uv sync --frozen --no-dev

# Production stage
FROM python:3.13-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv in production for runtime
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create non-root user
RUN useradd -m -u 1000 appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application code
COPY --chown=appuser:appuser . .

# Set environment variables
ENV PATH=/app/.venv/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]