"""SQLAlchemy models for VibeCode AI Mentor with TiDB vector support."""

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.mysql import CHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator, UserDefinedType

Base = declarative_base()


class UUID(TypeDecorator):
    """Platform-independent UUID type for SQLAlchemy."""

    impl = CHAR(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        return uuid.UUID(value)


class Vector(UserDefinedType):
    """TiDB VECTOR type for storing embeddings."""

    def __init__(self, dim: int):
        self.dim = dim

    def get_col_spec(self, **kw):
        return f"VECTOR({self.dim})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return value
            # TiDB expects vector as JSON array string
            if isinstance(value, (list, tuple)):
                return f"[{','.join(map(str, value))}]"
            return value
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            # Parse JSON array string back to list
            if isinstance(value, str):
                return [float(x) for x in value.strip("[]").split(",")]
            return value
        return process


class RepositoryStatus(enum.Enum):
    """Status of repository indexing."""

    PENDING = "pending"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class Repository(Base):
    """Track indexed repositories."""

    __tablename__ = "repositories"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, index=True)
    url = Column(String(500), nullable=True)
    last_indexed_at = Column(DateTime, nullable=True)
    total_files = Column(Integer, default=0)
    status = Column(
        Enum(RepositoryStatus),
        default=RepositoryStatus.PENDING,
        nullable=False,
        index=True,
    )
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    code_snippets = relationship(
        "CodeSnippet", back_populates="repository", cascade="all, delete-orphan"
    )


class CodeSnippet(Base):
    """Store code snippets with embeddings for vector search."""

    __tablename__ = "code_snippets"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    repository_id = Column(UUID, ForeignKey("repositories.id"), nullable=False)
    file_path = Column(String(500), nullable=False, index=True)
    language = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=True)  # Gemini/OpenAI embedding size
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    complexity_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    repository = relationship("Repository", back_populates="code_snippets")
    analysis_results = relationship(
        "AnalysisResult", back_populates="code_snippet", cascade="all, delete-orphan"
    )

    # Indexes are created in schema.py for vector columns


class AnalysisResult(Base):
    """Store code analysis results with recommendations."""

    __tablename__ = "analysis_results"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    snippet_id = Column(UUID, ForeignKey("code_snippets.id"), nullable=False)
    recommendations = Column(JSON, nullable=False, default=list)
    similar_patterns = Column(JSON, nullable=False, default=list)
    quality_score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)

    # Relationships
    code_snippet = relationship("CodeSnippet", back_populates="analysis_results")


# Additional models for caching and performance
class EmbeddingCache(Base):
    """Cache embeddings to avoid recomputation."""

    __tablename__ = "embedding_cache"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    embedding = Column(Vector(1536), nullable=False)
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)


class SearchCache(Base):
    """Cache search results for performance."""

    __tablename__ = "search_cache"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    query_hash = Column(String(64), nullable=False, index=True)
    language = Column(String(50), nullable=True)
    results = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)


# Type hints for better IDE support
def get_repository_by_name(name: str) -> Optional[Repository]:
    """Type hint helper for repository queries."""
    pass


def get_similar_snippets(
    embedding: list[float], limit: int = 10
) -> list[CodeSnippet]:
    """Type hint helper for vector similarity queries."""
    pass