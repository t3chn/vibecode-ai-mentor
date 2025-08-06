"""Pydantic models for API requests and responses."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field, validator


# Enums
class AnalysisStatus(str, Enum):
    """Status of code analysis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RepositoryStatus(str, Enum):
    """Status of repository indexing."""
    PENDING = "pending"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class RecommendationType(str, Enum):
    """Type of code recommendation."""
    BUG_FIX = "bug_fix"
    PERFORMANCE = "performance"
    SECURITY = "security"
    STYLE = "style"
    BEST_PRACTICE = "best_practice"
    REFACTORING = "refactoring"


class MessageType(str, Enum):
    """WebSocket message types."""
    PROGRESS = "progress"
    ERROR = "error"
    COMPLETED = "completed"
    LOG = "log"


# Request Models
class AnalyzeRequest(BaseModel):
    """Request model for code analysis."""
    filename: str = Field(..., description="Name of the file being analyzed")
    content: str = Field(..., description="Source code content to analyze")
    language: str = Field("python", description="Programming language")
    repository_id: Optional[str] = Field(None, description="Associated repository ID")
    
    @validator("content")
    def validate_content(cls, v):
        """Ensure content is not empty."""
        if not v.strip():
            raise ValueError("Content cannot be empty")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "filename": "main.py",
                "content": "def hello_world():\n    print('Hello, World!')",
                "language": "python",
            }
        }


class IndexRequest(BaseModel):
    """Request model for repository indexing."""
    repository_path: str = Field(..., description="Path or URL to the repository")
    branch: str = Field("main", description="Branch to index")
    include_patterns: List[str] = Field(
        default=["**/*.py"],
        description="File patterns to include"
    )
    exclude_patterns: List[str] = Field(
        default=["**/test_*", "**/__pycache__/**"],
        description="File patterns to exclude"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "repository_path": "/path/to/repo",
                "branch": "main",
                "include_patterns": ["**/*.py"],
                "exclude_patterns": ["**/test_*"],
            }
        }


class SearchRequest(BaseModel):
    """Request model for code pattern search."""
    query: str = Field(..., description="Search query or code snippet")
    language: Optional[str] = Field(None, description="Filter by language")
    repository_ids: Optional[List[str]] = Field(None, description="Filter by repositories")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")
    similarity_threshold: float = Field(
        0.7, ge=0.0, le=1.0,
        description="Minimum similarity score"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "query": "async def fetch_data",
                "language": "python",
                "limit": 10,
                "similarity_threshold": 0.7,
            }
        }


# Response Models
class AnalysisResponse(BaseModel):
    """Response model for code analysis initiation."""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    status: AnalysisStatus = Field(..., description="Current analysis status")
    message: str = Field(..., description="Status message")
    estimated_time_seconds: Optional[int] = Field(None, description="Estimated completion time")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "message": "Analysis started",
                "estimated_time_seconds": 5,
                "created_at": "2025-01-20T10:00:00Z",
            }
        }


class CodeRecommendation(BaseModel):
    """Individual code recommendation."""
    type: str = Field(..., description="Type of recommendation")
    severity: str = Field(..., description="Severity level (info, warning, error)")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    message: str = Field(..., description="Recommendation message")
    suggestion: Optional[str] = Field(None, description="Suggested code fix")
    explanation: Optional[str] = Field(None, description="Detailed explanation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class RecommendationResponse(BaseModel):
    """Response model for analysis recommendations."""
    analysis_id: str = Field(..., description="Analysis identifier")
    status: AnalysisStatus = Field(..., description="Analysis status")
    recommendations: List[CodeRecommendation] = Field(..., description="List of recommendations")
    summary: str = Field(..., description="Analysis summary")
    score: int = Field(..., ge=0, le=100, description="Overall code quality score")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        schema_extra = {
            "example": {
                "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "recommendations": [],
                "summary": "Code quality is excellent",
                "score": 95,
                "analyzed_at": "2025-01-20T10:05:00Z",
            }
        }


class SearchResult(BaseModel):
    """Individual search result."""
    snippet_id: str = Field(..., description="Unique snippet identifier")
    repository_id: str = Field(..., description="Repository identifier")
    file_path: str = Field(..., description="File path within repository")
    content: str = Field(..., description="Code snippet content")
    line_start: int = Field(..., description="Starting line number")
    line_end: int = Field(..., description="Ending line number")
    similarity_score: float = Field(..., description="Similarity score")
    highlights: List[Dict[str, int]] = Field(..., description="Highlighted regions")


class SearchResponse(BaseModel):
    """Response model for pattern search."""
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total matching results")
    query: str = Field(..., description="Original search query")
    search_time_ms: float = Field(..., description="Search execution time")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [],
                "total_count": 0,
                "query": "async def fetch_data",
                "search_time_ms": 125.5,
            }
        }


class RepositoryResponse(BaseModel):
    """Response model for repository information."""
    repository_id: str = Field(..., description="Unique repository identifier")
    name: str = Field(..., description="Repository name")
    path: str = Field(..., description="Repository path or URL")
    status: str = Field(..., description="Indexing status")
    indexed_files: int = Field(..., description="Number of indexed files")
    total_files: int = Field(..., description="Total number of files")
    last_indexed_at: Optional[str] = Field(None, description="Last indexing timestamp")
    created_at: Optional[str] = Field(None, description="Repository creation timestamp")
    message: Optional[str] = Field(None, description="Status message")
    
    class Config:
        schema_extra = {
            "example": {
                "repository_id": "660e8400-e29b-41d4-a716-446655440001",
                "name": "my-project",
                "path": "/path/to/my-project",
                "status": "indexed",
                "indexed_files": 150,
                "total_files": 150,
                "last_indexed": "2025-01-20T10:00:00Z",
            }
        }


# WebSocket Models
class ProgressMessage(BaseModel):
    """WebSocket progress message."""
    type: MessageType = Field(MessageType.PROGRESS)
    repository_id: str = Field(..., description="Repository being indexed")
    status: RepositoryStatus = Field(..., description="Current status")
    indexed_files: int = Field(..., description="Files processed")
    total_files: int = Field(..., description="Total files to process")
    current_file: Optional[str] = Field(None, description="Currently processing file")
    percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "progress",
                "repository_id": "660e8400-e29b-41d4-a716-446655440001",
                "status": "indexing",
                "indexed_files": 50,
                "total_files": 150,
                "current_file": "src/main.py",
                "percentage": 33.3,
            }
        }


class ErrorMessage(BaseModel):
    """WebSocket error message."""
    type: MessageType = Field(MessageType.ERROR)
    error: str = Field(..., description="Error message")
    code: Optional[str] = Field(None, description="Error code")
    details: Optional[Dict] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "type": "error",
                "error": "Failed to index file",
                "code": "INDEX_ERROR",
                "details": {"file": "src/broken.py", "reason": "Syntax error"},
            }
        }