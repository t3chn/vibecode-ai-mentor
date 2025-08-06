"""API routes for VibeCode AI Mentor."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_api_key, get_db_session, get_rate_limiter
from src.api.models import (
    AnalysisResponse,
    AnalysisStatus,
    AnalyzeRequest,
    IndexRequest,
    RecommendationResponse,
    RepositoryResponse,
    SearchRequest,
    SearchResponse,
)
from src.core.logging import logger
from src.services.analysis import AnalysisService
from src.generator.recommendation_service import RecommendationService
from src.search.service import SearchServiceManager
from src.embeddings.factory import get_embedding_manager
from src.db.repositories import RepositoryRepo, CodeSnippetRepo, AnalysisResultRepo
from src.db.models import Repository, RepositoryStatus

router = APIRouter(prefix="/api/v1", tags=["api"])


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Analyze a single code file",
    description="Submit a code file for quality analysis and get recommendations",
)
async def analyze_code(
    request: AnalyzeRequest,
    api_key: str = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
    limiter=Depends(get_rate_limiter),
):
    """Analyze a single code file and return analysis ID."""
    analysis_id = str(uuid4())
    
    try:
        logger.info(f"Starting analysis {analysis_id} for file: {request.filename}")
        
        # Initialize services
        recommendation_service = RecommendationService()
        
        # Validate input
        if not request.content or not request.content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Code content cannot be empty",
            )
        
        # Store analysis request (in a real system, you'd use a task queue)
        # For now, we'll process it synchronously but return immediately
        
        # Start background task for analysis
        async def process_analysis():
            try:
                result = await recommendation_service.analyze_and_recommend(
                    code=request.content,
                    filename=request.filename,
                    language=request.language or "python",
                    find_similar=True
                )
                
                # Store result in cache or database
                # This would typically be handled by a background worker
                analysis_cache[analysis_id] = result
                logger.info(f"Analysis {analysis_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Analysis {analysis_id} failed: {e}")
                analysis_cache[analysis_id] = {
                    "analysis_id": analysis_id,
                    "status": AnalysisStatus.FAILED,
                    "error": str(e)
                }
        
        # Start background processing
        asyncio.create_task(process_analysis())
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="processing",
            message=f"Analysis started for {request.filename}",
            estimated_time_seconds=15,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting analysis: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start analysis",
        )


@router.post(
    "/index",
    response_model=RepositoryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Index a repository",
    description="Index an entire repository for code pattern analysis",
)
async def index_repository(
    request: IndexRequest,
    api_key: str = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
    limiter=Depends(get_rate_limiter),
):
    """Index a repository and return repository ID."""
    try:
        # Validate repository path
        repo_path = Path(request.repository_path)
        if not repo_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Repository path does not exist: {request.repository_path}",
            )
        
        if not repo_path.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Repository path must be a directory",
            )
        
        logger.info(f"Starting indexing for repository: {request.repository_path}")
        
        # Initialize services
        analysis_service = AnalysisService()
        embedding_manager = get_embedding_manager()
        repo_repo = RepositoryRepo(db)
        snippet_repo = CodeSnippetRepo(db)
        
        # Create or get repository record
        repo_name = repo_path.name
        existing_repo = await repo_repo.get_by_name(repo_name)
        
        if existing_repo:
            repository = existing_repo
            await repo_repo.update_status(repository.id, RepositoryStatus.INDEXING)
        else:
            repository = await repo_repo.create(
                name=repo_name,
                url=request.repository_url
            )
        
        # Get initial file count
        python_files = list(repo_path.glob("**/*.py"))
        total_files = len(python_files)
        
        await repo_repo.update_status(repository.id, RepositoryStatus.INDEXING, total_files)
        
        # Start background indexing process
        async def process_indexing():
            try:
                logger.info(f"Processing {total_files} files in repository {repository.id}")
                
                # Analyze repository
                repo_analysis = await analysis_service.analyze_repository(
                    repo_path,
                    include_patterns=["**/*.py"],
                    exclude_patterns=["**/__pycache__/**", "**/.*", "**/venv/**", "**/node_modules/**"]
                )
                
                # Process chunks and generate embeddings
                snippet_data = []
                for file_analysis in repo_analysis.file_analyses:
                    if file_analysis.status == "success" and file_analysis.chunks:
                        for chunk in file_analysis.chunks:
                            # Generate embedding for chunk
                            try:
                                embedding = await embedding_manager.generate_embedding(
                                    chunk["content"]
                                )
                            except Exception as e:
                                logger.warning(f"Failed to generate embedding for chunk: {e}")
                                embedding = None
                            
                            snippet_data.append({
                                "repository_id": repository.id,
                                "file_path": file_analysis.file_path,
                                "language": file_analysis.language,
                                "content": chunk["content"],
                                "embedding": embedding,
                                "start_line": chunk["start_line"],
                                "end_line": chunk["end_line"],
                                "complexity_score": file_analysis.metrics.get("cyclomatic_complexity", 0.0) if file_analysis.metrics else 0.0
                            })
                
                # Store snippets in batches
                batch_size = 100
                for i in range(0, len(snippet_data), batch_size):
                    batch = snippet_data[i:i + batch_size]
                    await snippet_repo.create_batch(batch)
                    logger.info(f"Stored batch {i//batch_size + 1}/{(len(snippet_data) + batch_size - 1)//batch_size}")
                
                # Update repository status
                await repo_repo.update_status(repository.id, RepositoryStatus.COMPLETED)
                
                # Store indexing result
                indexing_cache[str(repository.id)] = {
                    "repository_id": str(repository.id),
                    "status": "completed",
                    "total_files": repo_analysis.total_files,
                    "analyzed_files": repo_analysis.analyzed_files,
                    "failed_files": repo_analysis.failed_files,
                    "total_snippets": len(snippet_data),
                    "processing_time_seconds": repo_analysis.total_time_seconds
                }
                
                logger.info(f"Repository indexing completed: {repository.id}")
                
            except Exception as e:
                logger.error(f"Repository indexing failed: {e}")
                await repo_repo.update_status(repository.id, RepositoryStatus.FAILED)
                indexing_cache[str(repository.id)] = {
                    "repository_id": str(repository.id),
                    "status": "failed",
                    "error": str(e)
                }
        
        # Start background processing
        asyncio.create_task(process_indexing())
        
        return RepositoryResponse(
            repository_id=str(repository.id),
            name=repository.name,
            path=request.repository_path,
            status="indexing",
            indexed_files=0,
            total_files=total_files,
            message="Repository indexing started",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting repository indexing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start repository indexing",
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search for similar code patterns",
    description="Search indexed repositories for similar code patterns using vector search",
)
async def search_patterns(
    request: SearchRequest,
    api_key: str = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
    limiter=Depends(get_rate_limiter),
):
    """Search for similar code patterns."""
    start_time = datetime.utcnow()
    
    try:
        logger.info(f"Searching for: {request.query}")
        
        # Validate input
        if not request.query or not request.query.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty",
            )
        
        # Initialize search service
        search_manager = SearchServiceManager()
        
        # Perform search
        results = await search_manager.quick_search(
            query=request.query,
            language=request.language or "python",
            limit=request.limit or 20,
            similarity_threshold=request.similarity_threshold,
            repository_filter=request.repository_filter
        )
        
        # Calculate search time
        search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        logger.info(f"Search completed: {len(results)} results in {search_time_ms:.2f}ms")
        
        return SearchResponse(
            results=results,
            total_count=len(results),
            query=request.query,
            search_time_ms=search_time_ms,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching patterns: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search patterns",
        )


@router.get(
    "/recommendations/{analysis_id}",
    response_model=RecommendationResponse,
    summary="Get analysis recommendations",
    description="Retrieve code improvement recommendations for a completed analysis",
)
async def get_recommendations(
    analysis_id: UUID,
    api_key: str = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
):
    """Get recommendations for a specific analysis."""
    try:
        logger.info(f"Fetching recommendations for analysis: {analysis_id}")
        
        analysis_id_str = str(analysis_id)
        
        # Check if analysis exists in cache
        if analysis_id_str not in analysis_cache:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found",
            )
        
        analysis_result = analysis_cache[analysis_id_str]
        
        # Check if analysis failed
        if analysis_result.get("status") == AnalysisStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis failed: {analysis_result.get('error', 'Unknown error')}",
            )
        
        # Check if analysis is still processing
        if analysis_result.get("status") != AnalysisStatus.COMPLETED:
            return RecommendationResponse(
                analysis_id=analysis_id_str,
                status="processing",
                recommendations=[],
                summary="Analysis is still processing",
                score=0,
            )
        
        # Convert analysis result to response format
        recommendations = analysis_result.get("recommendations", [])
        refactoring_suggestions = analysis_result.get("refactoring_suggestions", [])
        anti_pattern_fixes = analysis_result.get("anti_pattern_fixes", [])
        
        # Combine all recommendations
        all_recommendations = []
        
        # Add regular recommendations
        for rec in recommendations:
            all_recommendations.append({
                "type": rec.get("type", "improvement"),
                "severity": rec.get("severity", "info"),
                "message": rec.get("message", ""),
                "suggestion": rec.get("suggestion", ""),
                "line_start": rec.get("line_start", 1),
                "line_end": rec.get("line_end", 1),
                "confidence": rec.get("confidence", 0.5)
            })
        
        # Add refactoring suggestions
        for ref in refactoring_suggestions:
            all_recommendations.append({
                "type": "refactoring",
                "severity": "suggestion",
                "message": ref.get("description", ""),
                "suggestion": f"Refactor: {ref.get('refactor_type', 'unknown')}",
                "line_start": 1,
                "line_end": 1,
                "confidence": ref.get("confidence", 0.5)
            })
        
        # Add anti-pattern fixes
        for fix in anti_pattern_fixes:
            all_recommendations.append({
                "type": "anti_pattern",
                "severity": "warning",
                "message": fix.get("fix_description", ""),
                "suggestion": f"Fix {fix.get('pattern_type', 'anti-pattern')}",
                "line_start": 1,
                "line_end": 1,
                "confidence": fix.get("confidence", 0.5)
            })
        
        return RecommendationResponse(
            analysis_id=analysis_id_str,
            status="completed",
            recommendations=all_recommendations,
            summary=analysis_result.get("summary", "Analysis completed"),
            score=analysis_result.get("overall_score", 100),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching recommendations: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch recommendations",
        )


@router.get(
    "/repositories",
    response_model=List[RepositoryResponse],
    summary="List indexed repositories",
    description="Get a list of all indexed repositories",
)
async def list_repositories(
    skip: int = 0,
    limit: int = 20,
    status_filter: Optional[str] = None,
    api_key: str = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
):
    """List all indexed repositories."""
    try:
        logger.info(f"Listing repositories (skip={skip}, limit={limit}, status={status_filter})")
        
        # Build query
        from sqlalchemy import select
        query = select(Repository).offset(skip).limit(limit)
        
        # Apply status filter if provided
        if status_filter:
            try:
                status_enum = RepositoryStatus(status_filter)
                query = query.where(Repository.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status filter: {status_filter}",
                )
        
        # Execute query
        result = await db.execute(query)
        repositories = result.scalars().all()
        
        # Convert to response format
        response_repos = []
        for repo in repositories:
            # Get additional data from cache if available
            cache_data = indexing_cache.get(str(repo.id), {})
            
            # Calculate indexed files from total_files if completed
            indexed_files = repo.total_files if repo.status == RepositoryStatus.COMPLETED else cache_data.get("analyzed_files", 0)
            
            response_repos.append(RepositoryResponse(
                repository_id=str(repo.id),
                name=repo.name,
                path=repo.url or f"Unknown path for {repo.name}",
                status=repo.status.value,
                indexed_files=indexed_files,
                total_files=repo.total_files,
                message=f"Repository {repo.status.value}",
                last_indexed_at=repo.last_indexed_at.isoformat() if repo.last_indexed_at else None,
                created_at=repo.created_at.isoformat() if repo.created_at else None,
            ))
        
        logger.info(f"Found {len(response_repos)} repositories")
        return response_repos
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing repositories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list repositories",
        )


# In-memory cache for analysis and indexing results
# In production, use Redis or database storage
analysis_cache: Dict[str, Dict] = {}
indexing_cache: Dict[str, Dict] = {}

# WebSocket endpoint for real-time indexing progress
@router.websocket("/ws/index/{repository_id}")
async def websocket_indexing_progress(
    websocket: WebSocket,
    repository_id: str,
    api_key: Optional[str] = None,
):
    """WebSocket endpoint for real-time repository indexing progress."""
    # Validate API key from query params or headers
    if not api_key:
        api_key = websocket.query_params.get("api_key")
    
    if not api_key:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    # Basic API key validation (in production, use proper validation)
    if not api_key.startswith("sk-"):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    
    await websocket.accept()
    logger.info(f"WebSocket connection established for repository: {repository_id}")
    
    try:
        # Send initial status
        cache_data = indexing_cache.get(repository_id, {})
        
        await websocket.send_json({
            "type": "status",
            "repository_id": repository_id,
            "status": cache_data.get("status", "unknown"),
            "indexed_files": cache_data.get("analyzed_files", 0),
            "total_files": cache_data.get("total_files", 0),
            "current_file": None,
            "percentage": 0.0,
            "message": cache_data.get("error", "Repository indexing status"),
        })
        
        # Keep connection alive and periodically send updates
        while True:
            try:
                # Wait for client message or timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
                # Handle any client messages if needed
                logger.debug(f"Received WebSocket message: {data}")
            except asyncio.TimeoutError:
                # Send periodic status updates
                current_data = indexing_cache.get(repository_id, {})
                
                await websocket.send_json({
                    "type": "heartbeat",
                    "repository_id": repository_id,
                    "status": current_data.get("status", "unknown"),
                    "indexed_files": current_data.get("analyzed_files", 0),
                    "total_files": current_data.get("total_files", 0),
                    "timestamp": datetime.utcnow().isoformat(),
                })
                
                # If indexing is complete, close connection
                if current_data.get("status") in ["completed", "failed"]:
                    await websocket.send_json({
                        "type": "complete",
                        "repository_id": repository_id,
                        "status": current_data.get("status"),
                        "final_message": current_data.get("error", "Indexing completed"),
                    })
                    break
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for repository: {repository_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass


# Additional utility endpoints
@router.get(
    "/analysis/{analysis_id}/status",
    summary="Check analysis status",
    description="Get the current status of a running analysis",
)
async def get_analysis_status(
    analysis_id: UUID,
    api_key: str = Depends(get_api_key),
):
    """Get current status of an analysis."""
    try:
        analysis_id_str = str(analysis_id)
        
        if analysis_id_str not in analysis_cache:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found",
            )
        
        analysis_result = analysis_cache[analysis_id_str]
        
        return {
            "analysis_id": analysis_id_str,
            "status": analysis_result.get("status", "unknown"),
            "message": analysis_result.get("error", "Analysis in progress"),
            "progress_percentage": 100 if analysis_result.get("status") == AnalysisStatus.COMPLETED else 50,
            "estimated_completion": None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting analysis status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get analysis status",
        )


@router.get(
    "/repositories/{repository_id}/status",
    summary="Check repository indexing status",
    description="Get the current status of repository indexing",
)
async def get_repository_status(
    repository_id: UUID,
    api_key: str = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
):
    """Get current status of repository indexing."""
    try:
        repo_repo = RepositoryRepo(db)
        repository = await repo_repo.get_by_id(repository_id)
        
        if not repository:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Repository not found",
            )
        
        # Get additional data from cache
        cache_data = indexing_cache.get(str(repository_id), {})
        
        return {
            "repository_id": str(repository.id),
            "name": repository.name,
            "status": repository.status.value,
            "total_files": repository.total_files,
            "indexed_files": cache_data.get("analyzed_files", 0),
            "failed_files": cache_data.get("failed_files", 0),
            "progress_percentage": (
                (cache_data.get("analyzed_files", 0) / repository.total_files * 100)
                if repository.total_files > 0 else 0
            ),
            "last_indexed_at": repository.last_indexed_at.isoformat() if repository.last_indexed_at else None,
            "processing_time_seconds": cache_data.get("processing_time_seconds", 0),
            "error_message": cache_data.get("error") if repository.status == RepositoryStatus.FAILED else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting repository status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get repository status",
        )


@router.post(
    "/repositories/{repository_id}/search",
    response_model=SearchResponse,
    summary="Search within specific repository",
    description="Search for code patterns within a specific repository",
)
async def search_in_repository(
    repository_id: UUID,
    request: SearchRequest,
    api_key: str = Depends(get_api_key),
    db: AsyncSession = Depends(get_db_session),
    limiter=Depends(get_rate_limiter),
):
    """Search for patterns within a specific repository."""
    start_time = datetime.utcnow()
    
    try:
        # Validate repository exists
        repo_repo = RepositoryRepo(db)
        repository = await repo_repo.get_by_id(repository_id)
        
        if not repository:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Repository not found",
            )
        
        if repository.status != RepositoryStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Repository indexing not completed. Status: {repository.status.value}",
            )
        
        # Initialize search service
        search_manager = SearchServiceManager()
        
        # Perform repository-specific search
        results = await search_manager.search_repository_patterns(
            repository_id=repository_id,
            pattern_query=request.query,
            language=request.language,
            limit=request.limit or 20
        )
        
        # Calculate search time
        search_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        search_results = results.get("results", [])
        
        return SearchResponse(
            results=search_results,
            total_count=len(search_results),
            query=request.query,
            search_time_ms=search_time_ms,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching in repository: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search in repository",
        )


@router.get(
    "/health",
    summary="Service health check",
    description="Check the health status of all service components",
)
async def health_check():
    """Get health status of all service components."""
    try:
        # Check recommendation service health
        recommendation_service = RecommendationService()
        rec_health = await recommendation_service.health_check()
        
        # Check search service health
        search_manager = SearchServiceManager()
        search_health = await search_manager.get_service_health()
        
        # Check embedding service health
        embedding_manager = get_embedding_manager()
        embedding_health = embedding_manager.health_check()
        
        # Overall health status
        all_healthy = all([
            rec_health.get("status") == "healthy",
            search_health.get("status") == "healthy",
            embedding_health.get("primary", False)
        ])
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "recommendation_service": rec_health,
                "search_service": search_health,
                "embedding_service": {
                    "status": "healthy" if embedding_health.get("primary") else "degraded",
                    "primary_healthy": embedding_health.get("primary", False),
                    "fallback_healthy": embedding_health.get("fallback"),
                }
            },
            "cache_stats": {
                "active_analyses": len(analysis_cache),
                "active_indexing_jobs": len(indexing_cache)
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }