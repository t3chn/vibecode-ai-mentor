"""Index command for repository indexing and embedding generation."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import click
from rich.console import Console

from cli.utils.config import CLIConfig
from cli.utils.logging import get_cli_logger
from cli.utils.output import OutputFormatter
from cli.utils.progress import create_progress_callback
from src.services.analysis import AnalysisService
from src.embeddings.factory import EmbeddingServiceFactory
from src.db.connection import get_async_session
from src.db.repositories import CodeSnippetRepository, RepositoryInfoRepository


@click.command(name="index")
@click.argument("repository_path", type=click.Path(exists=True), required=True)
@click.option(
    "--name",
    "-n",
    help="Repository name (default: directory name)"
)
@click.option(
    "--include-pattern",
    "-i",
    multiple=True,
    help="File patterns to include (e.g., '*.py')"
)
@click.option(
    "--exclude-pattern",
    "-e", 
    multiple=True,
    help="File patterns to exclude (e.g., '*test*')"
)
@click.option(
    "--batch-size",
    "-b",
    default=50,
    help="Batch size for embedding generation"
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force re-indexing if repository already exists"
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be indexed without actually doing it"
)
@click.option(
    "--skip-embeddings",
    is_flag=True,
    help="Index code without generating embeddings"
)
@click.pass_context
def index_cmd(
    ctx: click.Context,
    repository_path: str,
    name: Optional[str],
    include_pattern: tuple,
    exclude_pattern: tuple,
    batch_size: int,
    force: bool,
    dry_run: bool,
    skip_embeddings: bool
):
    """Index a repository for vector search and analysis.
    
    This command analyzes all code in a repository, generates embeddings,
    and stores them in the TiDB database for similarity search.
    
    REPOSITORY_PATH is the path to the repository to index.
    
    Examples:
        vibecode index ./my-project
        vibecode index ./src --name "MyApp Core"
        vibecode index ./repo -i "*.py" -e "*test*" --batch-size 100
        vibecode index ./project --dry-run
    """
    # Get CLI context
    config: CLIConfig = ctx.obj["config"]
    quiet: bool = ctx.obj["quiet"]
    verbose: bool = ctx.obj["verbose"]
    output_json: bool = ctx.obj["output_json"]
    
    logger = get_cli_logger(__name__, quiet)
    console = Console()
    
    # Check database configuration
    if not config.has_database_config and not dry_run:
        logger.error("Database configuration missing. Run 'vibecode setup' first.")
        sys.exit(1)
    
    # Check API keys for embeddings
    if not config.has_api_keys and not skip_embeddings and not dry_run:
        logger.error("API keys missing for embedding generation. Configure GEMINI_API_KEY or OPENAI_API_KEY.")
        sys.exit(1)
    
    # Determine output format
    format_type = "json" if output_json else config.output_format
    formatter = OutputFormatter(console, format_type)
    
    repo_path = Path(repository_path)
    repo_name = name or repo_path.name
    
    try:
        # Build patterns
        include_patterns = list(include_pattern) or config.include_patterns
        exclude_patterns = list(exclude_pattern) or config.exclude_patterns
        
        logger.step(f"Starting indexing of repository: {repo_name}")
        
        if dry_run:
            result = await _dry_run_index(
                repo_path, repo_name, include_patterns, exclude_patterns, config, logger
            )
        else:
            result = await _index_repository(
                repo_path, repo_name, include_patterns, exclude_patterns,
                batch_size, force, skip_embeddings, config, logger, quiet
            )
        
        # Display results
        formatter.output(result, "Indexing Results")
        
        if not dry_run:
            logger.success("Repository indexing completed successfully")
        
    except KeyboardInterrupt:
        logger.error("Indexing cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


async def _dry_run_index(
    repo_path: Path,
    repo_name: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
    config: CLIConfig,
    logger
) -> dict:
    """Perform dry run to show what would be indexed."""
    logger.step("Performing dry run analysis...")
    
    analysis_service = AnalysisService()
    
    # Analyze repository without progress callback for dry run
    result = await analysis_service.analyze_repository(
        repo_path, None, include_patterns, exclude_patterns
    )
    
    # Calculate what would be indexed
    total_chunks = sum(len(fa.chunks) for fa in result.file_analyses if fa.status == "success")
    total_functions = sum(len(fa.functions) for fa in result.file_analyses if fa.status == "success")
    total_classes = sum(len(fa.classes) for fa in result.file_analyses if fa.status == "success")
    
    return {
        "dry_run": True,
        "repository_name": repo_name,
        "repository_path": str(repo_path),
        "would_analyze": {
            "total_files": result.total_files,
            "python_files": result.analyzed_files,
            "total_chunks": total_chunks,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "total_lines": result.total_lines,
        },
        "patterns": {
            "include": include_patterns,
            "exclude": exclude_patterns,
        },
        "estimated_time_seconds": result.total_time_seconds * 3,  # Rough estimate with embeddings
        "skipped_files": result.skipped_files,
        "failed_files": result.failed_files,
    }


async def _index_repository(
    repo_path: Path,
    repo_name: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
    batch_size: int,
    force: bool,
    skip_embeddings: bool,
    config: CLIConfig,
    logger,
    quiet: bool
) -> dict:
    """Index repository with full analysis and embedding generation."""
    
    # Set environment variables for database connection
    import os
    env_vars = config.to_env_dict()
    for key, value in env_vars.items():
        os.environ[key] = value
    
    async with await get_async_session() as session:
        repo_repository = RepositoryInfoRepository(session)
        snippet_repository = CodeSnippetRepository(session)
        
        # Check if repository already exists
        existing_repo = await repo_repository.get_by_name_and_path(repo_name, str(repo_path))
        if existing_repo and not force:
            logger.error(f"Repository '{repo_name}' already indexed. Use --force to re-index.")
            sys.exit(1)
        
        # Delete existing repository if forcing re-index
        if existing_repo and force:
            logger.step("Removing existing repository data...")
            await repo_repository.delete(existing_repo.id)
            await session.commit()
        
        # Step 1: Analyze repository
        logger.step("Analyzing repository structure...")
        
        analysis_service = AnalysisService()
        
        if not quiet:
            with create_progress_callback("Analyzing repository") as callback:
                analysis_result = await analysis_service.analyze_repository(
                    repo_path, callback, include_patterns, exclude_patterns
                )
        else:
            analysis_result = await analysis_service.analyze_repository(
                repo_path, None, include_patterns, exclude_patterns
            )
        
        # Step 2: Create repository record
        logger.step("Creating repository record...")
        
        repo_id = str(uuid4())
        await repo_repository.create({
            "id": repo_id,
            "name": repo_name,
            "path": str(repo_path),
            "description": f"Auto-indexed repository: {repo_name}",
            "language": "python",
            "total_files": analysis_result.analyzed_files,
            "total_lines": analysis_result.total_lines,
            "indexed_at": analysis_result.completed_at or analysis_result.started_at
        })
        
        # Step 3: Process code chunks and generate embeddings
        total_chunks = 0
        stored_chunks = 0
        embedding_errors = 0
        
        if not skip_embeddings:
            logger.step("Generating embeddings...")
            
            # Initialize embedding service
            embedding_service = EmbeddingServiceFactory.create_service(
                config.gemini_api_key, config.openai_api_key
            )
            
            # Process files in batches
            all_chunks = []
            for file_analysis in analysis_result.file_analyses:
                if file_analysis.status != "success":
                    continue
                
                for chunk_data in file_analysis.chunks:
                    chunk_record = {
                        "id": str(uuid4()),
                        "repository_id": repo_id,
                        "file_path": file_analysis.file_path,
                        "content": chunk_data["content"],
                        "start_line": chunk_data["start_line"],
                        "end_line": chunk_data["end_line"],
                        "chunk_type": chunk_data["chunk_type"],
                        "language": file_analysis.language,
                        "token_count": chunk_data["token_count"],
                        "metadata": chunk_data["metadata"] or {}
                    }
                    all_chunks.append(chunk_record)
                    total_chunks += 1
            
            # Generate embeddings in batches
            for i in range(0, len(all_chunks), batch_size):
                batch = all_chunks[i:i + batch_size]
                
                if not quiet:
                    logger.progress(f"Processing batch {i // batch_size + 1}/{(len(all_chunks) + batch_size - 1) // batch_size}")
                
                try:
                    # Generate embeddings for batch
                    texts = [chunk["content"] for chunk in batch]
                    embeddings = await embedding_service.generate_embeddings(texts)
                    
                    # Store chunks with embeddings
                    for chunk, embedding in zip(batch, embeddings):
                        chunk["embedding"] = embedding
                        await snippet_repository.create(chunk)
                        stored_chunks += 1
                    
                    await session.commit()
                    
                except Exception as e:
                    logger.warning(f"Failed to process batch: {e}")
                    embedding_errors += len(batch)
                    # Store chunks without embeddings
                    for chunk in batch:
                        chunk["embedding"] = None
                        await snippet_repository.create(chunk)
                        stored_chunks += 1
                    
                    await session.commit()
        
        else:
            # Store chunks without embeddings
            logger.step("Storing code chunks...")
            
            for file_analysis in analysis_result.file_analyses:
                if file_analysis.status != "success":
                    continue
                
                for chunk_data in file_analysis.chunks:
                    chunk_record = {
                        "id": str(uuid4()),
                        "repository_id": repo_id,
                        "file_path": file_analysis.file_path,
                        "content": chunk_data["content"],
                        "start_line": chunk_data["start_line"],
                        "end_line": chunk_data["end_line"],
                        "chunk_type": chunk_data["chunk_type"],
                        "language": file_analysis.language,
                        "token_count": chunk_data["token_count"],
                        "metadata": chunk_data["metadata"] or {},
                        "embedding": None
                    }
                    await snippet_repository.create(chunk_record)
                    total_chunks += 1
                    stored_chunks += 1
            
            await session.commit()
        
        # Update repository with final stats
        await repo_repository.update(repo_id, {
            "total_chunks": stored_chunks,
            "last_updated": analysis_result.completed_at or analysis_result.started_at
        })
        await session.commit()
        
        return {
            "repository_id": repo_id,
            "repository_name": repo_name,
            "repository_path": str(repo_path),
            "analysis_stats": {
                "total_files": analysis_result.total_files,
                "analyzed_files": analysis_result.analyzed_files,
                "failed_files": analysis_result.failed_files,
                "total_lines": analysis_result.total_lines,
                "total_functions": analysis_result.total_functions,
                "total_classes": analysis_result.total_classes,
                "analysis_time_seconds": analysis_result.total_time_seconds,
            },
            "indexing_stats": {
                "total_chunks": total_chunks,
                "stored_chunks": stored_chunks,
                "embedding_errors": embedding_errors,
                "embeddings_generated": stored_chunks - embedding_errors if not skip_embeddings else 0,
                "skip_embeddings": skip_embeddings,
            },
            "patterns": {
                "include": include_patterns,
                "exclude": exclude_patterns,
            },
        }