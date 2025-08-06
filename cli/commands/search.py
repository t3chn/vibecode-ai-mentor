"""Search command for finding similar code patterns."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from cli.utils.config import CLIConfig
from cli.utils.logging import get_cli_logger
from cli.utils.output import OutputFormatter
from src.search.service import SearchServiceManager


@click.command(name="search")
@click.argument("query", required=True)
@click.option(
    "--language",
    "-l",
    default="python",
    help="Programming language to search in"
)
@click.option(
    "--limit",
    "-n",
    default=10,
    help="Maximum number of results to return"
)
@click.option(
    "--similarity-threshold",
    "-t",
    type=float,
    default=0.7,
    help="Minimum similarity threshold (0.0-1.0)"
)
@click.option(
    "--repository",
    "-r",
    help="Filter results by repository name"
)
@click.option(
    "--file-pattern",
    "-f",
    help="Filter results by file pattern (e.g., '*test*')"
)
@click.option(
    "--chunk-type",
    "-c",
    type=click.Choice(["function", "class", "block", "all"]),
    default="all",
    help="Type of code chunks to search"
)
@click.option(
    "--show-code",
    "-s",
    is_flag=True,
    help="Show code snippets in results"
)
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Show detailed match information"
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    help="Save results to file"
)
@click.pass_context
def search_cmd(
    ctx: click.Context,
    query: str,
    language: str,
    limit: int,
    similarity_threshold: float,
    repository: Optional[str],
    file_pattern: Optional[str],
    chunk_type: str,
    show_code: bool,
    detailed: bool,
    output_file: Optional[str]
):
    """Search for similar code patterns using vector similarity.
    
    QUERY can be natural language description or code snippet.
    
    Examples:
        vibecode search "async function with error handling"
        vibecode search "def calculate_total" --language python
        vibecode search "class with multiple methods" -n 5 -t 0.8
        vibecode search "database connection" --repository myapp
        vibecode search "unit test" --file-pattern "*test*" --show-code
    """
    # Get CLI context
    config: CLIConfig = ctx.obj["config"]
    quiet: bool = ctx.obj["quiet"]
    verbose: bool = ctx.obj["verbose"]
    output_json: bool = ctx.obj["output_json"]
    
    logger = get_cli_logger(__name__, quiet)
    console = Console()
    
    # Check database configuration
    if not config.has_database_config:
        logger.error("Database configuration missing. Run 'vibecode setup' first.")
        sys.exit(1)
    
    # Determine output format
    format_type = "json" if output_json else config.output_format
    formatter = OutputFormatter(console, format_type)
    
    # Validate parameters
    if not 0.0 <= similarity_threshold <= 1.0:
        logger.error("Similarity threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    if limit <= 0:
        logger.error("Limit must be positive")
        sys.exit(1)
    
    try:
        # Set environment variables for database connection
        import os
        env_vars = config.to_env_dict()
        for key, value in env_vars.items():
            os.environ[key] = value
        
        logger.step(f"Searching for: '{query}'")
        
        # Perform search
        result = await _perform_search(
            query, language, limit, similarity_threshold,
            repository, file_pattern, chunk_type, config, logger
        )
        
        # Format and display results
        if detailed:
            formatter.output(result, "Search Results")
        else:
            _show_search_summary(result, formatter, logger, show_code)
        
        # Save to file if requested
        if output_file:
            _save_results(result, output_file, logger)
        
        # Show summary
        total_results = len(result.get("results", []))
        if total_results > 0:
            logger.success(f"Found {total_results} matching code snippets")
        else:
            logger.warning("No matching code snippets found")
        
    except KeyboardInterrupt:
        logger.error("Search cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Search failed: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


async def _perform_search(
    query: str,
    language: str,
    limit: int,
    similarity_threshold: float,
    repository: Optional[str],
    file_pattern: Optional[str],
    chunk_type: str,
    config: CLIConfig,
    logger
) -> dict:
    """Perform the actual search operation."""
    
    search_manager = SearchServiceManager()
    
    # Determine if query looks like code or natural language
    is_code_query = any(keyword in query.lower() for keyword in [
        "def ", "class ", "import ", "from ", "async ", "await ",
        "{", "}", "(", ")", "=", "==", "!=", "if ", "for ", "while "
    ])
    
    if is_code_query:
        logger.step("Searching for similar code patterns...")
        
        # Use code similarity search
        result = await search_manager.find_similar_code(
            code_snippet=query,
            language=language,
            threshold=similarity_threshold,
            limit=limit,
            exclude_repository=None  # Could add exclude option
        )
        
        # Reformat for consistent output
        search_results = result.get("similar_examples", [])
        recommendations = result.get("recommendations", [])
        
    else:
        logger.step("Searching with natural language query...")
        
        # Use text-based search
        search_results = await search_manager.quick_search(
            query=query,
            language=language,
            limit=limit,
            similarity_threshold=similarity_threshold,
            repository_filter=repository
        )
        
        recommendations = []
    
    # Apply additional filters
    if file_pattern:
        import fnmatch
        search_results = [
            result for result in search_results
            if fnmatch.fnmatch(result.get("file_path", ""), f"*{file_pattern}*")
        ]
    
    if chunk_type != "all":
        search_results = [
            result for result in search_results
            if result.get("chunk_type", "").lower() == chunk_type.lower()
        ]
    
    if repository:
        search_results = [
            result for result in search_results
            if repository.lower() in result.get("repository_name", "").lower()
        ]
    
    # Calculate statistics
    if search_results:
        avg_similarity = sum(
            result.get("similarity_score", 0) for result in search_results
        ) / len(search_results)
        
        # Group by file
        files = set(result.get("file_path", "") for result in search_results)
        
        # Group by repository
        repositories = set(result.get("repository_name", "") for result in search_results)
        
        # Group by chunk type
        chunk_types = {}
        for result in search_results:
            chunk_type = result.get("chunk_type", "unknown")
            chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
    
    else:
        avg_similarity = 0
        files = set()
        repositories = set()
        chunk_types = {}
    
    return {
        "query": query,
        "query_type": "code" if is_code_query else "text",
        "language": language,
        "results": search_results,
        "recommendations": recommendations,
        "statistics": {
            "total_matches": len(search_results),
            "average_similarity": avg_similarity,
            "unique_files": len(files),
            "unique_repositories": len(repositories),
            "chunk_type_distribution": chunk_types,
            "similarity_threshold": similarity_threshold,
        },
        "filters": {
            "repository": repository,
            "file_pattern": file_pattern,
            "chunk_type": chunk_type,
            "limit": limit,
        }
    }


def _show_search_summary(result: dict, formatter: OutputFormatter, logger, show_code: bool) -> None:
    """Show search results summary."""
    
    results = result.get("results", [])
    stats = result.get("statistics", {})
    recommendations = result.get("recommendations", [])
    
    if not results:
        logger.warning("No matching results found")
        
        # Show search parameters
        query_type = result.get("query_type", "unknown")
        logger.info(f"Query type: {query_type}")
        logger.info(f"Language: {result.get('language', 'unknown')}")
        logger.info(f"Similarity threshold: {stats.get('similarity_threshold', 0):.2f}")
        
        return
    
    # Show summary statistics
    summary = {
        "Total Matches": stats.get("total_matches", 0),
        "Average Similarity": f"{stats.get('average_similarity', 0):.1%}",
        "Unique Files": stats.get("unique_files", 0),
        "Unique Repositories": stats.get("unique_repositories", 0),
        "Query Type": result.get("query_type", "unknown").title(),
    }
    
    formatter.output(summary, "Search Summary")
    
    # Show top results
    from rich.table import Table
    from rich.panel import Panel
    
    results_table = Table(title="Top Matches")
    results_table.add_column("File", style="cyan", max_width=30)
    results_table.add_column("Similarity", justify="right", style="green")
    results_table.add_column("Lines", justify="center")
    results_table.add_column("Type", style="blue")
    results_table.add_column("Repository", style="dim", max_width=20)
    
    for result_item in results[:10]:  # Show top 10
        file_path = result_item.get("file_path", "")
        file_name = Path(file_path).name if file_path else "Unknown"
        
        similarity = result_item.get("similarity_score", 0)
        # Convert distance to similarity if needed
        if similarity > 1.0:
            similarity = 1.0 - similarity
        
        results_table.add_row(
            file_name,
            f"{similarity:.1%}",
            f"{result_item.get('start_line', 0)}-{result_item.get('end_line', 0)}",
            result_item.get("chunk_type", "code"),
            result_item.get("repository_name", "Unknown")
        )
    
    formatter.console.print(results_table)
    
    # Show code snippets if requested
    if show_code:
        formatter.console.print("\n[bold blue]Code Snippets:[/bold blue]")
        
        for i, result_item in enumerate(results[:3], 1):  # Show top 3 code snippets
            content = result_item.get("content", "")
            if content:
                # Truncate long content
                if len(content) > 500:
                    content = content[:500] + "..."
                
                file_path = result_item.get("file_path", "")
                file_name = Path(file_path).name if file_path else "Unknown"
                
                similarity = result_item.get("similarity_score", 0)
                if similarity > 1.0:
                    similarity = 1.0 - similarity
                
                from rich.syntax import Syntax
                
                code_panel = Panel(
                    Syntax(content, result.get("language", "python"), theme="monokai"),
                    title=f"Match {i}: {file_name} ({similarity:.1%} similar)"
                )
                formatter.console.print(code_panel)
    
    # Show recommendations
    if recommendations:
        formatter.console.print("\n[bold green]AI Recommendations:[/bold green]")
        for i, rec in enumerate(recommendations[:3], 1):
            formatter.console.print(f"{i}. {rec}")
    
    # Show chunk type distribution
    chunk_dist = stats.get("chunk_type_distribution", {})
    if chunk_dist and len(chunk_dist) > 1:
        formatter.console.print("\n[dim]Match distribution by type:[/dim]")
        for chunk_type, count in sorted(chunk_dist.items()):
            formatter.console.print(f"  {chunk_type}: {count}")


def _save_results(result: dict, output_file: str, logger) -> None:
    """Save search results to file."""
    import json
    
    output_path = Path(output_file)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.success(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")