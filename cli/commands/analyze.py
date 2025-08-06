"""Analyze command for code analysis."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console

from cli.utils.config import CLIConfig
from cli.utils.logging import get_cli_logger
from cli.utils.output import OutputFormatter
from cli.utils.progress import create_progress_callback
from src.services.analysis import AnalysisService


@click.command(name="analyze")
@click.argument("target", type=click.Path(exists=True), required=True)
@click.option(
    "--recursive",
    "-r",
    is_flag=True,
    help="Recursively analyze directories"
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
    "--language",
    "-l",
    default="python",
    help="Programming language to analyze"
)
@click.option(
    "--metrics-only",
    is_flag=True,
    help="Only calculate metrics, skip chunking"
)
@click.option(
    "--detailed",
    "-d",
    is_flag=True,
    help="Show detailed analysis including function-level metrics"
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(),
    help="Save results to file"
)
@click.pass_context
def analyze_cmd(
    ctx: click.Context,
    target: str,
    recursive: bool,
    include_pattern: tuple,
    exclude_pattern: tuple,
    language: str,
    metrics_only: bool,
    detailed: bool,
    output_file: Optional[str]
):
    """Analyze code files or directories for quality metrics and patterns.
    
    TARGET can be a single file or directory path.
    
    Examples:
        vibecode analyze ./my_file.py
        vibecode analyze ./my_project --recursive
        vibecode analyze ./src -i "*.py" -e "*test*"
        vibecode analyze ./app.py --detailed --output-file report.json
    """
    # Get CLI context
    config: CLIConfig = ctx.obj["config"]
    quiet: bool = ctx.obj["quiet"]
    verbose: bool = ctx.obj["verbose"]
    output_json: bool = ctx.obj["output_json"]
    
    logger = get_cli_logger(__name__, quiet)
    console = Console()
    
    # Determine output format
    format_type = "json" if output_json else config.output_format
    formatter = OutputFormatter(console, format_type)
    
    target_path = Path(target)
    
    try:
        if target_path.is_file():
            # Analyze single file
            logger.step(f"Analyzing file: {target_path}")
            result = asyncio.run(_analyze_single_file(
                target_path, config, logger, quiet
            ))
            
        elif target_path.is_dir() and recursive:
            # Analyze directory recursively
            logger.step(f"Analyzing directory: {target_path}")
            
            # Build patterns
            include_patterns = list(include_pattern) or config.include_patterns
            exclude_patterns = list(exclude_pattern) or config.exclude_patterns
            
            result = asyncio.run(_analyze_repository(
                target_path, include_patterns, exclude_patterns, config, logger, quiet
            ))
            
        elif target_path.is_dir():
            # Analyze directory non-recursively
            logger.step(f"Analyzing directory (non-recursive): {target_path}")
            
            # Find Python files in directory
            python_files = list(target_path.glob("*.py"))
            if not python_files:
                logger.error(f"No Python files found in {target_path}")
                sys.exit(1)
            
            # Analyze each file
            result = asyncio.run(_analyze_directory_files(
                python_files, config, logger, quiet
            ))
            
        else:
            logger.error(f"Invalid target: {target_path}")
            sys.exit(1)
        
        # Format and display results
        if detailed and isinstance(result, dict):
            formatter.output(result, "Analysis Results")
        else:
            # Show summary
            _show_analysis_summary(result, formatter, logger)
        
        # Save to file if requested
        if output_file:
            _save_results(result, output_file, logger)
            
        logger.success("Analysis completed successfully")
        
    except KeyboardInterrupt:
        logger.error("Analysis cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


async def _analyze_single_file(
    file_path: Path,
    config: CLIConfig,
    logger,
    quiet: bool,
    show_progress: bool = True
) -> dict:
    """Analyze a single file."""
    analysis_service = AnalysisService()
    
    if show_progress and not quiet:
        with create_progress_callback(f"Analyzing {file_path.name}") as callback:
            result = await analysis_service.analyze_file(file_path, callback)
    else:
        result = await analysis_service.analyze_file(file_path)
    
    # Convert FileAnalysis to dict
    return {
        "file_path": result.file_path,
        "language": result.language,
        "status": result.status,
        "functions": result.functions,
        "classes": result.classes,
        "imports": result.imports,
        "metrics": result.metrics,
        "chunks": result.chunks,
        "analysis_time_ms": result.analysis_time_ms,
        "error_message": result.error_message,
        "warnings": result.warnings,
    }


async def _analyze_repository(
    repo_path: Path,
    include_patterns: List[str],
    exclude_patterns: List[str],
    config: CLIConfig,
    logger,
    quiet: bool
) -> dict:
    """Analyze an entire repository."""
    analysis_service = AnalysisService()
    
    if not quiet:
        with create_progress_callback("Analyzing repository") as callback:
            result = await analysis_service.analyze_repository(
                repo_path, callback, include_patterns, exclude_patterns
            )
    else:
        result = await analysis_service.analyze_repository(
            repo_path, None, include_patterns, exclude_patterns
        )
    
    # Convert RepositoryAnalysis to dict
    return {
        "repository_path": result.repository_path,
        "total_files": result.total_files,
        "analyzed_files": result.analyzed_files,
        "skipped_files": result.skipped_files,
        "failed_files": result.failed_files,
        "total_lines": result.total_lines,
        "average_complexity": result.average_complexity,
        "total_functions": result.total_functions,
        "total_classes": result.total_classes,
        "file_analyses": [
            {
                "file_path": fa.file_path,
                "language": fa.language,
                "status": fa.status,
                "functions": fa.functions,
                "classes": fa.classes,
                "imports": fa.imports,
                "metrics": fa.metrics,
                "chunks": fa.chunks,
                "analysis_time_ms": fa.analysis_time_ms,
                "error_message": fa.error_message,
                "warnings": fa.warnings,
            }
            for fa in result.file_analyses
        ],
        "errors": result.errors,
        "total_time_seconds": result.total_time_seconds,
        "started_at": result.started_at.isoformat() if result.started_at else None,
        "completed_at": result.completed_at.isoformat() if result.completed_at else None,
    }


async def _analyze_directory_files(
    python_files: List[Path], 
    config: CLIConfig, 
    logger, 
    quiet: bool
) -> dict:
    """Analyze files in a directory."""
    results = []
    for file_path in python_files:
        file_result = await _analyze_single_file(
            file_path, config, logger, quiet, show_progress=False
        )
        results.append(file_result)
    
    # Create summary result
    return {
        "type": "directory_analysis",
        "directory_path": str(python_files[0].parent) if python_files else "",
        "total_files": len(results),
        "file_analyses": results,
        "analyzed_files": sum(1 for r in results if r.get("status") == "success"),
        "failed_files": sum(1 for r in results if r.get("status") == "error"),
    }


def _show_analysis_summary(result: dict, formatter: OutputFormatter, logger) -> None:
    """Show analysis summary."""
    if isinstance(result, dict):
        if "repository_path" in result:
            # Repository analysis summary
            summary = {
                "Files Analyzed": result.get("analyzed_files", 0),
                "Files Failed": result.get("failed_files", 0),
                "Total Lines": result.get("total_lines", 0),
                "Average Complexity": f"{result.get('average_complexity', 0):.2f}",
                "Total Functions": result.get("total_functions", 0),
                "Total Classes": result.get("total_classes", 0),
                "Analysis Time": f"{result.get('total_time_seconds', 0):.2f}s",
            }
            
            formatter.output(summary, "Repository Analysis Summary")
            
            # Show any errors
            errors = result.get("errors", [])
            if errors:
                logger.warning(f"Encountered {len(errors)} errors during analysis")
                for error in errors[:5]:  # Show first 5 errors
                    logger.error(f"  {error.get('file', 'Unknown')}: {error.get('error', 'Unknown error')}")
                
                if len(errors) > 5:
                    logger.info(f"  ... and {len(errors) - 5} more errors")
        
        elif "file_path" in result:
            # Single file analysis summary
            metrics = result.get("metrics", {})
            summary = {
                "File": Path(result.get("file_path", "")).name,
                "Status": result.get("status", "unknown"),
                "Lines of Code": metrics.get("lines_of_code", 0),
                "Cyclomatic Complexity": metrics.get("cyclomatic_complexity", 0),
                "Maintainability Index": f"{metrics.get('maintainability_index', 0):.1f}",
                "Risk Level": metrics.get("risk_level", "Unknown"),
                "Functions": len(result.get("functions", [])),
                "Classes": len(result.get("classes", [])),
                "Analysis Time": f"{result.get('analysis_time_ms', 0):.1f}ms",
            }
            
            formatter.output(summary, "File Analysis Summary")
            
            # Show warnings if any
            warnings = result.get("warnings", [])
            if warnings:
                logger.warning("Code quality warnings:")
                for warning in warnings:
                    logger.warning(f"  {warning}")


def _save_results(result: dict, output_file: str, logger) -> None:
    """Save analysis results to file."""
    import json
    
    output_path = Path(output_file)
    
    try:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.success(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")