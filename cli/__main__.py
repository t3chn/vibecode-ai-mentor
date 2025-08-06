"""Main CLI entry point for VibeCode AI Mentor.

This module provides the primary command-line interface with Click-based commands
for analyzing code, indexing repositories, searching patterns, and managing the system.
"""

import asyncio
import sys
from pathlib import Path

import click
from rich import print as rprint
from rich.console import Console

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cli.commands.analyze import analyze_cmd
from cli.commands.index import index_cmd
from cli.commands.search import search_cmd
from cli.commands.serve import serve_cmd, status_cmd
from cli.commands.setup import setup_cmd
from cli.utils.config import CLIConfig
from cli.utils.logging import setup_cli_logging

console = Console()


@click.group(name="vibecode")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file"
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output"
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress output except errors"
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output results in JSON format"
)
@click.version_option(version="0.1.0", prog_name="VibeCode AI Mentor")
@click.pass_context
def cli(ctx: click.Context, config: str, verbose: bool, quiet: bool, output_json: bool):
    """VibeCode AI Mentor - AI-powered code quality analysis tool.
    
    Analyze code quality, find similar patterns, and get AI-powered recommendations
    using TiDB Cloud Vector Search and advanced code analysis.
    
    Examples:
        vibecode analyze ./my_project.py
        vibecode index ./my_repository
        vibecode search "async function with error handling"
        vibecode serve --port 8080
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Load configuration
    try:
        cli_config = CLIConfig.load(config_path=config)
        ctx.obj["config"] = cli_config
    except Exception as e:
        if not quiet:
            rprint(f"[red]Warning: Failed to load configuration: {e}[/red]")
        ctx.obj["config"] = CLIConfig()
    
    # Store CLI options
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["output_json"] = output_json
    
    # Setup logging
    log_level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    setup_cli_logging(log_level)
    
    # Handle conflicting options
    if verbose and quiet:
        raise click.UsageError("Cannot use both --verbose and --quiet")


# Add commands
cli.add_command(analyze_cmd)
cli.add_command(index_cmd)
cli.add_command(search_cmd)
cli.add_command(serve_cmd)
cli.add_command(status_cmd)
cli.add_command(setup_cmd)


def main():
    """Main entry point for CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()