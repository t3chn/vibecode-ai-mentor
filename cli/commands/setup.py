"""Setup command for database initialization and configuration."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.panel import Panel

from cli.utils.config import CLIConfig
from cli.utils.logging import get_cli_logger


@click.command(name="setup")
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=True,
    help="Run interactive setup (default)"
)
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Run non-interactive setup using environment variables"
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(),
    help="Save configuration to specific file"
)
@click.option(
    "--skip-db-test",
    is_flag=True,
    help="Skip database connection test"
)
@click.option(
    "--create-tables",
    is_flag=True,
    help="Create database tables"
)
@click.pass_context
def setup_cmd(
    ctx: click.Context,
    interactive: bool,
    non_interactive: bool,
    config_file: Optional[str],
    skip_db_test: bool,
    create_tables: bool
):
    """Set up VibeCode AI Mentor configuration and database.
    
    This command helps you configure database connections, API keys,
    and initialize the database schema.
    
    Examples:
        vibecode setup                    # Interactive setup
        vibecode setup --non-interactive  # Use environment variables
        vibecode setup --create-tables    # Also create database tables
        vibecode setup -c ./config.yaml   # Save to specific config file
    """
    # Get CLI context
    config: CLIConfig = ctx.obj["config"]
    quiet: bool = ctx.obj["quiet"]
    verbose: bool = ctx.obj["verbose"]
    
    logger = get_cli_logger(__name__, quiet)
    console = Console()
    
    if non_interactive:
        interactive = False
    
    try:
        if interactive and not quiet:
            console.print(Panel.fit(
                "[bold blue]VibeCode AI Mentor Setup[/bold blue]\n\n"
                "This setup will help you configure:\n"
                "• TiDB database connection\n"
                "• API keys for embeddings (Gemini/OpenAI)\n"
                "• Application preferences\n"
                "• Database schema initialization",
                title="Welcome"
            ))
        
        # Step 1: Configure database connection
        logger.step("Configuring database connection...")
        if interactive:
            config = _interactive_database_setup(config, console)
        else:
            config = _load_database_from_env(config, logger)
        
        # Step 2: Configure API keys
        logger.step("Configuring API keys...")
        if interactive:
            config = _interactive_api_keys_setup(config, console)
        else:
            config = _load_api_keys_from_env(config, logger)
        
        # Step 3: Configure preferences
        if interactive:
            logger.step("Configuring preferences...")
            config = _interactive_preferences_setup(config, console)
        
        # Step 4: Test database connection
        if not skip_db_test:
            logger.step("Testing database connection...")
            await _test_database_connection(config, logger)
        
        # Step 5: Create tables if requested
        if create_tables:
            logger.step("Creating database tables...")
            await _create_database_tables(config, logger)
        
        # Step 6: Save configuration
        logger.step("Saving configuration...")
        config_path = config.save(config_file)
        logger.success(f"Configuration saved to {config_path}")
        
        # Show setup summary
        if not quiet:
            _show_setup_summary(config, console)
        
        logger.success("Setup completed successfully!")
        
        if not quiet:
            console.print("\n[green]Next steps:[/green]")
            console.print("1. Start the API server: [cyan]vibecode serve[/cyan]")
            console.print("2. Index a repository: [cyan]vibecode index ./my-project[/cyan]")
            console.print("3. Search for patterns: [cyan]vibecode search \"async function\"[/cyan]")
        
    except KeyboardInterrupt:
        logger.error("Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


def _interactive_database_setup(config: CLIConfig, console: Console) -> CLIConfig:
    """Interactive database configuration."""
    console.print("\n[bold cyan]Database Configuration[/bold cyan]")
    
    # TiDB host
    current_host = config.tidb_host or "gateway01.us-west-2.prod.aws.tidbcloud.com"
    config.tidb_host = Prompt.ask(
        "TiDB host",
        default=current_host
    )
    
    # TiDB port
    config.tidb_port = int(Prompt.ask(
        "TiDB port",
        default=str(config.tidb_port)
    ))
    
    # TiDB user
    config.tidb_user = Prompt.ask(
        "TiDB username",
        default=config.tidb_user or ""
    )
    
    # TiDB password
    config.tidb_password = Prompt.ask(
        "TiDB password",
        password=True,
        default=config.tidb_password or ""
    )
    
    # TiDB database
    config.tidb_database = Prompt.ask(
        "TiDB database name",
        default=config.tidb_database
    )
    
    return config


def _load_database_from_env(config: CLIConfig, logger) -> CLIConfig:
    """Load database configuration from environment variables."""
    required_vars = ["TIDB_HOST", "TIDB_USER", "TIDB_PASSWORD"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Set these variables or use --interactive mode")
        sys.exit(1)
    
    # Load from environment (already handled by CLIConfig)
    logger.success("Database configuration loaded from environment")
    return config


def _interactive_api_keys_setup(config: CLIConfig, console: Console) -> CLIConfig:
    """Interactive API keys configuration."""
    console.print("\n[bold cyan]API Keys Configuration[/bold cyan]")
    console.print("Configure at least one API key for embedding generation:")
    
    # Gemini API key
    gemini_key = Prompt.ask(
        "Google Gemini API key (recommended)",
        password=True,
        default=config.gemini_api_key or "",
        show_default=False
    )
    if gemini_key:
        config.gemini_api_key = gemini_key
    
    # OpenAI API key
    openai_key = Prompt.ask(
        "OpenAI API key (optional fallback)",
        password=True,
        default=config.openai_api_key or "",
        show_default=False
    )
    if openai_key:
        config.openai_api_key = openai_key
    
    if not config.gemini_api_key and not config.openai_api_key:
        console.print("[yellow]Warning: No API keys configured. Embedding generation will not work.[/yellow]")
        if not Confirm.ask("Continue without API keys?"):
            sys.exit(1)
    
    return config


def _load_api_keys_from_env(config: CLIConfig, logger) -> CLIConfig:
    """Load API keys from environment variables."""
    if not config.gemini_api_key and not config.openai_api_key:
        logger.warning("No API keys found in environment variables")
        logger.info("Set GEMINI_API_KEY or OPENAI_API_KEY for embedding generation")
    else:
        logger.success("API keys loaded from environment")
    
    return config


def _interactive_preferences_setup(config: CLIConfig, console: Console) -> CLIConfig:
    """Interactive preferences configuration."""
    console.print("\n[bold cyan]Application Preferences[/bold cyan]")
    
    # API host and port
    config.api_host = Prompt.ask(
        "API server host",
        default=config.api_host
    )
    
    config.api_port = int(Prompt.ask(
        "API server port",
        default=str(config.api_port)
    ))
    
    # Default language
    config.default_language = Prompt.ask(
        "Default programming language",
        default=config.default_language,
        choices=["python", "javascript", "typescript", "java", "go", "rust"]
    )
    
    # Output format
    config.output_format = Prompt.ask(
        "Default output format",
        default=config.output_format,
        choices=["table", "json", "yaml"]
    )
    
    # Batch size
    config.batch_size = int(Prompt.ask(
        "Batch size for processing",
        default=str(config.batch_size)
    ))
    
    return config


async def _test_database_connection(config: CLIConfig, logger) -> None:
    """Test database connection."""
    try:
        # Set environment variables
        env_vars = config.to_env_dict()
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Import and test connection
        from src.db.connection import get_async_session
        
        async with await get_async_session() as session:
            # Execute a simple query
            result = await session.execute("SELECT 1 as test")
            row = result.fetchone()
            
            if row and row[0] == 1:
                logger.success("Database connection successful")
            else:
                logger.error("Database connection test failed")
                sys.exit(1)
    
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.info("Please check your database configuration")
        sys.exit(1)


async def _create_database_tables(config: CLIConfig, logger) -> None:
    """Create database tables."""
    try:
        # Set environment variables
        env_vars = config.to_env_dict()
        for key, value in env_vars.items():
            os.environ[key] = value
        
        # Import and run schema creation
        from src.db.schema import create_tables
        
        await create_tables()
        logger.success("Database tables created successfully")
    
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        logger.info("You may need to create tables manually or check permissions")
        # Don't exit here as this is optional


def _show_setup_summary(config: CLIConfig, console: Console) -> None:
    """Show setup summary."""
    from rich.table import Table
    
    table = Table(title="Configuration Summary", show_header=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Database settings
    table.add_row("Database Host", config.tidb_host or "Not configured")
    table.add_row("Database Port", str(config.tidb_port))
    table.add_row("Database Name", config.tidb_database)
    table.add_row("Database User", config.tidb_user or "Not configured")
    
    # API settings
    table.add_row("API Host", config.api_host)
    table.add_row("API Port", str(config.api_port))
    
    # API keys
    table.add_row("Gemini API Key", "✅ Configured" if config.gemini_api_key else "❌ Not configured")
    table.add_row("OpenAI API Key", "✅ Configured" if config.openai_api_key else "❌ Not configured")
    
    # Preferences
    table.add_row("Default Language", config.default_language)
    table.add_row("Output Format", config.output_format)
    table.add_row("Batch Size", str(config.batch_size))
    
    console.print(table)