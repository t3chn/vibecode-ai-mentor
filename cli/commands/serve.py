"""Serve command to start the FastAPI server."""

import os
import sys
from typing import Optional

import click
from rich.console import Console

from cli.utils.config import CLIConfig
from cli.utils.logging import get_cli_logger


@click.command(name="serve")
@click.option(
    "--host",
    "-h",
    default="0.0.0.0",
    help="Host to bind the server to"
)
@click.option(
    "--port",
    "-p",
    default=8000,
    help="Port to bind the server to"
)
@click.option(
    "--workers",
    "-w",
    default=1,
    help="Number of worker processes"
)
@click.option(
    "--reload",
    is_flag=True,
    help="Enable auto-reload on code changes"
)
@click.option(
    "--log-level",
    type=click.Choice(["debug", "info", "warning", "error", "critical"]),
    default="info",
    help="Log level for uvicorn"
)
@click.option(
    "--access-log/--no-access-log",
    default=True,
    help="Enable/disable access logging"
)
@click.pass_context
def serve_cmd(
    ctx: click.Context,
    host: str,
    port: int,
    workers: int,
    reload: bool,
    log_level: str,
    access_log: bool
):
    """Start the FastAPI server for VibeCode AI Mentor.
    
    This starts the REST API server that provides endpoints for code analysis,
    repository indexing, and similarity search.
    
    Examples:
        vibecode serve
        vibecode serve --host localhost --port 8080
        vibecode serve --reload --log-level debug
        vibecode serve --workers 4 --no-access-log
    """
    # Get CLI context
    config: CLIConfig = ctx.obj["config"]
    quiet: bool = ctx.obj["quiet"]
    verbose: bool = ctx.obj["verbose"]
    
    logger = get_cli_logger(__name__, quiet)
    console = Console()
    
    # Check database configuration
    if not config.has_database_config:
        logger.error("Database configuration missing. Run 'vibecode setup' first.")
        sys.exit(1)
    
    # Check API keys
    if not config.has_api_keys:
        logger.warning("No API keys configured. Some features may not work.")
        logger.info("Configure GEMINI_API_KEY or OPENAI_API_KEY for full functionality.")
    
    # Set environment variables
    env_vars = config.to_env_dict()
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Override with CLI options
    if host != "0.0.0.0":
        config.api_host = host
    if port != 8000:
        config.api_port = port
    
    # Determine uvicorn options
    uvicorn_kwargs = {
        "app": "src.api.app:app",
        "host": config.api_host,
        "port": config.api_port,
        "log_level": log_level,
        "access_log": access_log,
    }
    
    # Add reload if in development
    if reload or config.environment == "development":
        uvicorn_kwargs["reload"] = True
        uvicorn_kwargs["reload_dirs"] = ["src", "cli"]
        workers = 1  # Force single worker with reload
    
    # Add workers if not reloading
    if not uvicorn_kwargs.get("reload", False):
        uvicorn_kwargs["workers"] = workers
    
    try:
        import uvicorn
        
        logger.step(f"Starting server at http://{config.api_host}:{config.api_port}")
        
        if reload:
            logger.info("Auto-reload enabled - server will restart on code changes")
        
        if workers > 1 and not reload:
            logger.info(f"Starting {workers} worker processes")
        
        if not quiet:
            console.print(f"[green]Server configuration:[/green]")
            console.print(f"  Host: {config.api_host}")
            console.print(f"  Port: {config.api_port}")
            console.print(f"  Workers: {workers if not reload else 1}")
            console.print(f"  Log level: {log_level}")
            console.print(f"  Environment: {config.environment}")
            console.print(f"  Database: {config.tidb_host}:{config.tidb_port}/{config.tidb_database}")
            
            console.print(f"\n[blue]API Documentation:[/blue]")
            console.print(f"  Swagger UI: http://{config.api_host}:{config.api_port}/docs")
            console.print(f"  ReDoc: http://{config.api_host}:{config.api_port}/redoc")
            console.print(f"  OpenAPI JSON: http://{config.api_host}:{config.api_port}/openapi.json")
            
            console.print(f"\n[dim]Press Ctrl+C to stop the server[/dim]")
        
        # Start the server
        uvicorn.run(**uvicorn_kwargs)
        
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)


@click.command(name="status")
@click.option(
    "--host",
    "-h",
    help="API server host (default: from config)"
)
@click.option(
    "--port",
    "-p",
    type=int,
    help="API server port (default: from config)"
)
@click.option(
    "--timeout",
    "-t",
    default=5,
    help="Request timeout in seconds"
)
@click.pass_context
def status_cmd(
    ctx: click.Context,
    host: Optional[str],
    port: Optional[int],
    timeout: int
):
    """Check the status of the running API server.
    
    Examples:
        vibecode status
        vibecode status --host localhost --port 8080
        vibecode status --timeout 10
    """
    # Get CLI context
    config: CLIConfig = ctx.obj["config"]
    quiet: bool = ctx.obj["quiet"]
    verbose: bool = ctx.obj["verbose"]
    
    logger = get_cli_logger(__name__, quiet)
    console = Console()
    
    # Use config values if not overridden
    server_host = host or config.api_host
    server_port = port or config.api_port
    
    try:
        import httpx
        
        base_url = f"http://{server_host}:{server_port}"
        
        logger.step(f"Checking server status at {base_url}")
        
        # Check health endpoint
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{base_url}/health")
            
            if response.status_code == 200:
                health_data = response.json()
                
                logger.success("Server is running and healthy")
                
                if not quiet:
                    console.print(f"[green]Server Status:[/green]")
                    console.print(f"  URL: {base_url}")
                    console.print(f"  Status: {health_data.get('status', 'unknown')}")
                    console.print(f"  Version: {health_data.get('version', 'unknown')}")
                    console.print(f"  Uptime: {health_data.get('uptime', 'unknown')}")
                    
                    # Show database status if available
                    db_status = health_data.get('database', {})
                    if db_status:
                        console.print(f"  Database: {db_status.get('status', 'unknown')}")
                    
                    # Show service statuses
                    services = health_data.get('services', {})
                    if services:
                        console.print(f"[blue]Services:[/blue]")
                        for service, status in services.items():
                            status_icon = "✅" if status == "healthy" else "❌"
                            console.print(f"  {status_icon} {service}: {status}")
            
            else:
                logger.error(f"Server returned status {response.status_code}")
                if verbose:
                    console.print(f"Response: {response.text}")
                sys.exit(1)
    
    except ImportError:
        logger.error("httpx not installed. Install with: pip install httpx")
        sys.exit(1)
    except httpx.ConnectError:
        logger.error(f"Cannot connect to server at {base_url}")
        logger.info("Make sure the server is running with 'vibecode serve'")
        sys.exit(1)
    except httpx.TimeoutException:
        logger.error(f"Server at {base_url} is not responding (timeout: {timeout}s)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to check server status: {e}")
        if verbose:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)