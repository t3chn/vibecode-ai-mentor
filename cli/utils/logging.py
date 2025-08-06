"""Logging utilities for CLI."""

import logging
import sys
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler


def setup_cli_logging(level: str = "INFO", show_path: bool = False) -> None:
    """Setup logging for CLI with rich formatting.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        show_path: Whether to show file paths in log messages
    """
    # Remove existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create rich handler
    console = Console(file=sys.stderr)
    rich_handler = RichHandler(
        console=console,
        show_time=False,
        show_path=show_path,
        rich_tracebacks=True,
        tracebacks_show_locals=level == "DEBUG"
    )
    
    # Set up formatting
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]"
    )
    rich_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.addHandler(rich_handler)
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Suppress some noisy loggers in CLI context
    if level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("asyncio").setLevel(logging.WARNING)


class CLILogger:
    """Enhanced logger for CLI operations."""
    
    def __init__(self, name: str, quiet: bool = False):
        self.logger = logging.getLogger(name)
        self.console = Console()
        self.quiet = quiet
    
    def info(self, message: str, style: Optional[str] = None) -> None:
        """Log info message."""
        if not self.quiet:
            if style:
                self.console.print(message, style=style)
            else:
                self.console.print(message)
    
    def success(self, message: str) -> None:
        """Log success message."""
        if not self.quiet:
            self.console.print(f"✅ {message}", style="green")
    
    def warning(self, message: str) -> None:
        """Log warning message."""
        if not self.quiet:
            self.console.print(f"⚠️  {message}", style="yellow")
    
    def error(self, message: str) -> None:
        """Log error message."""
        self.console.print(f"❌ {message}", style="red")
    
    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)
    
    def step(self, message: str) -> None:
        """Log step message."""
        if not self.quiet:
            self.console.print(f"▶️  {message}", style="blue")
    
    def progress(self, message: str) -> None:
        """Log progress message."""
        if not self.quiet:
            self.console.print(f"🔄 {message}", style="dim")


def get_cli_logger(name: str, quiet: bool = False) -> CLILogger:
    """Get CLI logger instance."""
    return CLILogger(name, quiet)