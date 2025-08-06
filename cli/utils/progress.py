"""Progress tracking utilities for CLI operations."""

import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressTracker:
    """Enhanced progress tracker for CLI operations."""
    
    def __init__(self, console: Optional[Console] = None, quiet: bool = False):
        self.console = console or Console()
        self.quiet = quiet
        self.progress: Optional[Progress] = None
        self.current_task: Optional[TaskID] = None
        
    @contextmanager
    def track_progress(
        self,
        description: str,
        total: Optional[int] = None,
        show_speed: bool = False
    ) -> Generator["ProgressTracker", None, None]:
        """Context manager for tracking progress.
        
        Args:
            description: Task description
            total: Total number of items (None for indeterminate)
            show_speed: Whether to show processing speed
        """
        if self.quiet:
            yield self
            return
            
        # Create progress columns
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ]
        
        if total is not None:
            columns.extend([
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ])
        else:
            columns.append(TimeElapsedColumn())
        
        # Create and start progress
        self.progress = Progress(*columns, console=self.console)
        self.progress.start()
        
        try:
            self.current_task = self.progress.add_task(description, total=total)
            yield self
        finally:
            if self.progress:
                self.progress.stop()
                self.progress = None
                self.current_task = None
    
    def update(self, advance: int = 1, description: Optional[str] = None) -> None:
        """Update progress.
        
        Args:
            advance: Number of items to advance
            description: Optional new description
        """
        if self.progress and self.current_task is not None:
            if description:
                self.progress.update(self.current_task, description=description, advance=advance)
            else:
                self.progress.advance(self.current_task, advance)
    
    def set_total(self, total: int) -> None:
        """Set total for indeterminate progress."""
        if self.progress and self.current_task is not None:
            self.progress.update(self.current_task, total=total)
    
    def complete(self, message: Optional[str] = None) -> None:
        """Mark task as complete."""
        if self.progress and self.current_task is not None:
            if message:
                self.progress.update(self.current_task, description=message)
            self.progress.update(self.current_task, completed=True)


class AsyncProgressCallback:
    """Callback for async operations with progress tracking."""
    
    def __init__(self, tracker: ProgressTracker):
        self.tracker = tracker
        self.start_time = time.time()
        self.last_update = 0
        
    def __call__(self, progress_data: Dict[str, Any]) -> None:
        """Handle progress callback.
        
        Args:
            progress_data: Progress information from async operation
        """
        progress_type = progress_data.get("type", "unknown")
        
        if progress_type == "repository_start":
            total_files = progress_data.get("total_files", 0)
            self.tracker.set_total(total_files)
            
        elif progress_type == "progress":
            current = progress_data.get("files_processed", 0)
            current_file = progress_data.get("current_file", "")
            
            # Update progress less frequently to avoid spam
            if current - self.last_update >= 1:
                file_name = current_file.split("/")[-1] if current_file else "processing..."
                self.tracker.update(
                    advance=current - self.last_update,
                    description=f"Analyzing: {file_name}"
                )
                self.last_update = current
                
        elif progress_type == "file_complete":
            file_path = progress_data.get("file", "")
            status = progress_data.get("status", "completed")
            file_name = file_path.split("/")[-1] if file_path else "file"
            
            if status == "success":
                description = f"✅ Completed: {file_name}"
            elif status == "error":
                description = f"❌ Failed: {file_name}"
            else:
                description = f"⚠️  Skipped: {file_name}"
                
            self.tracker.update(description=description)
            
        elif progress_type == "repository_complete":
            analyzed = progress_data.get("analyzed_files", 0)
            failed = progress_data.get("failed_files", 0)
            elapsed = progress_data.get("total_time_seconds", 0)
            
            summary = f"Completed: {analyzed} analyzed, {failed} failed ({elapsed:.1f}s)"
            self.tracker.complete(summary)


@contextmanager
def create_progress_callback(
    description: str,
    console: Optional[Console] = None,
    quiet: bool = False
) -> Generator[Callable[[Dict[str, Any]], None], None, None]:
    """Create progress callback for async operations.
    
    Args:
        description: Initial task description
        console: Optional console instance
        quiet: Whether to suppress progress display
        
    Yields:
        Callback function for progress updates
    """
    tracker = ProgressTracker(console, quiet)
    
    with tracker.track_progress(description) as progress_tracker:
        callback = AsyncProgressCallback(progress_tracker)
        yield callback


# Convenience function for simple progress tracking
def track_iterable(
    iterable,
    description: str = "Processing...",
    console: Optional[Console] = None,
    quiet: bool = False
):
    """Track progress over an iterable.
    
    Args:
        iterable: Items to iterate over
        description: Task description
        console: Optional console instance
        quiet: Whether to suppress progress display
        
    Yields:
        Items from the iterable
    """
    if quiet:
        yield from iterable
        return
        
    items = list(iterable)
    tracker = ProgressTracker(console, quiet)
    
    with tracker.track_progress(description, total=len(items)) as progress_tracker:
        for item in items:
            yield item
            progress_tracker.update()