"""Output formatting utilities for CLI."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from rich.console import Console
from rich.table import Table
from rich.tree import Tree
from rich.panel import Panel
from rich.syntax import Syntax


class OutputFormatter:
    """Handles different output formats for CLI results."""
    
    def __init__(self, console: Optional[Console] = None, format_type: str = "table"):
        self.console = console or Console()
        self.format_type = format_type
    
    def output(self, data: Any, title: Optional[str] = None) -> None:
        """Output data in the specified format.
        
        Args:
            data: Data to output
            title: Optional title for the output
        """
        if self.format_type == "json":
            self._output_json(data)
        elif self.format_type == "yaml":
            self._output_yaml(data)
        else:
            self._output_rich(data, title)
    
    def _output_json(self, data: Any) -> None:
        """Output data as JSON."""
        try:
            json_str = json.dumps(data, indent=2, default=str)
            self.console.print(json_str)
        except Exception as e:
            self.console.print(f"[red]Error formatting JSON: {e}[/red]")
            self.console.print(str(data))
    
    def _output_yaml(self, data: Any) -> None:
        """Output data as YAML."""
        try:
            yaml_str = yaml.dump(data, default_flow_style=False, sort_keys=False)
            self.console.print(yaml_str)
        except Exception as e:
            self.console.print(f"[red]Error formatting YAML: {e}[/red]")
            self.console.print(str(data))
    
    def _output_rich(self, data: Any, title: Optional[str] = None) -> None:
        """Output data with rich formatting."""
        if isinstance(data, dict):
            self._output_dict(data, title)
        elif isinstance(data, list):
            self._output_list(data, title)
        else:
            self.console.print(str(data))
    
    def _output_dict(self, data: Dict, title: Optional[str] = None) -> None:
        """Output dictionary with rich formatting."""
        if title:
            self.console.print(f"\n[bold blue]{title}[/bold blue]")
        
        # Handle specific data types
        if "file_analyses" in data:
            self._output_repository_analysis(data)
        elif "metrics" in data and "chunks" in data:
            self._output_file_analysis(data)
        elif "similar_examples" in data:
            self._output_similarity_results(data)
        elif "results" in data:
            self._output_search_results(data)
        else:
            self._output_generic_dict(data)
    
    def _output_list(self, data: List, title: Optional[str] = None) -> None:
        """Output list with rich formatting."""
        if title:
            self.console.print(f"\n[bold blue]{title}[/bold blue]")
        
        if not data:
            self.console.print("[dim]No results found[/dim]")
            return
        
        # Handle different list types
        if all(isinstance(item, dict) for item in data):
            self._output_table_from_list(data)
        else:
            for i, item in enumerate(data):
                self.console.print(f"{i+1}. {item}")
    
    def _output_repository_analysis(self, data: Dict) -> None:
        """Format repository analysis results."""
        # Summary panel
        summary_table = Table(show_header=False, box=None)
        summary_table.add_row("Repository", data.get("repository_path", "Unknown"))
        summary_table.add_row("Total Files", str(data.get("total_files", 0)))
        summary_table.add_row("Analyzed", str(data.get("analyzed_files", 0)))
        summary_table.add_row("Failed", str(data.get("failed_files", 0)))
        summary_table.add_row("Total Time", f"{data.get('total_time_seconds', 0):.2f}s")
        
        if data.get("total_lines"):
            summary_table.add_row("Total Lines", str(data["total_lines"]))
        if data.get("average_complexity"):
            summary_table.add_row("Avg Complexity", f"{data['average_complexity']:.2f}")
        
        self.console.print(Panel(summary_table, title="Repository Analysis Summary"))
        
        # File details table
        if data.get("file_analyses"):
            files_table = Table(title="File Analysis Results")
            files_table.add_column("File", style="cyan")
            files_table.add_column("Status", style="green")
            files_table.add_column("Lines", justify="right")
            files_table.add_column("Complexity", justify="right")
            files_table.add_column("Functions", justify="right")
            files_table.add_column("Time (ms)", justify="right")
            
            for file_analysis in data["file_analyses"][:20]:  # Limit to first 20
                metrics = file_analysis.get("metrics", {})
                status_style = "green" if file_analysis.get("status") == "success" else "red"
                
                files_table.add_row(
                    Path(file_analysis.get("file_path", "")).name,
                    f"[{status_style}]{file_analysis.get('status', 'unknown')}[/{status_style}]",
                    str(metrics.get("lines_of_code", 0)),
                    f"{metrics.get('average_complexity', 0):.1f}",
                    str(len(file_analysis.get("functions", []))),
                    f"{file_analysis.get('analysis_time_ms', 0):.1f}"
                )
            
            self.console.print(files_table)
            
            if len(data["file_analyses"]) > 20:
                self.console.print(f"[dim]... and {len(data['file_analyses']) - 20} more files[/dim]")
    
    def _output_file_analysis(self, data: Dict) -> None:
        """Format single file analysis results."""
        file_path = data.get("file_path", "Unknown")
        self.console.print(f"\n[bold cyan]Analysis: {Path(file_path).name}[/bold cyan]")
        
        # Metrics panel
        metrics = data.get("metrics", {})
        if metrics:
            metrics_table = Table(show_header=False, box=None)
            metrics_table.add_row("Lines of Code", str(metrics.get("lines_of_code", 0)))
            metrics_table.add_row("Cyclomatic Complexity", str(metrics.get("cyclomatic_complexity", 0)))
            metrics_table.add_row("Maintainability Index", f"{metrics.get('maintainability_index', 0):.1f}")
            metrics_table.add_row("Risk Level", metrics.get("risk_level", "Unknown"))
            
            self.console.print(Panel(metrics_table, title="Code Metrics"))
        
        # Functions table
        functions = data.get("functions", [])
        if functions:
            func_table = Table(title="Functions")
            func_table.add_column("Name", style="cyan")
            func_table.add_column("Lines", justify="right")
            func_table.add_column("Parameters", justify="right")
            func_table.add_column("Complexity", justify="right")
            
            for func in functions[:10]:  # Limit to first 10
                func_table.add_row(
                    func.get("name", ""),
                    str(func.get("end_line", 0) - func.get("start_line", 0)),
                    str(len(func.get("parameters", []))),
                    str(func.get("complexity", 0))
                )
            
            self.console.print(func_table)
        
        # Chunks info
        chunks = data.get("chunks", [])
        if chunks:
            self.console.print(f"\n[dim]Generated {len(chunks)} chunks for embedding[/dim]")
    
    def _output_similarity_results(self, data: Dict) -> None:
        """Format similarity search results."""
        similar_examples = data.get("similar_examples", [])
        recommendations = data.get("recommendations", [])
        
        if similar_examples:
            results_table = Table(title="Similar Code Examples")
            results_table.add_column("File", style="cyan")
            results_table.add_column("Similarity", justify="right", style="green")
            results_table.add_column("Lines", justify="center")
            results_table.add_column("Repository", style="dim")
            
            for example in similar_examples:
                similarity = example.get("similarity_score", 0)
                results_table.add_row(
                    Path(example.get("file_path", "")).name,
                    f"{similarity:.1%}",
                    f"{example.get('start_line', 0)}-{example.get('end_line', 0)}",
                    example.get("repository_name", "Unknown")
                )
            
            self.console.print(results_table)
        
        if recommendations:
            self.console.print("\n[bold green]Recommendations:[/bold green]")
            for i, rec in enumerate(recommendations[:5], 1):
                self.console.print(f"{i}. {rec}")
    
    def _output_search_results(self, data: Dict) -> None:
        """Format search results."""
        results = data.get("results", [])
        
        if not results:
            self.console.print("[dim]No search results found[/dim]")
            return
        
        results_table = Table(title="Search Results")
        results_table.add_column("File", style="cyan")
        results_table.add_column("Match Score", justify="right", style="green")
        results_table.add_column("Lines", justify="center")
        results_table.add_column("Type", style="blue")
        
        for result in results:
            match_score = result.get("similarity_score", 0)
            results_table.add_row(
                Path(result.get("file_path", "")).name,
                f"{(1.0 - match_score):.1%}",
                f"{result.get('start_line', 0)}-{result.get('end_line', 0)}",
                result.get("chunk_type", "code")
            )
        
        self.console.print(results_table)
        
        # Show statistics if available
        stats = data.get("statistics", {})
        if stats:
            stats_table = Table(show_header=False, box=None)
            for key, value in stats.items():
                stats_table.add_row(key.replace("_", " ").title(), str(value))
            
            self.console.print(Panel(stats_table, title="Search Statistics"))
    
    def _output_generic_dict(self, data: Dict) -> None:
        """Output generic dictionary as a table."""
        table = Table(show_header=False, box=None)
        
        for key, value in data.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                value_str = f"{type(value).__name__} ({len(value)} items)"
            else:
                value_str = str(value)
            
            table.add_row(key.replace("_", " ").title(), value_str)
        
        self.console.print(table)
    
    def _output_table_from_list(self, data: List[Dict]) -> None:
        """Create table from list of dictionaries."""
        if not data:
            return
        
        # Get all unique keys
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        # Create table with common columns
        table = Table()
        common_keys = ["name", "file", "path", "type", "status", "score", "similarity"]
        
        # Add columns in preferred order
        added_keys = set()
        for key in common_keys:
            if key in all_keys:
                table.add_column(key.title(), style="cyan" if key in ["name", "file", "path"] else None)
                added_keys.add(key)
        
        # Add remaining keys
        for key in sorted(all_keys - added_keys):
            table.add_column(key.replace("_", " ").title())
            added_keys.add(key)
        
        # Add rows
        for item in data[:20]:  # Limit to first 20 items
            row = []
            for key in added_keys:
                value = item.get(key, "")
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            table.add_row(*row)
        
        self.console.print(table)
        
        if len(data) > 20:
            self.console.print(f"[dim]... and {len(data) - 20} more items[/dim]")


def format_code_snippet(code: str, language: str = "python", theme: str = "monokai") -> Syntax:
    """Format code snippet with syntax highlighting.
    
    Args:
        code: Code to format
        language: Programming language
        theme: Syntax highlighting theme
        
    Returns:
        Rich Syntax object
    """
    return Syntax(code, language, theme=theme, line_numbers=True)


def create_summary_panel(title: str, data: Dict[str, Union[str, int, float]]) -> Panel:
    """Create a summary panel from key-value data.
    
    Args:
        title: Panel title
        data: Dictionary of summary data
        
    Returns:
        Rich Panel object
    """
    table = Table(show_header=False, box=None)
    
    for key, value in data.items():
        if isinstance(value, float):
            value_str = f"{value:.2f}"
        else:
            value_str = str(value)
        
        table.add_row(key.replace("_", " ").title(), value_str)
    
    return Panel(table, title=title)