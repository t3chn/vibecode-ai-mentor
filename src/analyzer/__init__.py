"""Code analysis and parsing functionality using tree-sitter.

This module handles parsing source code files, extracting AST nodes,
identifying code patterns, and preparing code chunks for embedding generation.
"""

from src.analyzer.parser import PythonParser

__all__ = ["PythonParser"]