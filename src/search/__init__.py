"""Vector similarity search functionality using TiDB.

This module implements vector search queries, hybrid search combining
vector similarity with SQL filters, result ranking, and search optimization
strategies for finding similar code patterns.
"""

from src.search.vector_search import (
    VectorSearchService,
    CodeMatch,
    SearchFilters,
)
from src.search.service import (
    SearchServiceManager,
    search_manager,
    quick_code_search,
    find_code_similarities,
    detect_code_duplicates,
)

__all__ = [
    "VectorSearchService",
    "CodeMatch", 
    "SearchFilters",
    "SearchServiceManager",
    "search_manager",
    "quick_code_search",
    "find_code_similarities", 
    "detect_code_duplicates",
]