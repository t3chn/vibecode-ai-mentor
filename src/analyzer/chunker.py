"""Smart code chunking strategy for VibeCode AI Mentor.

Implements various chunking strategies that respect code structure and boundaries.
Ensures chunks are within the configured size limits while preserving semantic meaning.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import tiktoken

from src.analyzer.parser import PythonParser

# Make module work without full config
try:
    from src.core.config import get_settings
    from src.core.logging import get_logger
    logger = get_logger(__name__)
    settings = get_settings()
except Exception:
    # Fallback for standalone usage
    import logging
    logger = logging.getLogger(__name__)
    
    class MockSettings:
        min_chunk_size = 512
        max_chunk_size = 2048
    
    settings = MockSettings()


class ChunkStrategy(Enum):
    """Available chunking strategies."""
    BY_FUNCTION = "by_function"
    BY_CLASS = "by_class"
    SLIDING_WINDOW = "sliding_window"
    SMART_CHUNK = "smart_chunk"


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""
    content: str
    start_line: int
    end_line: int
    chunk_type: str  # function, class, block, or mixed
    metadata: dict  # Additional info like function/class name
    token_count: int


class CodeChunker:
    """Smart code chunking with structure awareness."""
    
    def __init__(self):
        """Initialize chunker with token encoder."""
        self.parser = PythonParser()
        # Use cl100k_base encoding (same as GPT-4)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.min_chunk_size = settings.min_chunk_size
        self.max_chunk_size = settings.max_chunk_size
        self.overlap_ratio = 0.1  # 10% overlap for sliding window
        
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoder.encode(text))
    
    async def chunk_file(
        self, 
        file_path: str | Path, 
        strategy: ChunkStrategy = ChunkStrategy.SMART_CHUNK
    ) -> List[CodeChunk]:
        """Chunk a file using the specified strategy.
        
        Args:
            file_path: Path to the Python file
            strategy: Chunking strategy to use
            
        Returns:
            List of code chunks
        """
        file_path = Path(file_path)
        
        # Parse the file
        try:
            tree = await self.parser.parse_file(file_path)
        except Exception as e:
            logger.error(f"Failed to parse {file_path}: {e}")
            # Fall back to sliding window if parsing fails
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return self._chunk_by_sliding_window(code)
        
        # Read the code for chunking
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
            
        return self.chunk_code(code, strategy)
    
    def chunk_code(
        self, 
        code: str, 
        strategy: ChunkStrategy = ChunkStrategy.SMART_CHUNK
    ) -> List[CodeChunk]:
        """Chunk code using the specified strategy.
        
        Args:
            code: Python code to chunk
            strategy: Chunking strategy to use
            
        Returns:
            List of code chunks
        """
        # Parse the code
        self.parser.parse_code(code)
        
        if strategy == ChunkStrategy.BY_FUNCTION:
            return self._chunk_by_function(code)
        elif strategy == ChunkStrategy.BY_CLASS:
            return self._chunk_by_class(code)
        elif strategy == ChunkStrategy.SLIDING_WINDOW:
            return self._chunk_by_sliding_window(code)
        elif strategy == ChunkStrategy.SMART_CHUNK:
            return self._smart_chunk(code)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _chunk_by_function(self, code: str) -> List[CodeChunk]:
        """Create one chunk per function/method."""
        chunks = []
        functions = self.parser.extract_functions()
        code_lines = code.split('\n')
        
        for func in functions:
            start_line = func['location']['start_line']
            end_line = func['location']['end_line']
            
            # Include decorators if present
            if func['decorators']:
                # Find decorator start line by looking backwards
                for i in range(start_line - 2, -1, -1):
                    if i < len(code_lines) and code_lines[i].strip().startswith('@'):
                        start_line = i + 1
                    else:
                        break
            
            # Extract function code
            func_lines = code_lines[start_line - 1:end_line]
            func_code = '\n'.join(func_lines)
            
            # Check token count
            token_count = self.count_tokens(func_code)
            
            # If function is too large, split it
            if token_count > self.max_chunk_size:
                sub_chunks = self._split_large_block(
                    func_code, start_line, 'function', {'name': func['name']}
                )
                chunks.extend(sub_chunks)
            else:
                chunk = CodeChunk(
                    content=func_code,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type='function',
                    metadata={'name': func['name'], 'params': func['parameters']},
                    token_count=token_count
                )
                chunks.append(chunk)
        
        # Add module-level code (imports, globals) if any
        module_chunks = self._extract_module_level_chunks(code, functions)
        chunks.extend(module_chunks)
        
        return sorted(chunks, key=lambda c: c.start_line)
    
    def _chunk_by_class(self, code: str) -> List[CodeChunk]:
        """Create one chunk per class (including all methods)."""
        chunks = []
        classes = self.parser.extract_classes()
        code_lines = code.split('\n')
        
        # Debug: log number of classes found
        logger.debug(f"Found {len(classes)} classes")
        
        for cls in classes:
            start_line = cls['location']['start_line']
            end_line = cls['location']['end_line']
            
            # Include decorators if present
            if cls['decorators']:
                for i in range(start_line - 2, -1, -1):
                    if i < len(code_lines) and code_lines[i].strip().startswith('@'):
                        start_line = i + 1
                    else:
                        break
            
            # Extract class code
            class_lines = code_lines[start_line - 1:end_line]
            class_code = '\n'.join(class_lines)
            
            # Check token count
            token_count = self.count_tokens(class_code)
            
            # If class is too large, split by methods
            if token_count > self.max_chunk_size:
                sub_chunks = self._split_class_by_methods(
                    cls, code_lines, start_line
                )
                chunks.extend(sub_chunks)
            else:
                chunk = CodeChunk(
                    content=class_code,
                    start_line=start_line,
                    end_line=end_line,
                    chunk_type='class',
                    metadata={
                        'name': cls['name'],
                        'bases': cls['bases'],
                        'methods': cls['methods']
                    },
                    token_count=token_count
                )
                chunks.append(chunk)
        
        # Add non-class code only if there are significant portions
        non_class_chunks = self._extract_non_class_chunks(code, classes)
        # Filter out very small chunks that are just whitespace or minimal
        non_class_chunks = [c for c in non_class_chunks 
                           if c.token_count >= 10 or len(c.content.strip()) > 50]
        chunks.extend(non_class_chunks)
        
        return sorted(chunks, key=lambda c: c.start_line)
    
    def _chunk_by_sliding_window(self, code: str) -> List[CodeChunk]:
        """Create chunks using sliding window (for unstructured code)."""
        chunks = []
        lines = code.split('\n')
        
        # Calculate overlap size
        overlap_tokens = int(self.max_chunk_size * self.overlap_ratio)
        
        current_start = 0
        while current_start < len(lines):
            current_chunk = []
            current_tokens = 0
            current_end = current_start
            
            # Build chunk up to max size
            while current_end < len(lines) and current_tokens < self.max_chunk_size:
                line = lines[current_end]
                line_tokens = self.count_tokens(line + '\n')
                
                if current_tokens + line_tokens > self.max_chunk_size:
                    break
                    
                current_chunk.append(line)
                current_tokens += line_tokens
                current_end += 1
            
            # Ensure minimum chunk size
            if current_tokens < self.min_chunk_size and current_end < len(lines):
                # Keep adding lines until min size
                while current_end < len(lines) and current_tokens < self.min_chunk_size:
                    line = lines[current_end]
                    current_chunk.append(line)
                    current_tokens += self.count_tokens(line + '\n')
                    current_end += 1
            
            if current_chunk:
                chunk = CodeChunk(
                    content='\n'.join(current_chunk),
                    start_line=current_start + 1,
                    end_line=current_end,
                    chunk_type='block',
                    metadata={'window': True},
                    token_count=current_tokens
                )
                chunks.append(chunk)
            
            # Move window with overlap
            if current_end >= len(lines):
                break
                
            # Calculate next start position with overlap
            overlap_lines = 0
            overlap_token_count = 0
            for i in range(current_end - 1, current_start - 1, -1):
                overlap_token_count += self.count_tokens(lines[i] + '\n')
                if overlap_token_count >= overlap_tokens:
                    overlap_lines = current_end - i
                    break
            
            current_start = current_end - overlap_lines if overlap_lines > 0 else current_end
        
        return chunks
    
    def _smart_chunk(self, code: str) -> List[CodeChunk]:
        """Adaptive chunking that picks the best strategy based on code structure."""
        # Analyze code structure
        functions = self.parser.extract_functions()
        classes = self.parser.extract_classes()
        
        # Debug logging
        logger.debug(f"Smart chunk analysis: {len(classes)} classes, {len(functions)} functions")
        
        # If code is mostly classes, use by_class strategy
        class_lines = sum(c['location']['end_line'] - c['location']['start_line'] + 1 
                         for c in classes)
        total_lines = len(code.split('\n'))
        
        if classes and class_lines > total_lines * 0.5:  # 50% of code is in classes
            logger.debug("Using by_class strategy")
            return self._chunk_by_class(code)
        
        # If code has many standalone functions, use by_function
        standalone_functions = [f for f in functions 
                               if not any(self._is_function_in_class(f, c) for c in classes)]
        
        if len(standalone_functions) > 5:
            logger.debug("Using by_function strategy")
            return self._chunk_by_function(code)
        
        # For mixed code, use hybrid approach
        logger.debug("Using hybrid strategy")
        return self._hybrid_chunk(code, functions, classes)
    
    def _hybrid_chunk(
        self, 
        code: str, 
        functions: List[dict], 
        classes: List[dict]
    ) -> List[CodeChunk]:
        """Hybrid chunking for mixed code structures."""
        chunks = []
        code_lines = code.split('\n')
        processed_lines = set()
        
        # Process classes first
        for cls in classes:
            start = cls['location']['start_line']
            end = cls['location']['end_line']
            
            # Mark lines as processed
            for i in range(start, end + 1):
                processed_lines.add(i)
            
            # Create class chunk
            class_code = '\n'.join(code_lines[start - 1:end])
            token_count = self.count_tokens(class_code)
            
            if token_count > self.max_chunk_size:
                sub_chunks = self._split_class_by_methods(cls, code_lines, start)
                chunks.extend(sub_chunks)
            else:
                chunk = CodeChunk(
                    content=class_code,
                    start_line=start,
                    end_line=end,
                    chunk_type='class',
                    metadata={'name': cls['name']},
                    token_count=token_count
                )
                chunks.append(chunk)
        
        # Process standalone functions
        for func in functions:
            start = func['location']['start_line']
            
            # Skip if already in a class
            if start in processed_lines:
                continue
                
            end = func['location']['end_line']
            for i in range(start, end + 1):
                processed_lines.add(i)
            
            func_code = '\n'.join(code_lines[start - 1:end])
            token_count = self.count_tokens(func_code)
            
            chunk = CodeChunk(
                content=func_code,
                start_line=start,
                end_line=end,
                chunk_type='function',
                metadata={'name': func['name']},
                token_count=token_count
            )
            chunks.append(chunk)
        
        # Process remaining code (imports, globals, etc.)
        remaining_chunks = self._process_remaining_code(
            code_lines, processed_lines
        )
        chunks.extend(remaining_chunks)
        
        return sorted(chunks, key=lambda c: c.start_line)
    
    def _split_large_block(
        self, 
        code: str, 
        start_line: int, 
        chunk_type: str, 
        metadata: dict
    ) -> List[CodeChunk]:
        """Split a large code block into smaller chunks."""
        chunks = []
        lines = code.split('\n')
        
        current_chunk = []
        current_tokens = 0
        chunk_start = start_line
        
        for i, line in enumerate(lines):
            line_tokens = self.count_tokens(line + '\n')
            
            if current_tokens + line_tokens > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk = CodeChunk(
                    content='\n'.join(current_chunk),
                    start_line=chunk_start,
                    end_line=chunk_start + len(current_chunk) - 1,
                    chunk_type=chunk_type,
                    metadata={**metadata, 'part': len(chunks) + 1},
                    token_count=current_tokens
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk = [line]
                current_tokens = line_tokens
                chunk_start = start_line + i
            else:
                current_chunk.append(line)
                current_tokens += line_tokens
        
        # Add final chunk
        if current_chunk:
            chunk = CodeChunk(
                content='\n'.join(current_chunk),
                start_line=chunk_start,
                end_line=chunk_start + len(current_chunk) - 1,
                chunk_type=chunk_type,
                metadata={**metadata, 'part': len(chunks) + 1},
                token_count=current_tokens
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_class_by_methods(
        self, 
        cls: dict, 
        code_lines: List[str], 
        class_start: int
    ) -> List[CodeChunk]:
        """Split a large class into method-based chunks."""
        chunks = []
        
        # First chunk: class definition + __init__ if exists
        class_header_end = class_start
        init_method = None
        
        # Find class body start
        for i in range(class_start - 1, min(class_start + 10, len(code_lines))):
            if ':' in code_lines[i]:
                class_header_end = i + 2  # Include the line after colon
                break
        
        # Look for __init__ method
        for method in cls['methods']:
            if method == '__init__':
                # Need to find its location
                for i in range(class_header_end - 1, cls['location']['end_line']):
                    if 'def __init__' in code_lines[i]:
                        init_start = i + 1
                        # Find end of __init__
                        init_end = init_start
                        indent_level = len(code_lines[i]) - len(code_lines[i].lstrip())
                        for j in range(i + 1, cls['location']['end_line']):
                            if code_lines[j].strip() and len(code_lines[j]) - len(code_lines[j].lstrip()) <= indent_level:
                                init_end = j
                                break
                        else:
                            init_end = cls['location']['end_line']
                        
                        init_method = (init_start, init_end)
                        break
        
        # Create header chunk with __init__
        if init_method:
            header_lines = (code_lines[class_start - 1:class_header_end - 1] + 
                           code_lines[init_method[0] - 1:init_method[1]])
        else:
            header_lines = code_lines[class_start - 1:class_header_end]
        
        header_content = '\n'.join(header_lines)
        header_tokens = self.count_tokens(header_content)
        
        chunks.append(CodeChunk(
            content=header_content,
            start_line=class_start,
            end_line=init_method[1] if init_method else class_header_end,
            chunk_type='class',
            metadata={'name': cls['name'], 'part': 'header'},
            token_count=header_tokens
        ))
        
        # Create chunks for other methods
        # This is simplified - in production you'd parse method locations properly
        current_chunk_lines = []
        current_tokens = 0
        current_start = None
        
        for i in range(class_header_end, cls['location']['end_line']):
            if init_method and init_method[0] <= i + 1 <= init_method[1]:
                continue  # Skip __init__ as it's in header
                
            line = code_lines[i]
            
            # Check if this is a method definition
            if line.strip().startswith('def ') and current_chunk_lines:
                # Save current chunk
                chunk = CodeChunk(
                    content='\n'.join(current_chunk_lines),
                    start_line=current_start,
                    end_line=i,
                    chunk_type='class',
                    metadata={'name': cls['name'], 'part': 'methods'},
                    token_count=current_tokens
                )
                chunks.append(chunk)
                
                # Start new chunk
                current_chunk_lines = [line]
                current_tokens = self.count_tokens(line + '\n')
                current_start = i + 1
            else:
                if current_start is None:
                    current_start = i + 1
                current_chunk_lines.append(line)
                current_tokens += self.count_tokens(line + '\n')
        
        # Add final chunk
        if current_chunk_lines:
            chunk = CodeChunk(
                content='\n'.join(current_chunk_lines),
                start_line=current_start,
                end_line=cls['location']['end_line'],
                chunk_type='class',
                metadata={'name': cls['name'], 'part': 'methods'},
                token_count=current_tokens
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_module_level_chunks(
        self, 
        code: str, 
        functions: List[dict]
    ) -> List[CodeChunk]:
        """Extract module-level code (imports, globals) not in functions."""
        chunks = []
        code_lines = code.split('\n')
        
        # Find lines that are not in any function
        function_lines = set()
        for func in functions:
            for i in range(func['location']['start_line'], func['location']['end_line'] + 1):
                function_lines.add(i)
        
        # Group consecutive non-function lines
        current_chunk = []
        current_start = None
        
        for i, line in enumerate(code_lines, 1):
            if i not in function_lines and line.strip():
                if current_start is None:
                    current_start = i
                current_chunk.append(line)
            elif current_chunk:
                # End of module-level block
                content = '\n'.join(current_chunk)
                token_count = self.count_tokens(content)
                
                chunk = CodeChunk(
                    content=content,
                    start_line=current_start,
                    end_line=i - 1,
                    chunk_type='block',
                    metadata={'module_level': True},
                    token_count=token_count
                )
                chunks.append(chunk)
                
                current_chunk = []
                current_start = None
        
        # Handle final chunk
        if current_chunk:
            content = '\n'.join(current_chunk)
            token_count = self.count_tokens(content)
            
            chunk = CodeChunk(
                content=content,
                start_line=current_start,
                end_line=len(code_lines),
                chunk_type='block',
                metadata={'module_level': True},
                token_count=token_count
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_non_class_chunks(
        self, 
        code: str, 
        classes: List[dict]
    ) -> List[CodeChunk]:
        """Extract code that's not inside any class."""
        chunks = []
        code_lines = code.split('\n')
        
        # Find lines that are not in any class
        class_lines = set()
        for cls in classes:
            for i in range(cls['location']['start_line'], cls['location']['end_line'] + 1):
                class_lines.add(i)
        
        # Group consecutive non-class lines
        current_chunk = []
        current_start = None
        
        for i, line in enumerate(code_lines, 1):
            if i not in class_lines and line.strip():
                if current_start is None:
                    current_start = i
                current_chunk.append(line)
            elif current_chunk:
                # End of non-class block
                content = '\n'.join(current_chunk)
                token_count = self.count_tokens(content)
                
                if token_count > self.min_chunk_size or len(current_chunk) > 10:
                    chunk = CodeChunk(
                        content=content,
                        start_line=current_start,
                        end_line=i - 1,
                        chunk_type='block',
                        metadata={'non_class': True},
                        token_count=token_count
                    )
                    chunks.append(chunk)
                
                current_chunk = []
                current_start = None
        
        # Handle final chunk
        if current_chunk:
            content = '\n'.join(current_chunk)
            token_count = self.count_tokens(content)
            
            if token_count > self.min_chunk_size or len(current_chunk) > 10:
                chunk = CodeChunk(
                    content=content,
                    start_line=current_start,
                    end_line=len(code_lines),
                    chunk_type='block',
                    metadata={'non_class': True},
                    token_count=token_count
                )
                chunks.append(chunk)
        
        return chunks
    
    def _process_remaining_code(
        self, 
        code_lines: List[str], 
        processed_lines: set
    ) -> List[CodeChunk]:
        """Process code lines that haven't been chunked yet."""
        chunks = []
        current_chunk = []
        current_start = None
        current_tokens = 0
        
        for i, line in enumerate(code_lines, 1):
            if i not in processed_lines:
                if current_start is None:
                    current_start = i
                
                line_tokens = self.count_tokens(line + '\n')
                
                # Check if adding this line would exceed max size
                if current_tokens + line_tokens > self.max_chunk_size and current_chunk:
                    # Save current chunk
                    content = '\n'.join(current_chunk)
                    chunk = CodeChunk(
                        content=content,
                        start_line=current_start,
                        end_line=i - 1,
                        chunk_type='block',
                        metadata={'mixed': True},
                        token_count=current_tokens
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk
                    current_chunk = [line]
                    current_tokens = line_tokens
                    current_start = i
                else:
                    current_chunk.append(line)
                    current_tokens += line_tokens
            elif current_chunk:
                # End of unprocessed block
                if current_tokens >= self.min_chunk_size:
                    content = '\n'.join(current_chunk)
                    chunk = CodeChunk(
                        content=content,
                        start_line=current_start,
                        end_line=i - 1,
                        chunk_type='block',
                        metadata={'mixed': True},
                        token_count=current_tokens
                    )
                    chunks.append(chunk)
                
                current_chunk = []
                current_start = None
                current_tokens = 0
        
        # Handle final chunk
        if current_chunk and current_tokens >= self.min_chunk_size:
            content = '\n'.join(current_chunk)
            chunk = CodeChunk(
                content=content,
                start_line=current_start,
                end_line=len(code_lines),
                chunk_type='block',
                metadata={'mixed': True},
                token_count=current_tokens
            )
            chunks.append(chunk)
        
        return chunks
    
    def _is_function_in_class(self, func: dict, cls: dict) -> bool:
        """Check if a function is inside a class."""
        func_start = func['location']['start_line']
        func_end = func['location']['end_line']
        class_start = cls['location']['start_line']
        class_end = cls['location']['end_line']
        
        return class_start <= func_start <= func_end <= class_end