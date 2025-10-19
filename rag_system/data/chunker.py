"""
Advanced text and code chunking utilities for the RAG system.
Implements intelligent chunking strategies for different content types.
"""

import re
import ast
import logging
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
from abc import ABC, abstractmethod
import tiktoken

from ..core.config import get_config
from ..core.logger import setup_logger


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    id: str
    content: str
    start_idx: int
    end_idx: int
    metadata: Dict[str, Any]
    chunk_type: str
    parent_id: str
    language: Optional[str] = None


class BaseChunker(ABC):
    """Base class for all chunking strategies."""
    
    def __init__(self, chunk_size: int = 2048, overlap: int = 256):
        """
        Initialize base chunker.
        
        Args:
            chunk_size: Target chunk size in tokens/characters
            overlap: Number of tokens/characters to overlap between chunks
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.{self.__class__.__name__}")
        
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        # Initialize tokenizer for token-based chunking
        try:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer: {e}. Using character-based chunking.")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Rough approximation: ~4 characters per token
            return len(text) // 4
    
    @abstractmethod
    def chunk(self, content: str, document_id: str, **kwargs) -> List[Chunk]:
        """
        Chunk content into smaller pieces.
        
        Args:
            content: Text content to chunk
            document_id: Unique identifier for the source document
            **kwargs: Additional parameters specific to chunker type
        
        Returns:
            List of Chunk objects
        """
        pass
    
    def _create_chunk(
        self,
        content: str,
        chunk_id: str,
        start_idx: int,
        end_idx: int,
        parent_id: str,
        chunk_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        language: Optional[str] = None
    ) -> Chunk:
        """Create a Chunk object with metadata."""
        if metadata is None:
            metadata = {}
        
        # Add standard metadata
        metadata.update({
            'token_count': self.count_tokens(content),
            'char_count': len(content),
            'line_count': len(content.split('\\n')),
            'start_idx': start_idx,
            'end_idx': end_idx
        })
        
        return Chunk(
            id=chunk_id,
            content=content,
            start_idx=start_idx,
            end_idx=end_idx,
            metadata=metadata,
            chunk_type=chunk_type,
            parent_id=parent_id,
            language=language
        )


class TextChunker(BaseChunker):
    """
    Intelligent text chunker that preserves sentence and paragraph boundaries.
    """
    
    def __init__(self, chunk_size: int = 2048, overlap: int = 256):
        super().__init__(chunk_size, overlap)
        
        # Sentence boundary patterns
        self.sentence_patterns = [
            r'(?<=[.!?])\s+(?=[A-Z])',  # Standard sentence endings
            r'(?<=\.)\s+(?=[A-Z][a-z])',  # Period followed by capitalized word
            r'(?<=[.!?]["\'])\s+(?=[A-Z])',  # Sentence ending with quote
            r'\n\s*\n',  # Paragraph breaks
        ]
        
        # Compile patterns
        self.sentence_regex = re.compile('|'.join(self.sentence_patterns))
    
    def chunk(self, content: str, document_id: str, **kwargs) -> List[Chunk]:
        """
        Chunk text content preserving sentence boundaries.
        
        Args:
            content: Text content to chunk
            document_id: Source document ID
            **kwargs: Additional parameters
        
        Returns:
            List of text chunks
        """
        if not content.strip():
            return []
        
        chunks = []
        
        # Split into sentences/paragraphs
        sentences = self._split_into_sentences(content)
        
        current_chunk = ""
        current_start = 0
        chunk_count = 0
        
        for i, sentence in enumerate(sentences):
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + (" " if current_chunk else "") + sentence
            
            if self.count_tokens(potential_chunk) > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_end = current_start + len(current_chunk)
                chunk_id = f"{document_id}_chunk_{chunk_count}"
                
                chunk = self._create_chunk(
                    content=current_chunk.strip(),
                    chunk_id=chunk_id,
                    start_idx=current_start,
                    end_idx=chunk_end,
                    parent_id=document_id,
                    chunk_type="text",
                    metadata={'sentence_count': len(current_chunk.split('.'))}
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap > 0:
                    overlap_text = self._get_overlap_text(current_chunk, self.overlap)
                    current_chunk = overlap_text + " " + sentence
                    current_start = chunk_end - len(overlap_text)
                else:
                    current_chunk = sentence
                    current_start = chunk_end
                
                chunk_count += 1
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk_end = current_start + len(current_chunk)
            chunk_id = f"{document_id}_chunk_{chunk_count}"
            
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                chunk_id=chunk_id,
                start_idx=current_start,
                end_idx=chunk_end,
                parent_id=document_id,
                chunk_type="text",
                metadata={'sentence_count': len(current_chunk.split('.'))}
            )
            chunks.append(chunk)
        
        self.logger.debug(f"Created {len(chunks)} text chunks for document {document_id}")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        # First split by paragraph breaks
        paragraphs = text.split('\\n\\n')
        sentences = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # Split paragraph into sentences
            para_sentences = self.sentence_regex.split(paragraph)
            para_sentences = [s.strip() for s in para_sentences if s.strip()]
            sentences.extend(para_sentences)
        
        return sentences
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of current chunk."""
        if self.tokenizer:
            tokens = self.tokenizer.encode(text)
            if len(tokens) <= overlap_size:
                return text
            
            overlap_tokens = tokens[-overlap_size:]
            return self.tokenizer.decode(overlap_tokens)
        else:
            # Character-based overlap
            return text[-overlap_size * 4:] if len(text) > overlap_size * 4 else text


class CodeChunker(BaseChunker):
    """
    Advanced code chunker that preserves semantic structure.
    Attempts to keep functions, classes, and logical blocks together.
    """
    
    def __init__(self, chunk_size: int = 2048, overlap: int = 256):
        super().__init__(chunk_size, overlap)
        
        # Language-specific patterns for semantic elements
        self.semantic_patterns = {
            'python': {
                'function': r'^\\s*def\\s+\\w+.*?:',
                'class': r'^\\s*class\\s+\\w+.*?:',
                'method': r'^\\s+def\\s+\\w+.*?:',
                'import': r'^\\s*(import|from)\\s+.*$',
                'comment_block': r'^\\s*"""[\\s\\S]*?"""',
                'decorator': r'^\\s*@\\w+.*$'
            },
            'cpp': {
                'function': r'^\\s*\\w+\\s+\\w+\\s*\\([^)]*\\)\\s*\\{',
                'class': r'^\\s*class\\s+\\w+.*?\\{',
                'struct': r'^\\s*struct\\s+\\w+.*?\\{',
                'namespace': r'^\\s*namespace\\s+\\w+.*?\\{',
                'include': r'^\\s*#include\\s*[<"].*[>"]',
                'define': r'^\\s*#define\\s+\\w+.*$'
            },
            'javascript': {
                'function': r'^\\s*function\\s+\\w+.*?\\{',
                'arrow_function': r'^\\s*\\w+\\s*=\\s*\\([^)]*\\)\\s*=>',
                'class': r'^\\s*class\\s+\\w+.*?\\{',
                'method': r'^\\s*\\w+\\s*\\([^)]*\\)\\s*\\{',
                'import': r'^\\s*(import|export)\\s+.*$',
                'comment_block': r'/\\*[\\s\\S]*?\\*/'
            },
            'java': {
                'class': r'^\\s*public\\s+class\\s+\\w+.*?\\{',
                'method': r'^\\s*(public|private|protected)\\s+.*?\\w+\\s*\\([^)]*\\)\\s*\\{',
                'interface': r'^\\s*public\\s+interface\\s+\\w+.*?\\{',
                'import': r'^\\s*import\\s+.*?;',
                'package': r'^\\s*package\\s+.*?;'
            }
        }
    
    def chunk(self, content: str, document_id: str, language: str = None, **kwargs) -> List[Chunk]:
        """
        Chunk code content preserving semantic structure.
        
        Args:
            content: Code content to chunk
            document_id: Source document ID
            language: Programming language
            **kwargs: Additional parameters
        
        Returns:
            List of code chunks
        """
        if not content.strip():
            return []
        
        # Try to detect language if not provided
        if not language:
            language = self._detect_language(content)
        
        chunks = []
        
        # Use language-specific chunking if available
        if language and language in self.semantic_patterns:
            chunks = self._chunk_by_semantic_structure(content, document_id, language)
        else:
            # Fall back to generic code chunking
            chunks = self._chunk_by_lines(content, document_id, language)
        
        self.logger.debug(f"Created {len(chunks)} code chunks for document {document_id} (language: {language})")
        return chunks
    
    def _detect_language(self, content: str) -> Optional[str]:
        """Detect programming language from content patterns."""
        for language, patterns in self.semantic_patterns.items():
            score = 0
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, content, re.MULTILINE):
                    score += 1
            
            # If we find multiple patterns, it's likely this language
            if score >= 2:
                return language
        
        return None
    
    def _chunk_by_semantic_structure(self, content: str, document_id: str, language: str) -> List[Chunk]:
        """Chunk code by semantic structure (functions, classes, etc.)."""
        chunks = []
        lines = content.split('\\n')
        patterns = self.semantic_patterns[language]
        
        current_chunk_lines = []
        current_start_line = 0
        chunk_count = 0
        brace_depth = 0
        in_function = False
        function_start = 0
        
        for i, line in enumerate(lines):
            current_chunk_lines.append(line)
            
            # Track brace depth for languages that use braces
            if language in ['cpp', 'javascript', 'java']:
                brace_depth += line.count('{') - line.count('}')
            
            # Check if we're starting a new semantic element
            is_semantic_start = any(re.match(pattern, line) for pattern in patterns.values())
            
            if is_semantic_start and not in_function:
                in_function = True
                function_start = i
            
            # Check if we should create a chunk
            should_chunk = False
            
            # End of function/class (brace depth returns to 0)
            if in_function and brace_depth == 0 and language in ['cpp', 'javascript', 'java']:
                should_chunk = True
                in_function = False
            
            # For Python, use indentation
            elif language == 'python' and in_function:
                if i < len(lines) - 1:
                    current_indent = len(line) - len(line.lstrip())
                    next_line = lines[i + 1].strip()
                    if next_line and not next_line.startswith('#'):
                        next_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                        if next_indent <= current_indent and current_indent > 0:
                            should_chunk = True
                            in_function = False
            
            # Size-based chunking as fallback
            chunk_content = '\\n'.join(current_chunk_lines)
            if self.count_tokens(chunk_content) > self.chunk_size:
                should_chunk = True
            
            if should_chunk and current_chunk_lines:
                chunk_text = '\\n'.join(current_chunk_lines[:-1] if not in_function else current_chunk_lines)
                start_char = sum(len(lines[j]) + 1 for j in range(current_start_line, i))
                end_char = start_char + len(chunk_text)
                
                chunk = self._create_chunk(
                    content=chunk_text,
                    chunk_id=f"{document_id}_chunk_{chunk_count}",
                    start_idx=start_char,
                    end_idx=end_char,
                    parent_id=document_id,
                    chunk_type="code",
                    metadata={
                        'line_start': current_start_line + 1,
                        'line_end': i + (0 if in_function else 1),
                        'has_function': self._contains_function(chunk_text, language),
                        'has_class': self._contains_class(chunk_text, language)
                    },
                    language=language
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap if needed
                if self.overlap > 0 and not in_function:
                    overlap_lines = max(1, self.overlap // 50)  # Rough estimate
                    current_chunk_lines = current_chunk_lines[-overlap_lines:]
                    current_start_line = i - overlap_lines + 1
                else:
                    current_chunk_lines = [line] if in_function else []
                    current_start_line = i
                
                chunk_count += 1
        
        # Add final chunk
        if current_chunk_lines:
            chunk_text = '\\n'.join(current_chunk_lines)
            start_char = sum(len(lines[j]) + 1 for j in range(current_start_line))
            
            chunk = self._create_chunk(
                content=chunk_text,
                chunk_id=f"{document_id}_chunk_{chunk_count}",
                start_idx=start_char,
                end_idx=start_char + len(chunk_text),
                parent_id=document_id,
                chunk_type="code",
                metadata={
                    'line_start': current_start_line + 1,
                    'line_end': len(lines),
                    'has_function': self._contains_function(chunk_text, language),
                    'has_class': self._contains_class(chunk_text, language)
                },
                language=language
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_by_lines(self, content: str, document_id: str, language: Optional[str]) -> List[Chunk]:
        """Fallback chunking by lines with smart boundaries."""
        chunks = []
        lines = content.split('\\n')
        
        lines_per_chunk = max(10, self.chunk_size // 80)  # Rough estimate
        overlap_lines = max(1, self.overlap // 80)
        
        chunk_count = 0
        
        for i in range(0, len(lines), lines_per_chunk - overlap_lines):
            chunk_lines = lines[i:i + lines_per_chunk]
            chunk_text = '\\n'.join(chunk_lines)
            
            if not chunk_text.strip():
                continue
            
            start_char = sum(len(lines[j]) + 1 for j in range(i))
            
            chunk = self._create_chunk(
                content=chunk_text,
                chunk_id=f"{document_id}_chunk_{chunk_count}",
                start_idx=start_char,
                end_idx=start_char + len(chunk_text),
                parent_id=document_id,
                chunk_type="code",
                metadata={
                    'line_start': i + 1,
                    'line_end': min(i + lines_per_chunk, len(lines)),
                    'chunking_method': 'line_based'
                },
                language=language
            )
            chunks.append(chunk)
            chunk_count += 1
        
        return chunks
    
    def _contains_function(self, content: str, language: str) -> bool:
        """Check if content contains function definitions."""
        if language not in self.semantic_patterns:
            return False
        
        patterns = self.semantic_patterns[language]
        function_patterns = [p for name, p in patterns.items() if 'function' in name or 'method' in name]
        
        return any(re.search(pattern, content, re.MULTILINE) for pattern in function_patterns)
    
    def _contains_class(self, content: str, language: str) -> bool:
        """Check if content contains class definitions."""
        if language not in self.semantic_patterns:
            return False
        
        patterns = self.semantic_patterns[language]
        class_patterns = [p for name, p in patterns.items() if 'class' in name or 'struct' in name]
        
        return any(re.search(pattern, content, re.MULTILINE) for pattern in class_patterns)


class DiffChunker(BaseChunker):
    """
    Specialized chunker for git diff content.
    Preserves diff structure while creating manageable chunks.
    """
    
    def chunk(self, content: str, document_id: str, **kwargs) -> List[Chunk]:
        """
        Chunk diff content by file and hunk boundaries.
        
        Args:
            content: Diff content to chunk
            document_id: Source document ID
            **kwargs: Additional parameters
        
        Returns:
            List of diff chunks
        """
        if not content.strip():
            return []
        
        chunks = []
        
        # Split diff by file boundaries
        file_sections = re.split(r'^diff --git', content, flags=re.MULTILINE)
        
        chunk_count = 0
        
        for i, section in enumerate(file_sections):
            if not section.strip():
                continue
            
            # Add back the diff marker if it was split off
            if i > 0:
                section = "diff --git" + section
            
            # Further split by hunks if section is too large
            if self.count_tokens(section) > self.chunk_size:
                hunk_chunks = self._chunk_by_hunks(section, document_id, chunk_count)
                chunks.extend(hunk_chunks)
                chunk_count += len(hunk_chunks)
            else:
                # Create single chunk for this file
                chunk = self._create_chunk(
                    content=section.strip(),
                    chunk_id=f"{document_id}_diff_{chunk_count}",
                    start_idx=0,  # Would need full diff parsing for accurate indices
                    end_idx=len(section),
                    parent_id=document_id,
                    chunk_type="diff",
                    metadata=self._extract_diff_metadata(section)
                )
                chunks.append(chunk)
                chunk_count += 1
        
        self.logger.debug(f"Created {len(chunks)} diff chunks for document {document_id}")
        return chunks
    
    def _chunk_by_hunks(self, diff_section: str, document_id: str, start_count: int) -> List[Chunk]:
        """Chunk a diff section by individual hunks."""
        chunks = []
        
        # Split by hunk headers
        hunks = re.split(r'^@@.*?@@', diff_section, flags=re.MULTILINE)
        
        # Keep file header with first hunk
        file_header = hunks[0] if hunks else ""
        
        for i, hunk in enumerate(hunks[1:], 1):  # Skip file header
            hunk_content = file_header + "@@...@@\\n" + hunk.strip()
            
            chunk = self._create_chunk(
                content=hunk_content,
                chunk_id=f"{document_id}_diff_{start_count + i}",
                start_idx=0,
                end_idx=len(hunk_content),
                parent_id=document_id,
                chunk_type="diff_hunk",
                metadata=self._extract_hunk_metadata(hunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _extract_diff_metadata(self, diff_content: str) -> Dict[str, Any]:
        """Extract metadata from diff content."""
        metadata = {}
        
        # Extract file paths
        file_match = re.search(r'diff --git a/(.*?) b/(.*?)$', diff_content, re.MULTILINE)
        if file_match:
            metadata['old_file'] = file_match.group(1)
            metadata['new_file'] = file_match.group(2)
        
        # Count changes
        additions = len(re.findall(r'^\\+(?!\\+\\+)', diff_content, re.MULTILINE))
        deletions = len(re.findall(r'^-(?!---)', diff_content, re.MULTILINE))
        
        metadata.update({
            'additions': additions,
            'deletions': deletions,
            'total_changes': additions + deletions,
            'hunks': len(re.findall(r'^@@.*?@@', diff_content, re.MULTILINE))
        })
        
        return metadata
    
    def _extract_hunk_metadata(self, hunk_content: str) -> Dict[str, Any]:
        """Extract metadata from a single hunk."""
        additions = len(re.findall(r'^\\+(?!\\+\\+)', hunk_content, re.MULTILINE))
        deletions = len(re.findall(r'^-(?!---)', hunk_content, re.MULTILINE))
        
        return {
            'additions': additions,
            'deletions': deletions,
            'total_changes': additions + deletions,
            'context_lines': len(re.findall(r'^\\s', hunk_content, re.MULTILINE))
        }