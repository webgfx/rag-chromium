"""
Data preprocessing utilities for the RAG system.
Handles cleaning, normalization, and preparation of various data types.
"""

import re
import html
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
from tqdm import tqdm

from ..core.config import get_config
from ..core.logger import setup_logger


@dataclass
class ProcessedDocument:
    """Structure for a processed document."""
    id: str
    content: str
    metadata: Dict[str, Any]
    document_type: str
    language: Optional[str] = None
    preprocessed_content: Optional[str] = None


class DataPreprocessor:
    """
    Advanced data preprocessor for various types of content.
    Handles code, commit messages, documentation, and other text types.
    """
    
    def __init__(self):
        """Initialize the data preprocessor."""
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.DataPreprocessor")
        
        # Compile regex patterns for efficiency
        self._patterns = self._compile_patterns()
        
        # Language detection patterns
        self._language_patterns = {
            'python': [r'\.py$', r'python', r'#!/usr/bin/env python', r'#!/usr/bin/python'],
            'cpp': [r'\.(cpp|cc|cxx|c\+\+)$', r'#include\s*<', r'std::', r'namespace\s+\w+'],
            'c': [r'\.(c|h)$', r'#include\s*"', r'#include\s*<', r'int\s+main\s*\('],
            'javascript': [r'\.(js|jsx)$', r'function\s*\(', r'var\s+\w+', r'const\s+\w+', r'let\s+\w+'],
            'typescript': [r'\.(ts|tsx)$', r'interface\s+\w+', r'type\s+\w+', r': \w+\s*='],
            'java': [r'\.java$', r'public\s+class', r'import\s+java', r'package\s+\w+'],
            'go': [r'\.go$', r'package\s+\w+', r'import\s*\(', r'func\s+\w+'],
            'rust': [r'\.rs$', r'fn\s+\w+', r'let\s+mut', r'use\s+std::'],
            'html': [r'\.(html|htm)$', r'<html', r'<!DOCTYPE', r'<div', r'<span'],
            'css': [r'\.css$', r'\{[^}]*\}', r'@media', r'@import'],
            'markdown': [r'\.(md|markdown)$', r'^#{1,6}\s', r'\*\*.*\*\*', r'\[.*\]\(.*\)'],
            'yaml': [r'\.(yml|yaml)$', r'^\s*-\s', r'^\w+:\s*$', r'^\s*\w+:\s*[^\s]'],
            'json': [r'\.json$', r'^\s*\{', r'"\w+"\s*:', r'\[\s*\{'],
            'xml': [r'\.xml$', r'<\?xml', r'<\w+[^>]*>', r'</\w+>'],
            'shell': [r'\.(sh|bash)$', r'#!/bin/(ba)?sh', r'if\s*\[', r'for\s+\w+\s+in'],
            'sql': [r'\.sql$', r'SELECT\s+', r'FROM\s+\w+', r'WHERE\s+', r'INSERT\s+INTO']
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile commonly used regex patterns."""
        return {
            'email': re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'hash': re.compile(r'\\b[0-9a-f]{7,40}\\b'),  # Git hashes
            'bug_id': re.compile(r'(?:bug|issue|ticket)\\s*[#:]?\\s*(\\d+)', re.IGNORECASE),
            'code_block': re.compile(r'```[\\s\\S]*?```|`[^`]+`'),
            'html_tags': re.compile(r'<[^<]+?>'),
            'multiple_spaces': re.compile(r'\\s{2,}'),
            'multiple_newlines': re.compile(r'\\n{3,}'),
            'leading_trailing_spaces': re.compile(r'^\\s+|\\s+$', re.MULTILINE)
        }
    
    def detect_language(self, content: str, file_path: Optional[str] = None) -> Optional[str]:
        """
        Detect the programming language of the content.
        
        Args:
            content: Text content to analyze
            file_path: Optional file path for extension-based detection
        
        Returns:
            Detected language or None
        """
        if file_path:
            file_path = str(file_path).lower()
        
        for language, patterns in self._language_patterns.items():
            score = 0
            
            for pattern in patterns:
                # Check file extension
                if file_path and pattern.startswith('\\\\') and pattern.endswith('$'):
                    if re.search(pattern, file_path):
                        score += 10
                
                # Check content patterns
                elif re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                    score += 1
            
            # If we have a strong match, return it
            if score >= 5:
                return language
            
            # For simple extension matches, lower threshold
            if file_path and score >= 1:
                return language
        
        return None
    
    def clean_commit_message(self, message: str) -> str:
        """
        Clean and normalize commit messages.
        
        Args:
            message: Raw commit message
        
        Returns:
            Cleaned commit message
        """
        if not message:
            return ""
        
        # Remove HTML entities
        message = html.unescape(message)
        
        # Remove URLs (keep domain for context)
        message = re.sub(r'https?://[^\\s]+', '[URL]', message)
        
        # Normalize bug/issue references
        message = re.sub(self._patterns['bug_id'], r'Bug \\1', message)
        
        # Clean up git hashes
        message = re.sub(self._patterns['hash'], '[HASH]', message)
        
        # Remove excessive whitespace
        message = self._patterns['multiple_spaces'].sub(' ', message)
        message = self._patterns['multiple_newlines'].sub('\\n\\n', message)
        message = self._patterns['leading_trailing_spaces'].sub('', message)
        
        return message.strip()
    
    def clean_code_content(self, content: str, language: Optional[str] = None) -> str:
        """
        Clean and normalize code content.
        
        Args:
            content: Raw code content
            language: Programming language (if known)
        
        Returns:
            Cleaned code content
        """
        if not content:
            return ""
        
        # Remove excessive blank lines but preserve code structure
        content = re.sub(r'\\n{4,}', '\\n\\n\\n', content)
        
        # Normalize indentation (convert tabs to spaces)
        content = content.expandtabs(4)
        
        # Remove trailing whitespace from lines
        content = re.sub(r'[ \\t]+$', '', content, flags=re.MULTILINE)
        
        # Language-specific cleaning
        if language == 'python':
            # Remove excessive blank lines in Python files
            content = re.sub(r'\\n{3,}(def |class |import |from )', r'\\n\\n\\1', content)
        
        elif language in ['cpp', 'c', 'java']:
            # Normalize brace style somewhat
            content = re.sub(r'\\{\\s*\\n\\s*\\n', '{\\n', content)
        
        return content.strip()
    
    def clean_diff_content(self, diff: str) -> str:
        """
        Clean and normalize diff content.
        
        Args:
            diff: Raw diff content
        
        Returns:
            Cleaned diff content
        """
        if not diff:
            return ""
        
        # Remove binary file indicators
        diff = re.sub(r'^Binary files? .* differ$', '[BINARY FILE]', diff, flags=re.MULTILINE)
        
        # Clean up excessive context
        diff = re.sub(r'^(@@ -\\d+,\\d+ \\+\\d+,\\d+ @@).*$', r'\\1', diff, flags=re.MULTILINE)
        
        # Remove git index lines
        diff = re.sub(r'^index [0-9a-f]+\\.\\.[0-9a-f]+.*$', '', diff, flags=re.MULTILINE)
        
        # Remove mode changes
        diff = re.sub(r'^(old|new) mode \\d+$', '', diff, flags=re.MULTILINE)
        
        # Clean up excessive newlines
        diff = self._patterns['multiple_newlines'].sub('\\n\\n', diff)
        
        return diff.strip()
    
    def extract_metadata(self, content: str, document_type: str) -> Dict[str, Any]:
        """
        Extract metadata from content.
        
        Args:
            content: Document content
            document_type: Type of document (commit, code, diff, etc.)
        
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {
            'length': len(content),
            'lines': len(content.split('\\n')),
            'words': len(content.split()) if content else 0,
            'document_type': document_type
        }
        
        if document_type == 'commit':
            # Extract commit-specific metadata
            emails = self._patterns['email'].findall(content)
            metadata['emails'] = list(set(emails))
            metadata['has_bug_reference'] = bool(self._patterns['bug_id'].search(content))
            metadata['has_url'] = bool(self._patterns['url'].search(content))
            
        elif document_type == 'code':
            # Extract code-specific metadata
            language = self.detect_language(content)
            metadata['language'] = language
            metadata['has_comments'] = self._has_comments(content, language)
            metadata['complexity_score'] = self._estimate_complexity(content, language)
            
        elif document_type == 'diff':
            # Extract diff-specific metadata
            additions = len(re.findall(r'^\\+(?!\\+\\+)', content, re.MULTILINE))
            deletions = len(re.findall(r'^-(?!---)', content, re.MULTILINE))
            metadata['additions'] = additions
            metadata['deletions'] = deletions
            metadata['total_changes'] = additions + deletions
            metadata['files_affected'] = len(re.findall(r'^\\+\\+\\+ ', content, re.MULTILINE))
        
        return metadata
    
    def _has_comments(self, content: str, language: Optional[str]) -> bool:
        """Check if code content has comments."""
        if not content or not language:
            return False
        
        comment_patterns = {
            'python': [r'#.*$', r'"""[\\s\\S]*?"""', r"'''[\\s\\S]*?'''"],
            'cpp': [r'//.*$', r'/\\*[\\s\\S]*?\\*/'],
            'c': [r'//.*$', r'/\\*[\\s\\S]*?\\*/'],
            'javascript': [r'//.*$', r'/\\*[\\s\\S]*?\\*/'],
            'java': [r'//.*$', r'/\\*[\\s\\S]*?\\*/'],
            'go': [r'//.*$', r'/\\*[\\s\\S]*?\\*/'],
            'rust': [r'//.*$', r'/\\*[\\s\\S]*?\\*/'],
            'shell': [r'#.*$'],
            'sql': [r'--.*$', r'/\\*[\\s\\S]*?\\*/']
        }
        
        patterns = comment_patterns.get(language, [])
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                return True
        
        return False
    
    def _estimate_complexity(self, content: str, language: Optional[str]) -> int:
        """Estimate code complexity score."""
        if not content or not language:
            return 0
        
        score = 0
        
        # Common complexity indicators
        complexity_patterns = {
            'conditionals': [r'\\bif\\b', r'\\belse\\b', r'\\belif\\b', r'\\bswitch\\b', r'\\bcase\\b'],
            'loops': [r'\\bfor\\b', r'\\bwhile\\b', r'\\bdo\\b'],
            'functions': [r'\\bdef\\b', r'\\bfunction\\b', r'\\bfunc\\b', r'\\bfn\\b'],
            'classes': [r'\\bclass\\b', r'\\bstruct\\b', r'\\benum\\b'],
            'exceptions': [r'\\btry\\b', r'\\bcatch\\b', r'\\bexcept\\b', r'\\bfinally\\b']
        }
        
        for category, patterns in complexity_patterns.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, content, re.IGNORECASE))
                score += matches
        
        # Nesting depth (simplified)
        nesting_score = content.count('{') + content.count('(') + content.count('[')
        score += nesting_score // 10
        
        return min(score, 100)  # Cap at 100
    
    def process_document(
        self,
        content: str,
        document_id: str,
        document_type: str,
        file_path: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        Process a single document.
        
        Args:
            content: Raw document content
            document_id: Unique document identifier
            document_type: Type of document
            file_path: Optional file path
            additional_metadata: Additional metadata to include
        
        Returns:
            ProcessedDocument object
        """
        # Clean content based on type
        if document_type == 'commit':
            preprocessed_content = self.clean_commit_message(content)
        elif document_type == 'code':
            language = self.detect_language(content, file_path)
            preprocessed_content = self.clean_code_content(content, language)
        elif document_type == 'diff':
            preprocessed_content = self.clean_diff_content(content)
        else:
            # Generic text cleaning
            preprocessed_content = self._patterns['multiple_spaces'].sub(' ', content)
            preprocessed_content = self._patterns['multiple_newlines'].sub('\\n\\n', preprocessed_content)
            preprocessed_content = preprocessed_content.strip()
        
        # Extract metadata
        metadata = self.extract_metadata(content, document_type)
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Detect language if not already done
        language = metadata.get('language') or self.detect_language(content, file_path)
        
        return ProcessedDocument(
            id=document_id,
            content=content,
            metadata=metadata,
            document_type=document_type,
            language=language,
            preprocessed_content=preprocessed_content
        )
    
    def process_batch(
        self,
        documents: List[Tuple[str, str, str]],  # (content, id, type)
        batch_size: int = 100
    ) -> List[ProcessedDocument]:
        """
        Process multiple documents in batches.
        
        Args:
            documents: List of (content, document_id, document_type) tuples
            batch_size: Processing batch size
        
        Returns:
            List of ProcessedDocument objects
        """
        processed_docs = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Processing documents"):
            batch = documents[i:i + batch_size]
            
            for content, doc_id, doc_type in batch:
                try:
                    processed_doc = self.process_document(content, doc_id, doc_type)
                    processed_docs.append(processed_doc)
                except Exception as e:
                    self.logger.error(f"Failed to process document {doc_id}: {e}")
                    continue
        
        self.logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def filter_documents(
        self,
        documents: List[ProcessedDocument],
        min_length: int = 10,
        max_length: int = 50000,
        allowed_languages: Optional[List[str]] = None,
        allowed_types: Optional[List[str]] = None
    ) -> List[ProcessedDocument]:
        """
        Filter documents based on criteria.
        
        Args:
            documents: List of ProcessedDocument objects
            min_length: Minimum content length
            max_length: Maximum content length
            allowed_languages: Allowed programming languages
            allowed_types: Allowed document types
        
        Returns:
            Filtered list of documents
        """
        filtered = []
        
        for doc in documents:
            # Length filter
            if len(doc.preprocessed_content or doc.content) < min_length:
                continue
            if len(doc.preprocessed_content or doc.content) > max_length:
                continue
            
            # Language filter
            if allowed_languages and doc.language and doc.language not in allowed_languages:
                continue
            
            # Type filter
            if allowed_types and doc.document_type not in allowed_types:
                continue
            
            filtered.append(doc)
        
        self.logger.info(f"Filtered {len(documents)} documents to {len(filtered)}")
        return filtered