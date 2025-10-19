"""
Data ingestion and preprocessing module for the RAG system.
Handles extraction and processing of Chromium repository data.
"""

from .chromium import ChromiumDataExtractor
from .preprocessor import DataPreprocessor
from .chunker import CodeChunker, TextChunker

__all__ = [
    "ChromiumDataExtractor",
    "DataPreprocessor", 
    "CodeChunker",
    "TextChunker"
]