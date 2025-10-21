#!/usr/bin/env python3
"""
Vector database module for advanced RAG system.
Provides high-performance vector storage, indexing, and retrieval.
"""

# Switch to Qdrant for better scalability and performance
from .qdrant_database import VectorDatabase, VectorDocument, SearchResult

__all__ = [
    'VectorDatabase',
    'VectorDocument', 
    'SearchResult'
]
