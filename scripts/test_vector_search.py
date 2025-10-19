#!/usr/bin/env python3
"""
Quick test script for vector database search using pre-computed embeddings.
"""

import json
from pathlib import Path

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.vector.database import VectorDatabase
from rag_system.embeddings.generator import EmbeddingGenerator


def test_search_with_embeddings():
    """Test search using embeddings from our embedding generator."""
    logger = setup_logger(__name__)
    
    # Initialize components
    vector_db = VectorDatabase(collection_name="chromium_embeddings")
    embedding_generator = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
    
    # Test queries
    test_queries = [
        "bug fix memory leak",
        "performance optimization",
        "rendering pipeline",
        "crash fix",
        "security vulnerability"
    ]
    
    logger.info("Testing vector database search with generated embeddings...")
    
    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        
        # Generate embedding for the query
        query_embeddings = embedding_generator.encode_texts([query])
        query_embedding = query_embeddings[0].tolist()
        
        # Search using the embedding
        results = vector_db.search(
            query=query_embedding,
            n_results=3,
            include_embeddings=False
        )
        
        logger.info(f"Found {len(results)} results")
        
        for i, result in enumerate(results):
            logger.info(f"  {i+1}. Score: {result.score:.4f}")
            logger.info(f"     File: {result.document.metadata.get('file_path', 'N/A')}")
            logger.info(f"     Type: {result.document.metadata.get('chunk_type', 'N/A')}")
            logger.info(f"     Content: {result.document.content[:100]}...")
            
    logger.info("\nVector database search test completed!")


if __name__ == "__main__":
    test_search_with_embeddings()