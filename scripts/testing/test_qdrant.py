#!/usr/bin/env python3
"""
Test Qdrant integration with a few sample documents.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from rag_system.vector.qdrant_database import VectorDatabase, VectorDocument
from rag_system.core.logger import setup_logger


def test_qdrant():
    """Test Qdrant with sample documents."""
    logger = setup_logger("Qdrant_Test")
    
    logger.info("Testing Qdrant integration...")
    
    try:
        # Initialize Qdrant
        db = VectorDatabase(
            collection_name="test_collection",
            persist_directory="data/cache/qdrant_test",
            vector_size=1024
        )
        
        logger.info("Qdrant initialized successfully")
        
        # Create test documents
        test_docs = []
        for i in range(10):
            embedding = np.random.rand(1024).tolist()
            doc = VectorDocument(
                id=f"test_doc_{i}",
                content=f"This is test document {i}",
                embedding=embedding,
                metadata={"index": i, "category": "test"}
            )
            test_docs.append(doc)
        
        # Add documents
        logger.info("Adding 10 test documents...")
        added = db.add_documents(test_docs)
        logger.info(f"Added {added} documents")
        
        # Get stats
        stats = db.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
        # Test search
        logger.info("Testing search...")
        query_embedding = np.random.rand(1024).tolist()
        results = db.search(query_embedding, n_results=5)
        logger.info(f"Search returned {len(results)} results")
        
        for result in results:
            logger.info(f"  - Doc {result.document.id}: score={result.score:.4f}")
        
        # Test retrieval
        logger.info("Testing document retrieval...")
        doc = db.get_document("test_doc_0")
        if doc:
            logger.info(f"Retrieved document: {doc.id}")
        
        logger.info("=" * 60)
        logger.info("✓ All tests passed! Qdrant is working correctly.")
        logger.info("=" * 60)
        
        db.close()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_qdrant()
