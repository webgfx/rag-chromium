#!/usr/bin/env python3
"""
Vector database ingestion script for processed Chromium data.
Converts processed chunks and embeddings into vector database format.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.vector.database import VectorDatabase, VectorDocument


def load_embedding_data(file_path: Path) -> Dict[str, Any]:
    """Load embedding data from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_vector_documents(embedding_data: Dict[str, Any]) -> List[VectorDocument]:
    """Convert embedding data to vector documents."""
    documents = []
    
    chunks = embedding_data.get('chunks', [])
    embeddings = embedding_data.get('embeddings', [])
    metadata_list = embedding_data.get('metadata', [])
    
    if len(chunks) != len(embeddings):
        raise ValueError(f"Chunk count ({len(chunks)}) doesn't match embedding count ({len(embeddings)})")
    
    for i, chunk in enumerate(chunks):
        # Create document metadata
        metadata = {
            'commit_hash': chunk.get('commit_hash', ''),
            'file_path': chunk.get('file_path', ''),
            'chunk_type': chunk.get('chunk_type', chunk.get('type', 'unknown')),
            'language': chunk.get('language', ''),
            'author': chunk.get('author', ''),
            'timestamp': chunk.get('timestamp', ''),
            'message': chunk.get('message', ''),
            'chunk_index': i,
            'content_length': len(chunk.get('content', '')),
            'embedding_model': embedding_data.get('model', ''),
            'embedding_dimension': embedding_data.get('embedding_dimension', 0),
            'ingestion_date': datetime.now().isoformat()
        }
        
        # Add any additional metadata from the chunk
        if 'metadata' in chunk:
            metadata.update(chunk['metadata'])
        
        # Add metadata from the parallel metadata list if available
        if i < len(metadata_list):
            metadata.update(metadata_list[i])
        
        # Create document
        document = VectorDocument(
            id=chunk.get('id', f"chunk_{i}"),
            content=chunk.get('content', ''),
            embedding=embeddings[i] if i < len(embeddings) else None,
            metadata=metadata
        )
        
        documents.append(document)
    
    return documents


def main():
    parser = argparse.ArgumentParser(description="Ingest embeddings into vector database")
    parser.add_argument("embedding_file", help="Path to embedding JSON file")
    parser.add_argument("--collection", default="chromium_embeddings", 
                       help="Collection name in vector database")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for ingestion")
    parser.add_argument("--clear-existing", action="store_true",
                       help="Clear existing collection before ingestion")
    
    args = parser.parse_args()
    
    # Setup
    config = get_config()
    logger = setup_logger(__name__)
    
    embedding_file = Path(args.embedding_file)
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
    
    logger.info(f"Loading embedding data from {embedding_file}")
    embedding_data = load_embedding_data(embedding_file)
    
    logger.info(f"Converting to vector documents...")
    documents = create_vector_documents(embedding_data)
    logger.info(f"Created {len(documents)} vector documents")
    
    # Initialize vector database
    logger.info(f"Initializing vector database with collection: {args.collection}")
    vector_db = VectorDatabase(collection_name=args.collection)
    
    # Clear existing collection if requested
    if args.clear_existing:
        logger.info("Clearing existing collection...")
        try:
            vector_db.client.delete_collection(args.collection)
            vector_db.collection = vector_db._get_or_create_collection()
            logger.info("Collection cleared")
        except Exception as e:
            logger.warning(f"Could not clear collection: {e}")
    
    # Get initial stats
    initial_stats = vector_db.get_collection_stats()
    initial_count = initial_stats.get('total_documents', 0)
    logger.info(f"Initial collection size: {initial_count} documents")
    
    # Ingest documents
    logger.info(f"Starting ingestion of {len(documents)} documents...")
    start_time = datetime.now()
    
    added_count = vector_db.add_documents(documents, batch_size=args.batch_size)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Ingestion completed in {elapsed:.2f}s")
    
    # Get final stats
    final_stats = vector_db.get_collection_stats()
    final_count = final_stats.get('total_documents', 0)
    
    logger.info("Ingestion Summary:")
    logger.info(f"- Embedding file: {embedding_file}")
    logger.info(f"- Collection: {args.collection}")
    logger.info(f"- Documents processed: {len(documents)}")
    logger.info(f"- Documents added: {added_count}")
    logger.info(f"- Initial collection size: {initial_count}")
    logger.info(f"- Final collection size: {final_count}")
    logger.info(f"- Ingestion time: {elapsed:.2f}s")
    logger.info(f"- Average rate: {added_count/elapsed:.1f} docs/sec")
    
    # Test search functionality
    logger.info("Testing search functionality...")
    
    test_queries = [
        "bug fix",
        "memory leak",
        "performance optimization",
        "rendering pipeline"
    ]
    
    for query in test_queries:
        results = vector_db.search(query, n_results=3)
        logger.info(f"Query '{query}': {len(results)} results")
        
        if results:
            top_result = results[0]
            logger.info(f"  Top result (score: {top_result.score:.4f}): "
                       f"{top_result.document.content[:100]}...")
    
    # Create backup
    backup_path = vector_db.backup_collection()
    if backup_path:
        logger.info(f"Created backup: {backup_path}")
    
    logger.info("Vector database ingestion complete!")


if __name__ == "__main__":
    main()