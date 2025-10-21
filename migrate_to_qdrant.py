#!/usr/bin/env python3
"""
Migrate data from ChromaDB to Qdrant vector database.
Exports all documents and re-imports them into Qdrant for better scalability.
"""

import sys
from pathlib import Path
from typing import List
import chromadb
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.vector.qdrant_database import VectorDatabase as QdrantDB, VectorDocument
from rag_system.core.logger import setup_logger


def migrate_chromadb_to_qdrant(
    chroma_path: str = "data/cache/vector_db",
    chroma_collection: str = "chromium_complete",
    qdrant_path: str = "data/cache/qdrant_db",
    qdrant_collection: str = "chromium_complete",
    batch_size: int = 1000
):
    """
    Migrate all documents from ChromaDB to Qdrant.
    
    Args:
        chroma_path: Path to ChromaDB storage
        chroma_collection: ChromaDB collection name
        qdrant_path: Path to Qdrant storage
        qdrant_collection: Qdrant collection name
        batch_size: Batch size for migration
    """
    logger = setup_logger("ChromaDB_to_Qdrant_Migration")
    
    logger.info("=" * 80)
    logger.info("Starting ChromaDB to Qdrant migration")
    logger.info("=" * 80)
    
    try:
        # Connect to ChromaDB
        logger.info(f"Connecting to ChromaDB at {chroma_path}...")
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        chroma_col = chroma_client.get_collection(chroma_collection)
        
        # Get document count
        doc_count = chroma_col.count()
        logger.info(f"Found {doc_count:,} documents in ChromaDB")
        
        if doc_count == 0:
            logger.warning("No documents to migrate!")
            return
        
        # Initialize Qdrant
        logger.info(f"Initializing Qdrant at {qdrant_path}...")
        qdrant_db = QdrantDB(
            collection_name=qdrant_collection,
            persist_directory=qdrant_path,
            vector_size=1024  # BGE-large-en-v1.5
        )
        
        # Export from ChromaDB in batches
        logger.info("Exporting documents from ChromaDB...")
        
        offset = 0
        migrated_count = 0
        
        with tqdm(total=doc_count, desc="Migrating documents") as pbar:
            while offset < doc_count:
                # Fetch batch from ChromaDB
                results = chroma_col.get(
                    limit=batch_size,
                    offset=offset,
                    include=["documents", "metadatas", "embeddings"]
                )
                
                if not results['ids']:
                    break
                
                # Convert to VectorDocument format
                documents: List[VectorDocument] = []
                
                for i, doc_id in enumerate(results['ids']):
                    # Skip documents without embeddings
                    if not results['embeddings'] or i >= len(results['embeddings']):
                        logger.warning(f"Document {doc_id} has no embedding, skipping")
                        continue
                    
                    embedding = results['embeddings'][i]
                    if not embedding:
                        logger.warning(f"Document {doc_id} has empty embedding, skipping")
                        continue
                    
                    content = results['documents'][i] if results['documents'] and i < len(results['documents']) else ""
                    metadata = results['metadatas'][i] if results['metadatas'] and i < len(results['metadatas']) else {}
                    
                    documents.append(VectorDocument(
                        id=doc_id,
                        content=content,
                        embedding=embedding,
                        metadata=metadata
                    ))
                
                # Import to Qdrant
                if documents:
                    added = qdrant_db.add_documents(documents, batch_size=1000)
                    migrated_count += added
                    pbar.update(len(documents))
                
                offset += batch_size
        
        logger.info("=" * 80)
        logger.info(f"Migration completed successfully!")
        logger.info(f"Migrated: {migrated_count:,} / {doc_count:,} documents")
        logger.info("=" * 80)
        
        # Verify Qdrant collection
        stats = qdrant_db.get_collection_stats()
        logger.info(f"Qdrant collection stats: {stats}")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate ChromaDB to Qdrant")
    parser.add_argument("--chroma-path", default="data/cache/vector_db", help="ChromaDB storage path")
    parser.add_argument("--chroma-collection", default="chromium_complete", help="ChromaDB collection name")
    parser.add_argument("--qdrant-path", default="data/cache/qdrant_db", help="Qdrant storage path")
    parser.add_argument("--qdrant-collection", default="chromium_complete", help="Qdrant collection name")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for migration")
    
    args = parser.parse_args()
    
    migrate_chromadb_to_qdrant(
        chroma_path=args.chroma_path,
        chroma_collection=args.chroma_collection,
        qdrant_path=args.qdrant_path,
        qdrant_collection=args.qdrant_collection,
        batch_size=args.batch_size
    )
