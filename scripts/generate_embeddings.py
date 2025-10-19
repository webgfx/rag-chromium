#!/usr/bin/env python3
"""
Generate embeddings for processed Chromium data.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from rag_system.core.config import Config
from rag_system.core.logger import setup_logger
from rag_system.embeddings.models import EmbeddingModelManager
from rag_system.embeddings.generator import EmbeddingGenerator
from rag_system.embeddings.cache import EmbeddingCache


def load_processed_data(file_path: Path) -> List[Dict[str, Any]]:
    """Load processed chunks from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract chunks from the data structure
    if isinstance(data, dict) and 'chunks' in data:
        return data['chunks']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected data format in {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for processed data")
    parser.add_argument("input_file", help="Path to processed data JSON file")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", 
                       help="Embedding model to use")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for embedding generation")
    parser.add_argument("--output-dir", default="data/embeddings",
                       help="Output directory for embeddings")
    
    args = parser.parse_args()
    
    # Setup
    config = Config()
    logger = setup_logger(__name__)
    
    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading processed data from {input_path}")
    chunks = load_processed_data(input_path)
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Initialize embedding system
    logger.info(f"Initializing embedding model: {args.model}")
    embedding_generator = EmbeddingGenerator(model_name=args.model)
    
    start_time = time.time()
    
    # Generate embeddings
    logger.info("Starting embedding generation...")
    texts = [chunk.get('content', '') for chunk in chunks]
    metadata_list = [
        {
            'chunk_id': chunk.get('id', ''),
            'commit_hash': chunk.get('commit_hash', ''),
            'file_path': chunk.get('file_path', ''),
            'chunk_type': chunk.get('type', ''),
            'language': chunk.get('language', ''),
            'size': len(chunk.get('content', ''))
        }
        for chunk in chunks
    ]
    
    embeddings = embedding_generator.encode_texts(texts)
    
    elapsed = time.time() - start_time
    logger.info(f"Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
    
    # Save embeddings
    output_file = output_dir / f"embeddings_{input_path.stem}.json"
    embedding_data = {
        'model': args.model,
        'embedding_dimension': len(embeddings[0]) if len(embeddings) > 0 else 0,
        'total_embeddings': len(embeddings),
        'generation_time': elapsed,
        'chunks': chunks,
        'embeddings': embeddings.tolist(),  # Convert numpy array to list for JSON serialization
        'metadata': metadata_list
    }
    
    logger.info(f"Saving embeddings to {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(embedding_data, f, indent=2, default=str)
    
    logger.info(f"Embedding generation complete!")
    logger.info(f"- Total chunks: {len(chunks)}")
    logger.info(f"- Total embeddings: {len(embeddings)}")
    logger.info(f"- Embedding dimension: {len(embeddings[0]) if len(embeddings) > 0 else 0}")
    logger.info(f"- Generation time: {elapsed:.2f}s")
    logger.info(f"- Output file: {output_file}")


if __name__ == "__main__":
    main()