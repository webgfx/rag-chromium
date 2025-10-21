#!/usr/bin/env python3
"""
Build commit index buckets from existing database.
Uses the ranges already processed in Qdrant to create bucket boundaries.
This is much faster than scanning the entire git repository.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List
from datetime import datetime

# Suppress Qdrant warnings
warnings.filterwarnings('ignore', message='.*Local mode is not recommended.*')

from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_buckets_from_database(
    bucket_size: int = 10000,
    collection_name: str = "chromium_complete",
    db_path: str = "data/cache/qdrant_db",
    output_file: str = "data/commit_buckets.json",
    repo_name: str = "chromium"
) -> Dict:
    """
    Build bucket metadata from existing database documents.
    
    Args:
        bucket_size: Number of commits per bucket
        collection_name: Qdrant collection name
        db_path: Path to Qdrant database
        output_file: Where to save bucket metadata
        
    Returns:
        Dictionary with bucket information
    """
    logger.info("Building commit buckets from database...")
    logger.info(f"Bucket size: {bucket_size:,} commits")
    
    # Connect to database
    client = QdrantClient(path=db_path)
    
    # Get collection info
    collection_info = client.get_collection(collection_name)
    total_docs = collection_info.points_count
    logger.info(f"Total documents in database: {total_docs:,}")
    
    # Scan all documents to get commit indices
    logger.info("Scanning database for commit indices...")
    commits = []
    
    offset = None
    batch_size = 1000
    scanned = 0
    
    while True:
        results = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = results
        
        if not points:
            break
        
        for point in points:
            payload = point.payload
            if 'commit_index' in payload:
                commits.append({
                    'index': payload['commit_index'],
                    'sha': payload.get('commit_sha', ''),
                    'date': payload.get('commit_date', ''),
                    'author': payload.get('author', ''),
                    'subject': payload.get('commit_message', '').split('\n')[0] if payload.get('commit_message') else ''
                })
        
        scanned += len(points)
        if scanned % 5000 == 0:
            logger.info(f"  Scanned {scanned:,}/{total_docs:,} documents...")
        
        if next_offset is None:
            break
        offset = next_offset
    
    logger.info(f"Found {len(commits):,} commits with indices")
    
    # Sort by index
    commits.sort(key=lambda x: x['index'])
    
    # Find min and max indices
    min_index = commits[0]['index']
    max_index = commits[-1]['index']
    logger.info(f"Index range: {min_index:,} - {max_index:,}")
    
    # Create simple index-to-SHA mapping for bucket boundaries
    # Store every Nth commit (bucket boundaries)
    boundary_commits = {}
    
    for i, commit in enumerate(commits):
        idx = commit['index']
        # Round index down to nearest bucket_size
        rounded_idx = (idx // bucket_size) * bucket_size
        
        # Store bucket boundaries (every bucket_size commits, rounded)
        if rounded_idx not in boundary_commits:
            boundary_commits[rounded_idx] = commit['sha']
            logger.info(f"  Boundary at index {rounded_idx:,}: {commit['sha'][:8]}")
    
    logger.info(f"Stored {len(boundary_commits):,} boundary commits")
    
    # Create simplified metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'repo_name': repo_name,
        'source': 'database',
        'bucket_size': bucket_size,
        'total_commits': len(commits),
        'min_index': min_index,
        'max_index': max_index,
        'index_to_sha': boundary_commits
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Bucket metadata saved to: {output_file}")
    logger.info(f"  Total commits: {len(commits):,}")
    logger.info(f"  Boundary commits stored: {len(boundary_commits):,}")
    logger.info(f"  Index range: {min_index:,} - {max_index:,}")
    
    return metadata


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build commit index buckets from database')
    parser.add_argument('--bucket-size', type=int, default=10000, help='Number of commits per bucket (default: 10000)')
    parser.add_argument('--collection', default='chromium_complete', help='Qdrant collection name')
    parser.add_argument('--db-path', default='data/cache/qdrant_db', help='Path to Qdrant database')
    parser.add_argument('--output', default='data/commit_buckets.json', help='Output file path')
    parser.add_argument('--repo-name', default='chromium', help='Repository name (default: chromium)')
    
    args = parser.parse_args()
    
    build_buckets_from_database(
        bucket_size=args.bucket_size,
        collection_name=args.collection,
        db_path=args.db_path,
        output_file=args.output,
        repo_name=args.repo_name
    )
