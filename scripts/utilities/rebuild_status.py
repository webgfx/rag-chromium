#!/usr/bin/env python3
"""
Rebuild status file by scanning the Qdrant database for all processed commits.
This creates a comprehensive status with all processed ranges.
"""

import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Any

# Suppress Qdrant warnings
warnings.filterwarnings('ignore', message='.*Local mode is not recommended.*')

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def scan_database_for_ranges() -> List[Dict[str, Any]]:
    """Scan the Qdrant database and extract all commit ranges based on commit_index."""
    
    logger.info("Connecting to Qdrant database...")
    db_path = Path("data/cache/qdrant_db")
    client = QdrantClient(path=str(db_path))
    
    collection_name = "chromium_complete"
    
    # Get collection info
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        logger.info(f"Found {total_points} documents in collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        return []
    
    if total_points == 0:
        logger.warning("No documents in database")
        return []
    
    # Scroll through all documents and extract commit info
    logger.info("Scanning all documents for commit information...")
    commits = []
    offset = None
    batch_size = 1000
    scanned = 0
    
    while True:
        try:
            records, next_offset = client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            if not records:
                break
            
            for record in records:
                payload = record.payload
                if payload and 'commit_sha' in payload and 'commit_index' in payload:
                    commits.append({
                        'sha': payload['commit_sha'],
                        'index': payload['commit_index'],
                        'date': payload.get('commit_date', ''),
                        'message': payload.get('message', '')[:100],
                        'author': payload.get('author', ''),
                    })
            
            scanned += len(records)
            if scanned % 5000 == 0:
                logger.info(f"Scanned {scanned}/{total_points} documents...")
            
            offset = next_offset
            if offset is None:
                break
                
        except Exception as e:
            logger.error(f"Error during scroll: {e}")
            break
    
    logger.info(f"Extracted {len(commits)} commits from database")
    
    # Sort commits by index (lowest index = newest)
    commits.sort(key=lambda x: x['index'])
    
    # Group into ranges (detect gaps > 1000 in index)
    ranges = []
    if commits:
        current_range = {
            'start_commit': commits[0],
            'end_commit': commits[0],
            'commits': [commits[0]]
        }
        
        for i in range(1, len(commits)):
            # Detect gap in index (> 1000 indices apart indicates different processing batch)
            index_gap = commits[i]['index'] - current_range['end_commit']['index']
            
            if index_gap > 1000:
                # Finalize current range
                current_range['end_commit'] = commits[i-1]
                current_range['commits_count'] = len(current_range['commits'])
                del current_range['commits']  # Don't store all commits, just count
                ranges.append(current_range)
                
                # Start new range
                current_range = {
                    'start_commit': commits[i],
                    'end_commit': commits[i],
                    'commits': [commits[i]]
                }
            else:
                # Add to current range
                current_range['commits'].append(commits[i])
                current_range['end_commit'] = commits[i]
        
        # Add final range
        current_range['end_commit'] = commits[-1]
        current_range['commits_count'] = len(current_range['commits'])
        del current_range['commits']
        ranges.append(current_range)
    
    logger.info(f"Identified {len(ranges)} commit ranges")
    
    # Format ranges for status file (simplified structure)
    formatted_ranges = []
    for idx, r in enumerate(ranges):
        formatted_ranges.append({
            'range_id': idx + 1,
            'start': {
                'index': r['start_commit']['index'],
                'sha': r['start_commit']['sha'],
                'date': r['start_commit']['date'],
                'message': r['start_commit']['message'],
                'author': r['start_commit']['author']
            },
            'end': {
                'index': r['end_commit']['index'],
                'sha': r['end_commit']['sha'],
                'date': r['end_commit']['date'],
                'message': r['end_commit']['message'],
                'author': r['end_commit']['author']
            },
            'commits_count': r['commits_count']
        })
    
    return formatted_ranges


def get_database_stats() -> Dict[str, Any]:
    """Get current database statistics."""
    try:
        db_path = Path("data/cache/qdrant_db")
        client = QdrantClient(path=str(db_path))
        collection_name = "chromium_complete"
        
        collection_info = client.get_collection(collection_name)
        
        return {
            'total_documents': collection_info.points_count,
            'collection_name': collection_name,
            'is_healthy': collection_info.points_count > 0,
            'vectors_count': collection_info.vectors_count,
            'indexed_vectors_count': collection_info.indexed_vectors_count
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {
            'total_documents': 0,
            'collection_name': 'chromium_complete',
            'is_healthy': False
        }


def rebuild_status_file():
    """Rebuild the complete status file from database scan."""
    
    logger.info("Starting status file rebuild...")
    
    # Scan database for ranges
    processed_ranges = scan_database_for_ranges()
    
    # Get database stats
    db_stats = get_database_stats()
    
    # Calculate aggregate statistics
    total_commits = sum(r['commits_count'] for r in processed_ranges)
    
    # Build progress info from ranges
    progress = {
        'commits_processed': total_commits,
        'batches_completed': 0,  # Unknown from database scan
        'documents_created': db_stats['total_documents'],
        'embeddings_generated': db_stats['total_documents']
    }
    
    # Add first/last commit info if ranges exist
    if processed_ranges:
        first_range = processed_ranges[0]
        last_range = processed_ranges[-1]
        
        progress.update({
            'first_commit_sha': first_range['start']['sha'],
            'first_commit_date': first_range['start']['date'],
            'first_commit_message': first_range['start']['message'],
            'last_commit_sha': last_range['end']['sha'],
            'last_commit_date': last_range['end']['date'],
            'last_commit_message': last_range['end']['message']
        })
    
    # Get system resources
    import psutil
    system_resources = {
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('.').percent
    }
    
    # Build current_range (use last range as current if it exists)
    current_range = None
    if processed_ranges:
        last_range = processed_ranges[-1]
        current_range = {
            'start': last_range['start'],
            'end': last_range['end'],
            'commits_count': last_range['commits_count'],
            'batches_completed': 0,  # Unknown
            'documents_created': last_range['commits_count']  # Approximate
        }
    
    # Build complete status
    status = {
        'timestamp': datetime.now().isoformat(),
        'progress': progress,
        'database': db_stats,
        'processed_ranges': processed_ranges,
        'current_range': current_range,
        'system': system_resources,
        'stats': {
            'total_ranges': len(processed_ranges),
            'total_commits_processed': total_commits,
            'avg_commits_per_range': total_commits / len(processed_ranges) if processed_ranges else 0,
            'rebuild_timestamp': datetime.now().isoformat()
        }
    }
    
    # Write status file
    status_file = Path('data/status.json')
    status_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    
    logger.info(f"Status file rebuilt successfully!")
    logger.info(f"  - Total documents: {db_stats['total_documents']}")
    logger.info(f"  - Total commits: {total_commits}")
    logger.info(f"  - Total ranges: {len(processed_ranges)}")
    logger.info(f"  - Status file: {status_file}")
    
    # Also update progress.json
    progress_file = Path('data/massive_cache/progress.json')
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2, default=str)
    logger.info(f"  - Progress file: {progress_file}")
    
    # Print range summary
    if processed_ranges:
        logger.info("\nProcessed Ranges Summary:")
        for r in processed_ranges:
            logger.info(f"  Range {r['range_id']}: index {r['start']['index']}-{r['end']['index']} ({r['commits_count']} commits, dates {r['start']['date'][:10]} to {r['end']['date'][:10]})")

if __name__ == '__main__':
    try:
        rebuild_status_file()
        print("\n✅ Status file rebuild completed successfully!")
    except Exception as e:
        logger.error(f"Failed to rebuild status file: {e}", exc_info=True)
        print(f"\n❌ Failed to rebuild status file: {e}")
