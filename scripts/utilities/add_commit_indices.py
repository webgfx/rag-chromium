"""
Add commit_index to existing documents in the database.
This script scans the database, groups commits by their SHA,
and assigns indices based on commit_date order (approximation of git log order).
"""

import logging
from pathlib import Path
from qdrant_client import QdrantClient
from typing import Dict, List, Any
from collections import defaultdict
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def add_commit_indices():
    """Add commit_index field to all documents in the database."""
    
    logger.info("Starting commit index assignment...")
    
    # Connect to database
    db_path = Path("data/cache/qdrant_db")
    client = QdrantClient(path=str(db_path))
    collection_name = "chromium_complete"
    
    try:
        collection_info = client.get_collection(collection_name)
        total_points = collection_info.points_count
        logger.info(f"Found {total_points} documents in collection '{collection_name}'")
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        return False
    
    if total_points == 0:
        logger.warning("No documents in database")
        return True
    
    # Step 1: Scan all documents and group by commit_sha
    logger.info("Step 1: Scanning documents and grouping by commit...")
    commits_map = defaultdict(list)  # commit_sha -> list of (point_id, date, current_index)
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
                if payload and 'commit_sha' in payload:
                    commit_sha = payload['commit_sha']
                    commit_date = payload.get('commit_date', '')
                    current_index = payload.get('commit_index')  # May be None
                    commits_map[commit_sha].append((record.id, commit_date, current_index))
            
            scanned += len(records)
            if scanned % 5000 == 0:
                logger.info(f"Scanned {scanned}/{total_points} documents...")
            
            offset = next_offset
            if offset is None:
                break
                
        except Exception as e:
            logger.error(f"Error during scroll: {e}")
            return False
    
    logger.info(f"Found {len(commits_map)} unique commits")
    
    # Step 2: Sort commits by date (oldest to newest)
    logger.info("Step 2: Sorting commits by date...")
    commit_dates = []
    for commit_sha, points in commits_map.items():
        # Use the date from the first point for this commit
        commit_date = points[0][1]
        commit_dates.append((commit_sha, commit_date))
    
    # Sort by date (oldest first = highest index, newest first = lowest index)
    # Git log returns newest first, so index 0 = newest commit
    commit_dates.sort(key=lambda x: x[1], reverse=True)  # Newest first
    
    # Assign indices: newest commit = index 0
    commit_to_index = {}
    for idx, (commit_sha, _) in enumerate(commit_dates):
        commit_to_index[commit_sha] = idx
    
    logger.info(f"Assigned indices: 0 (newest) to {len(commit_to_index)-1} (oldest)")
    
    # Step 3: Update documents with commit_index
    logger.info("Step 3: Updating documents with commit_index...")
    updates_needed = []
    already_correct = 0
    
    for commit_sha, points in commits_map.items():
        correct_index = commit_to_index[commit_sha]
        for point_id, _, current_index in points:
            if current_index != correct_index:
                updates_needed.append((point_id, correct_index))
            else:
                already_correct += 1
    
    logger.info(f"Updates needed: {len(updates_needed)}, Already correct: {already_correct}")
    
    if not updates_needed:
        logger.info("All documents already have correct commit_index!")
        return True
    
    # Perform updates in batches
    logger.info("Updating documents...")
    batch_size = 100
    updated = 0
    
    for i in range(0, len(updates_needed), batch_size):
        batch = updates_needed[i:i+batch_size]
        
        try:
            # Fetch full payloads for this batch
            point_ids = [point_id for point_id, _ in batch]
            records = client.retrieve(
                collection_name=collection_name,
                ids=point_ids,
                with_payload=True,
                with_vectors=False
            )
            
            # Update payloads with new commit_index
            from qdrant_client.models import PointStruct
            updated_points = []
            for record, (point_id, new_index) in zip(records, batch):
                payload = dict(record.payload)
                payload['commit_index'] = new_index
                updated_points.append(PointStruct(
                    id=point_id,
                    payload=payload,
                    vector={}  # Empty, we're not updating vectors
                ))
            
            # Upsert (update existing points)
            client.upsert(
                collection_name=collection_name,
                points=updated_points
            )
            
            updated += len(batch)
            if updated % 1000 == 0:
                logger.info(f"Updated {updated}/{len(updates_needed)} documents...")
            
        except Exception as e:
            logger.error(f"Error updating batch: {e}")
            continue
    
    logger.info(f"✅ Successfully updated {updated} documents with commit_index")
    return True


if __name__ == "__main__":
    try:
        success = add_commit_indices()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
