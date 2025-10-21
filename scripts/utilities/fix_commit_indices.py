"""
Fix commit indices to match actual git repository order.
Index 0 = first commit in repo, increasing = newer commits.
"""

import logging
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import subprocess
import sys
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_git_commit_indices(repo_path: str) -> dict:
    """Get commit SHA to index mapping from git repository.
    Index 0 = first commit, increasing = newer commits."""
    
    logger.info(f"Querying git repository at {repo_path}...")
    
    # Get all commit SHAs in reverse order (oldest first)
    result = subprocess.run(
        ['git', 'log', '--all', '--reverse', '--format=%H'],
        cwd=repo_path,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    
    if result.returncode != 0:
        raise Exception(f"Git command failed: {result.stderr}")
    
    commit_shas = result.stdout.strip().split('\n')
    logger.info(f"Found {len(commit_shas)} total commits in repository")
    
    # Create SHA -> index mapping (oldest = 0, newest = N-1)
    sha_to_index = {}
    for idx, sha in enumerate(commit_shas):
        sha_to_index[sha] = idx
    
    logger.info(f"First commit (index 0): {commit_shas[0][:8]}")
    logger.info(f"Last commit (index {len(commit_shas)-1}): {commit_shas[-1][:8]}")
    
    return sha_to_index


def fix_database_indices(repo_path: str):
    """Fix commit indices in database to match git repository order."""
    
    # Get correct indices from git
    sha_to_index = get_git_commit_indices(repo_path)
    
    # Connect to database
    logger.info("Connecting to database...")
    db_path = Path("data/cache/qdrant_db")
    client = QdrantClient(path=str(db_path))
    collection_name = "chromium_complete"
    
    collection_info = client.get_collection(collection_name)
    total_points = collection_info.points_count
    logger.info(f"Database has {total_points} documents")
    
    # Scan database and collect updates
    logger.info("Scanning database for commits...")
    updates_needed = []
    found_in_git = 0
    not_found_in_git = 0
    offset = None
    batch_size = 1000
    scanned = 0
    
    while True:
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
            commit_sha = payload.get('commit_sha')
            current_index = payload.get('commit_index')
            
            if commit_sha in sha_to_index:
                correct_index = sha_to_index[commit_sha]
                if current_index != correct_index:
                    updates_needed.append((record.id, commit_sha, current_index, correct_index))
                found_in_git += 1
            else:
                not_found_in_git += 1
                logger.warning(f"Commit {commit_sha[:8]} not found in git repo")
        
        scanned += len(records)
        if scanned % 5000 == 0:
            logger.info(f"Scanned {scanned}/{total_points} documents...")
        
        offset = next_offset
        if offset is None:
            break
    
    logger.info(f"Found in git: {found_in_git}, Not found: {not_found_in_git}")
    logger.info(f"Updates needed: {len(updates_needed)}")
    
    if not updates_needed:
        logger.info("✅ All indices are already correct!")
        return
    
    # Perform updates
    logger.info("Updating database...")
    updated = 0
    batch_size = 100
    
    for i in range(0, len(updates_needed), batch_size):
        batch = updates_needed[i:i+batch_size]
        point_ids = [item[0] for item in batch]
        
        # Fetch full records
        records = client.retrieve(
            collection_name=collection_name,
            ids=point_ids,
            with_payload=True,
            with_vectors=False
        )
        
        # Update with correct indices
        updated_points = []
        for record, (point_id, sha, old_idx, new_idx) in zip(records, batch):
            payload = dict(record.payload)
            payload['commit_index'] = new_idx
            updated_points.append(PointStruct(
                id=point_id,
                payload=payload,
                vector={}
            ))
        
        client.upsert(
            collection_name=collection_name,
            points=updated_points
        )
        
        updated += len(batch)
        if updated % 1000 == 0:
            logger.info(f"Updated {updated}/{len(updates_needed)} documents...")
    
    logger.info(f"✅ Successfully updated {updated} documents with correct indices!")
    logger.info(f"Index 0 = oldest commit, Index {max(sha_to_index.values())} = newest commit")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix commit indices to match git repository order")
    parser.add_argument("--repo-path", default="d:\\r\\cr\\src", help="Path to Chromium repository")
    args = parser.parse_args()
    
    try:
        fix_database_indices(args.repo_path)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
