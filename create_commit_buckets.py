#!/usr/bin/env python3
"""
Create commit index buckets for fast lookup.
Splits the entire git history into buckets of N commits each, storing boundary information.
This allows O(1) bucket lookup followed by O(bucket_size) scan instead of O(total_commits) scan.
"""

import json
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_total_commits(repo_path: str) -> int:
    """Get total number of commits in the repository."""
    result = subprocess.run(
        ['git', 'rev-list', '--all', '--count'],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    return int(result.stdout.strip())


def get_commit_info(repo_path: str, sha: str) -> Dict:
    """Get detailed information for a specific commit."""
    result = subprocess.run(
        ['git', 'log', '--format=%H%n%an%n%ae%n%ai%n%s', '-n', '1', sha],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True
    )
    
    lines = result.stdout.strip().split('\n')
    if len(lines) >= 5:
        return {
            'sha': lines[0],
            'author': lines[1],
            'author_email': lines[2],
            'date': lines[3],
            'subject': lines[4]
        }
    return None


def create_buckets(
    repo_path: str,
    bucket_size: int = 10000,
    output_file: str = "data/commit_buckets.json"
) -> Dict:
    """
    Create commit index buckets.
    
    Args:
        repo_path: Path to git repository
        bucket_size: Number of commits per bucket
        output_file: Where to save bucket metadata
        
    Returns:
        Dictionary with bucket information
    """
    logger.info(f"Creating commit buckets for repository: {repo_path}")
    logger.info(f"Bucket size: {bucket_size:,} commits")
    
    # Get total commits
    total_commits = get_total_commits(repo_path)
    logger.info(f"Total commits in repository: {total_commits:,}")
    
    # Calculate number of buckets
    num_buckets = (total_commits + bucket_size - 1) // bucket_size  # Ceiling division
    logger.info(f"Creating {num_buckets:,} buckets")
    
    buckets = []
    
    # Process each bucket with retry logic
    for bucket_id in range(num_buckets):
        start_index = bucket_id * bucket_size
        end_index = min(start_index + bucket_size - 1, total_commits - 1)
        
        logger.info(f"Processing bucket {bucket_id + 1}/{num_buckets}: indices {start_index:,} - {end_index:,}")
        
        # Retry logic for git commands
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Get start commit (oldest in bucket)
                start_result = subprocess.run(
                    ['git', 'rev-list', '--all', '--reverse', '--skip', str(start_index), '--max-count', '1'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
                start_sha = start_result.stdout.strip()
                
                # Get end commit (newest in bucket)
                end_result = subprocess.run(
                    ['git', 'rev-list', '--all', '--reverse', '--skip', str(end_index), '--max-count', '1'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
                end_sha = end_result.stdout.strip()
                
                # Get commit details
                start_info = get_commit_info(repo_path, start_sha)
                end_info = get_commit_info(repo_path, end_sha)
                
                bucket = {
                    'bucket_id': bucket_id,
                    'start_index': start_index,
                    'end_index': end_index,
                    'size': end_index - start_index + 1,
                    'start_commit': start_info,
                    'end_commit': end_info
                }
                
                buckets.append(bucket)
                
                logger.info(f"  Bucket {bucket_id}: {start_sha[:8]} ({start_info['date']}) -> {end_sha[:8]} ({end_info['date']})")
                break  # Success, exit retry loop
                
            except KeyboardInterrupt:
                if attempt < max_retries - 1:
                    logger.warning(f"  Interrupted (attempt {attempt + 1}/{max_retries}), retrying...")
                    import time
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"  Failed after {max_retries} attempts, skipping bucket {bucket_id}")
                    # Add placeholder bucket
                    bucket = {
                        'bucket_id': bucket_id,
                        'start_index': start_index,
                        'end_index': end_index,
                        'size': end_index - start_index + 1,
                        'start_commit': None,
                        'end_commit': None,
                        'error': 'Failed to fetch'
                    }
                    buckets.append(bucket)
                    break
            except Exception as e:
                logger.error(f"  Error processing bucket {bucket_id}: {e}")
                if attempt < max_retries - 1:
                    logger.warning(f"  Retrying (attempt {attempt + 1}/{max_retries})...")
                    import time
                    time.sleep(1)
                    continue
                else:
                    # Add placeholder bucket
                    bucket = {
                        'bucket_id': bucket_id,
                        'start_index': start_index,
                        'end_index': end_index,
                        'size': end_index - start_index + 1,
                        'start_commit': None,
                        'end_commit': None,
                        'error': str(e)
                    }
                    buckets.append(bucket)
                    break
    
    # Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'repo_path': repo_path,
        'total_commits': total_commits,
        'bucket_size': bucket_size,
        'num_buckets': num_buckets,
        'buckets': buckets
    }
    
    # Save to file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✓ Bucket metadata saved to: {output_file}")
    logger.info(f"  Total commits: {total_commits:,}")
    logger.info(f"  Buckets created: {num_buckets:,}")
    logger.info(f"  Bucket size: {bucket_size:,}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(description='Create commit index buckets for fast lookup')
    parser.add_argument('--repo-path', required=True, help='Path to git repository')
    parser.add_argument('--bucket-size', type=int, default=10000, help='Number of commits per bucket (default: 10000)')
    parser.add_argument('--output', default='data/commit_buckets.json', help='Output file path')
    
    args = parser.parse_args()
    
    create_buckets(args.repo_path, args.bucket_size, args.output)


if __name__ == '__main__':
    main()
