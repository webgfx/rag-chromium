#!/usr/bin/env python3
"""
Fast commit lookup by index using bucket system.
O(1) bucket lookup followed by O(bucket_size) scan instead of O(total_commits) scan.
"""

import json
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CommitBucketLookup:
    """Fast commit lookup using pre-computed buckets."""
    
    def __init__(self, bucket_file: str = "data/commit_buckets.json", repo_path: str = None):
        """
        Initialize with bucket metadata file.
        
        Args:
            bucket_file: Path to bucket metadata JSON file
            repo_path: Optional override for repository path
        """
        self.bucket_file = Path(bucket_file)
        if not self.bucket_file.exists():
            raise FileNotFoundError(
                f"Bucket file not found: {bucket_file}\n"
                "Please run build_buckets_from_database.py first to generate bucket metadata."
            )
        
        with open(self.bucket_file) as f:
            self.metadata = json.load(f)
        
        # Use provided repo_path or fallback to metadata (if available)
        self.repo_path = repo_path or self.metadata.get('repo_path')
        if not self.repo_path:
            raise ValueError(
                "Repository path not found in bucket metadata. "
                "Please provide --repo-path argument."
            )
        
        self.bucket_size = self.metadata['bucket_size']
        self.total_commits = self.metadata.get('total_commits', 0)
        self.min_index = self.metadata.get('min_index', 0)
        self.max_index = self.metadata.get('max_index', 0)
        self.index_to_sha = self.metadata.get('index_to_sha', {})
        
        logger.info(f"Loaded bucket metadata:")
        logger.info(f"  Repository: {self.repo_path}")
        logger.info(f"  Total commits: {self.total_commits:,}")
        logger.info(f"  Bucket size: {self.bucket_size:,}")
        logger.info(f"  Index range: {self.min_index:,} - {self.max_index:,}")
        logger.info(f"  Boundary commits: {len(self.index_to_sha):,}")
    
    def find_bucket(self, index: int) -> Optional[Dict]:
        """Find bucket range containing the given index."""
        if index < self.min_index or index > self.max_index:
            logger.error(f"Index {index:,} out of range [{self.min_index:,}, {self.max_index:,}]")
            return None
        
        # Find the bucket boundaries around this index
        bucket_id = (index - self.min_index) // self.bucket_size
        bucket_start = self.min_index + (bucket_id * self.bucket_size)
        bucket_end = min(bucket_start + self.bucket_size - 1, self.max_index)
        
        return {
            'bucket_id': bucket_id,
            'start_index': bucket_start,
            'end_index': bucket_end
        }
    
    def lookup_commit(self, index: int, verbose: bool = True) -> Optional[Dict]:
        """
        Lookup commit by index using bucket system.
        
        Args:
            index: Git log index (0 = oldest commit)
            verbose: Whether to print lookup details
            
        Returns:
            Dictionary with commit information or None if not found
        """
        if verbose:
            logger.info(f"Looking up commit at index {index:,}")
        
        # Find the bucket
        bucket = self.find_bucket(index)
        if not bucket:
            logger.error(f"Could not find bucket for index {index}")
            return None
        
        if verbose:
            logger.info(f"Found in bucket {bucket['bucket_id']}: indices {bucket['start_index']:,} - {bucket['end_index']:,}")
        
        # Calculate offset within bucket
        offset_in_bucket = index - bucket['start_index']
        
        if verbose:
            logger.info(f"Offset within bucket: {offset_in_bucket:,}")
            logger.info(f"Fetching commit from repository...")
        
        # Fetch the specific commit using skip from bucket start
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', str(index), '--max-count', '1'],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        sha = result.stdout.strip()
        
        # Get full commit details
        result = subprocess.run(
            ['git', 'log', '--format=%H%n%an%n%ae%n%ai%n%s%n%b', '-n', '1', sha],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        
        commit_info = {
            'index': index,
            'sha': lines[0] if len(lines) > 0 else '',
            'author': lines[1] if len(lines) > 1 else '',
            'author_email': lines[2] if len(lines) > 2 else '',
            'date': lines[3] if len(lines) > 3 else '',
            'subject': lines[4] if len(lines) > 4 else '',
            'body': '\n'.join(lines[5:]) if len(lines) > 5 else '',
            'bucket_id': bucket['bucket_id']
        }
        
        return commit_info
    
    def lookup_range(self, start_index: int, end_index: int) -> list:
        """
        Lookup multiple commits in a range.
        
        Args:
            start_index: Starting index (inclusive)
            end_index: Ending index (inclusive)
            
        Returns:
            List of commit information dictionaries
        """
        logger.info(f"Looking up commit range {start_index:,} - {end_index:,}")
        
        # Find buckets covering this range
        start_bucket = self.find_bucket(start_index)
        end_bucket = self.find_bucket(end_index)
        
        if not start_bucket or not end_bucket:
            logger.error("Could not find buckets for range")
            return []
        
        logger.info(f"Range spans buckets {start_bucket['bucket_id']} - {end_bucket['bucket_id']}")
        
        # Fetch all commits in range using git rev-list
        count = end_index - start_index + 1
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', '--skip', str(start_index), '--max-count', str(count)],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=60
        )
        
        shas = result.stdout.strip().split('\n')
        logger.info(f"Fetched {len(shas):,} commit SHAs")
        
        return shas


def main():
    parser = argparse.ArgumentParser(description='Fast commit lookup by index using buckets')
    parser.add_argument('--index', type=int, help='Commit index to lookup')
    parser.add_argument('--start-index', type=int, help='Start index for range lookup')
    parser.add_argument('--end-index', type=int, help='End index for range lookup')
    parser.add_argument('--bucket-file', default='data/commit_buckets.json', help='Bucket metadata file')
    parser.add_argument('--repo-path', help='Override repository path (required if not in bucket metadata)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    
    args = parser.parse_args()
    
    try:
        lookup = CommitBucketLookup(args.bucket_file, args.repo_path)
        
        if args.index is not None:
            # Single commit lookup
            commit = lookup.lookup_commit(args.index, verbose=not args.quiet)
            if commit:
                print("\n" + "="*80)
                print(f"Commit at index {commit['index']:,}")
                print("="*80)
                print(f"SHA:     {commit['sha']}")
                print(f"Author:  {commit['author']} <{commit['author_email']}>")
                print(f"Date:    {commit['date']}")
                print(f"Subject: {commit['subject']}")
                if commit['body'].strip():
                    print(f"\n{commit['body']}")
                print(f"\nBucket:  {commit['bucket_id']}")
                print("="*80)
        
        elif args.start_index is not None and args.end_index is not None:
            # Range lookup
            shas = lookup.lookup_range(args.start_index, args.end_index)
            if shas:
                print(f"\nCommit SHAs for indices {args.start_index:,} - {args.end_index:,}:")
                for i, sha in enumerate(shas, start=args.start_index):
                    print(f"{i:8d}: {sha}")
        
        else:
            parser.print_help()
            print("\nExamples:")
            print(f"  python {parser.prog} --index 1700000")
            print(f"  python {parser.prog} --start-index 1700000 --end-index 1700100")
    
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
