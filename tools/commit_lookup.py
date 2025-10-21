#!/usr/bin/env python3
"""
Commit lookup utility: convert between commit hash and git log index.

Usage:
    python commit_lookup.py --hash <commit_sha>     # Get index from hash
    python commit_lookup.py --index <index>         # Get hash from index
    python commit_lookup.py --hash <sha> --repo <path>
"""

import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def load_bucket_info():
    """Load commit bucket information for fast lookups."""
    bucket_file = Path(__file__).parent.parent / "data" / "commit_buckets.json"
    if bucket_file.exists():
        with open(bucket_file) as f:
            return json.load(f)
    return None


def get_index_from_hash(commit_sha: str, repo_path: str = r"d:\r\cr\src") -> int:
    """
    Get the git log index for a given commit hash.
    
    Args:
        commit_sha: Full or partial commit SHA
        repo_path: Path to git repository
    
    Returns:
        Index in git log --all --reverse (0-based)
    """
    try:
        # Get all commits and find the position
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        commits = result.stdout.strip().split('\n')
        
        # Find matching commit (supports partial SHA)
        for i, sha in enumerate(commits):
            if sha.startswith(commit_sha):
                return i
        
        raise ValueError(f"Commit {commit_sha} not found in repository")
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e}")


def get_hash_from_index(index: int, repo_path: str = r"d:\r\cr\src") -> str:
    """
    Get the commit hash for a given git log index.
    Optimized using bucket information when available.
    
    Args:
        index: Index in git log --all --reverse (0-based)
        repo_path: Path to git repository
    
    Returns:
        Full commit SHA
    """
    # Try to use bucket information first for optimization
    bucket_info = load_bucket_info()
    
    if bucket_info:
        bucket_size = bucket_info.get('bucket_size', 10000)
        index_to_sha = bucket_info.get('index_to_sha', {})
        
        # Find the closest bucket boundary
        bucket_index = (index // bucket_size) * bucket_size
        bucket_key = str(bucket_index)
        
        if bucket_key in index_to_sha:
            # We have a nearby boundary commit, use it as starting point
            boundary_sha = index_to_sha[bucket_key]
            offset = index - bucket_index
            
            print(f"Using bucket optimization: starting from index {bucket_index}", file=sys.stderr)
            print(f"Boundary SHA: {boundary_sha[:8]}...", file=sys.stderr)
            print(f"Offset: +{offset} commits", file=sys.stderr)
            
            # Skip from boundary to target
            try:
                result = subprocess.run(
                    ['git', 'rev-list', '--reverse', '--ancestry-path', f'{boundary_sha}~1..HEAD'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30
                )
                
                commits = result.stdout.strip().split('\n')
                if offset < len(commits):
                    return commits[offset]
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                print(f"Bucket optimization failed, falling back to direct method", file=sys.stderr)
    
    # Fallback: direct skip method
    try:
        result = subprocess.run(
            ['git', 'rev-list', '--all', '--reverse', f'--skip={index}', '--max-count=1'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        sha = result.stdout.strip()
        if sha:
            return sha
        else:
            raise ValueError(f"No commit found at index {index}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git command failed: {e}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Git command timed out (index {index} may be too large)")


def verify_index(commit_sha: str, expected_index: int, repo_path: str = r"d:\r\cr\src") -> bool:
    """Verify that a commit is at the expected index."""
    try:
        actual_sha = get_hash_from_index(expected_index, repo_path)
        return actual_sha.startswith(commit_sha) or commit_sha.startswith(actual_sha)
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert between commit hash and git log index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get index from commit hash
  python commit_lookup.py --hash a1b2c3d4
  
  # Get hash from index
  python commit_lookup.py --index 1755000
  
  # Specify custom repository path
  python commit_lookup.py --index 1755000 --repo /path/to/repo
  
  # Verify a commit's index
  python commit_lookup.py --hash a1b2c3d4 --verify-index 1755000
        """
    )
    
    parser.add_argument('--hash', '--sha', dest='commit_sha',
                       help='Commit SHA (full or partial) to look up index')
    parser.add_argument('--index', type=int,
                       help='Git log index to look up commit hash')
    parser.add_argument('--verify-index', type=int,
                       help='Verify that --hash is at this index')
    parser.add_argument('--repo', '--repo-path', dest='repo_path',
                       default=r"d:\r\cr\src",
                       help='Path to git repository (default: d:\\r\\cr\\src)')
    parser.add_argument('--json', action='store_true',
                       help='Output results in JSON format')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.commit_sha and args.index is None:
        parser.error("Must specify either --hash or --index")
    
    if args.commit_sha and args.index is not None and args.verify_index is None:
        parser.error("Cannot specify both --hash and --index (use --verify-index for verification)")
    
    try:
        result = {}
        
        # Hash to index lookup
        if args.commit_sha and args.verify_index is None:
            print(f"Looking up index for commit {args.commit_sha}...", file=sys.stderr)
            index = get_index_from_hash(args.commit_sha, args.repo_path)
            result = {
                'commit_sha': args.commit_sha,
                'index': index,
                'lookup_type': 'hash_to_index'
            }
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\nCommit: {args.commit_sha}")
                print(f"Index:  {index:,}")
        
        # Index to hash lookup
        elif args.index is not None:
            print(f"Looking up commit hash for index {args.index:,}...", file=sys.stderr)
            commit_sha = get_hash_from_index(args.index, args.repo_path)
            result = {
                'commit_sha': commit_sha,
                'index': args.index,
                'lookup_type': 'index_to_hash'
            }
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print(f"\nIndex:  {args.index:,}")
                print(f"Commit: {commit_sha}")
        
        # Verification
        if args.verify_index is not None and args.commit_sha:
            print(f"\nVerifying commit {args.commit_sha} is at index {args.verify_index}...", file=sys.stderr)
            is_correct = verify_index(args.commit_sha, args.verify_index, args.repo_path)
            result['verified'] = is_correct
            result['expected_index'] = args.verify_index
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                if is_correct:
                    print(f"✓ Verified: {args.commit_sha} is at index {args.verify_index:,}")
                else:
                    print(f"✗ Mismatch: {args.commit_sha} is NOT at index {args.verify_index:,}")
                    sys.exit(1)
    
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
