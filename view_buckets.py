#!/usr/bin/env python3
"""
View commit bucket information and statistics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict

def view_buckets(bucket_file: str = "data/commit_buckets.json", detailed: bool = False):
    """Display bucket information."""
    
    if not Path(bucket_file).exists():
        print(f"❌ Bucket file not found: {bucket_file}")
        print("Run: python build_buckets_from_database.py")
        return
    
    with open(bucket_file) as f:
        metadata = json.load(f)
    
    print("="*80)
    print("COMMIT INDEX BUCKET SYSTEM")
    print("="*80)
    print(f"Repository: {metadata.get('repo_name', 'N/A')}")
    print(f"Created: {metadata['created_at']}")
    print(f"Source: {metadata.get('source', 'git repository')}")
    print(f"Bucket size: {metadata['bucket_size']:,} commits")
    print(f"Total commits: {metadata.get('total_commits', 'N/A'):,}")
    print(f"Index range: {metadata.get('min_index', 0):,} - {metadata.get('max_index', 'N/A'):,}")
    
    index_to_sha = metadata.get('index_to_sha', {})
    print(f"Boundary commits stored: {len(index_to_sha):,}")
    print("="*80)
    
    if detailed:
        print("\nINDEX TO SHA MAPPING:")
        print("-"*80)
        # Sort by index
        sorted_indices = sorted([int(k) for k in index_to_sha.keys()])
        for idx in sorted_indices:
            sha = index_to_sha[str(idx)]
            print(f"  {idx:12,}  ->  {sha}")
    else:
        print("\nBOUNDARY COMMITS (first 10):")
        print(f"{'Index':>12}  {'SHA':>42}")
        print("-"*80)
        # Show first 10 boundaries
        sorted_indices = sorted([int(k) for k in index_to_sha.keys()])
        for idx in sorted_indices[:10]:
            sha = index_to_sha[str(idx)]
            print(f"{idx:12,}  {sha}")
        
        if len(sorted_indices) > 10:
            print(f"  ... and {len(sorted_indices) - 10} more boundaries")
    
    print("="*80)
    print(f"\nUse --detailed to see all {len(index_to_sha)} boundary commits")
    print(f"Use lookup: python lookup_commit_by_index.py --index <N> --repo-path <PATH>")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View commit bucket information')
    parser.add_argument('--bucket-file', default='data/commit_buckets.json', help='Bucket metadata file')
    parser.add_argument('--detailed', action='store_true', help='Show detailed bucket information')
    
    args = parser.parse_args()
    
    view_buckets(args.bucket_file, args.detailed)
