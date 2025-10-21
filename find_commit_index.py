"""
Find commit index by SHA hash.
Usage: python find_commit_index.py <sha> --repo-path REPO_PATH
"""

import sys
import subprocess
import argparse
from pathlib import Path
from qdrant_client import QdrantClient
import warnings
warnings.filterwarnings('ignore')

def find_index_in_database(sha: str):
    """Check if commit exists in database and return its index."""
    try:
        db_path = Path("data/cache/qdrant_db")
        client = QdrantClient(path=str(db_path))
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Search for this SHA
        results = client.scroll(
            collection_name="chromium_complete",
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="commit_sha",
                        match=MatchValue(value=sha)
                    )
                ]
            ),
            limit=1,
            with_payload=['commit_index', 'commit_date', 'author'],
            with_vectors=False
        )
        
        records, _ = results
        client.close()
        
        if records:
            return records[0].payload.get('commit_index'), True
        return None, False
        
    except Exception as e:
        return None, False

def find_index_in_git(repo_path: str, sha: str):
    """Find commit index by scanning git repository."""
    print(f"🔍 Searching for commit {sha[:8]} in git repository...")
    print(f"   (This may take a few seconds for 1.7M commits...)")
    
    try:
        # Get all commits and find the index
        result = subprocess.run(
            ['git', 'log', '--all', '--reverse', '--format=%H'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        commit_shas = result.stdout.strip().split('\n')
        total_commits = len(commit_shas)
        
        # Find the SHA (support partial SHA)
        matching_indices = []
        for idx, commit_sha in enumerate(commit_shas):
            if commit_sha.startswith(sha):
                matching_indices.append((idx, commit_sha))
        
        if not matching_indices:
            print(f"❌ Commit {sha} not found in repository")
            return None
        
        if len(matching_indices) > 1:
            print(f"⚠️  Multiple commits match {sha}:")
            for idx, full_sha in matching_indices[:5]:
                print(f"   Index {idx}: {full_sha}")
            print(f"\n   Please use a longer SHA to be more specific")
            return None
        
        index, full_sha = matching_indices[0]
        
        # Get commit details
        result = subprocess.run(
            ['git', 'show', '--no-patch', '--format=%ai%n%an%n%s', full_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        commit_date = lines[0] if len(lines) > 0 else 'N/A'
        author = lines[1] if len(lines) > 1 else 'N/A'
        message = lines[2] if len(lines) > 2 else 'N/A'
        
        print(f"\n✅ Found commit!")
        print(f"   Index:   {index:,} (out of {total_commits:,} total)")
        print(f"   SHA:     {full_sha}")
        print(f"   Date:    {commit_date}")
        print(f"   Author:  {author}")
        print(f"   Message: {message[:80]}")
        
        return index
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Find commit index by SHA hash')
    parser.add_argument('sha', type=str, help='Commit SHA (full or partial)')
    parser.add_argument('--repo-path', type=str, help='Path to git repository')
    
    args = parser.parse_args()
    sha = args.sha.strip()
    
    # Try database first (fast)
    index, in_db = find_index_in_database(sha)
    if in_db:
        print(f"\n✅ Found in database!")
        print(f"   Commit SHA: {sha}")
        print(f"   Index:      {index:,}")
        print(f"\n   Use: python lookup_commit.py {index}")
        return
    
    # Not in database
    if not args.repo_path:
        print(f"\n❌ Commit {sha} not found in database.")
        print(f"   Use --repo-path to search git repository directly.")
        print(f"   Example: python find_commit_index.py {sha} --repo-path d:\\r\\cr\\src")
        return
    
    # Search git repository
    index = find_index_in_git(args.repo_path, sha)
    if index is not None:
        print(f"\n   Use: python lookup_commit.py {index} --repo-path {args.repo_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
