"""
Quick commit lookup by index.
Usage: python lookup_commit.py <index> [--repo-path REPO_PATH]
"""

import sys
import subprocess
from pathlib import Path
from qdrant_client import QdrantClient
import warnings
warnings.filterwarnings('ignore')

def get_commit_from_git(repo_path: str, index: int):
    """Get commit info from git repository by index."""
    try:
        # Get all commits in reverse order (oldest first)
        result = subprocess.run(
            ['git', 'log', '--all', '--reverse', '--format=%H'],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        commit_shas = result.stdout.strip().split('\n')
        
        if index < 0 or index >= len(commit_shas):
            print(f"❌ Index {index} out of range (valid: 0 to {len(commit_shas)-1})")
            return None
        
        sha = commit_shas[index]
        
        # Get commit details
        result = subprocess.run(
            ['git', 'show', '--no-patch', '--format=%H%n%ai%n%an%n%s', sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        lines = result.stdout.strip().split('\n')
        commit_sha = lines[0]
        commit_date = lines[1]
        author = lines[2]
        message = lines[3] if len(lines) > 3 else ''
        
        # Get file stats
        result = subprocess.run(
            ['git', 'show', '--stat', '--format=', sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        stats_lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
        files_changed = 0
        additions = 0
        deletions = 0
        
        if stats_lines:
            # Last line typically has summary: "X files changed, Y insertions(+), Z deletions(-)"
            summary = stats_lines[-1]
            if 'file' in summary:
                parts = summary.split(',')
                for part in parts:
                    if 'file' in part:
                        files_changed = int(''.join(filter(str.isdigit, part)))
                    elif 'insertion' in part:
                        additions = int(''.join(filter(str.isdigit, part)))
                    elif 'deletion' in part:
                        deletions = int(''.join(filter(str.isdigit, part)))
        
        return {
            'sha': commit_sha,
            'date': commit_date,
            'author': author,
            'message': message,
            'files_changed': files_changed,
            'additions': additions,
            'deletions': deletions
        }
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git error: {e}")
        return None
    except Exception as e:
        print(f"❌ Error querying git: {e}")
        return None

def lookup_commit(index: int, repo_path: str = None):
    """Look up commit by index in database first, then git if not found."""
    db_path = Path("data/cache/qdrant_db")
    
    # Try database first
    in_database = False
    try:
        client = QdrantClient(path=str(db_path))
        collection_name = "chromium_complete"
        
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        # Search for documents with this commit_index
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="commit_index",
                        match=MatchValue(value=index)
                    )
                ]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        records, _ = results
        
        if records:
            in_database = True
            payload = records[0].payload
            
            print(f"\n📍 Commit at index {index} (IN DATABASE):")
            print(f"   SHA:     {payload.get('commit_sha', 'N/A')}")
            print(f"   Date:    {payload.get('commit_date', 'N/A')}")
            print(f"   Author:  {payload.get('author', 'N/A')}")
            print(f"   Message: {payload.get('message', 'N/A')[:100]}")
            print(f"   Files:   {payload.get('files_changed', 'N/A')} changed")
            print(f"   Changes: +{payload.get('additions', 0)} -{payload.get('deletions', 0)}")
            
            # Count total documents for this commit
            all_results = client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="commit_index",
                            match=MatchValue(value=index)
                        )
                    ]
                ),
                limit=100,
                with_payload=False,
                with_vectors=False
            )
            
            total_docs = len(all_results[0])
            print(f"   Documents: {total_docs} chunks in database")
            return
            
        client.close()
    except Exception as e:
        print(f"⚠️  Database query failed: {e}")
    
    # Not in database, try git
    if not in_database:
        if not repo_path:
            print(f"\n❌ Commit index {index} not found in database.")
            print(f"   Use --repo-path to query git repository directly.")
            print(f"   Example: python lookup_commit.py {index} --repo-path d:\\r\\cr\\src")
            return
        
        print(f"\n🔍 Commit index {index} not in database, querying git repository...")
        commit_info = get_commit_from_git(repo_path, index)
        
        if commit_info:
            print(f"\n📍 Commit at index {index} (FROM GIT):")
            print(f"   SHA:     {commit_info['sha']}")
            print(f"   Date:    {commit_info['date']}")
            print(f"   Author:  {commit_info['author']}")
            print(f"   Message: {commit_info['message'][:100]}")
            print(f"   Files:   {commit_info['files_changed']} changed")
            print(f"   Changes: +{commit_info['additions']} -{commit_info['deletions']}")
            print(f"   Documents: 0 chunks in database (not ingested yet)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Look up commit by index')
    parser.add_argument('index', type=int, help='Commit index to look up')
    parser.add_argument('--repo-path', type=str, help='Path to git repository (for commits not in database)')
    
    args = parser.parse_args()
    
    try:
        lookup_commit(args.index, args.repo_path)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
