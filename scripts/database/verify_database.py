#!/usr/bin/env python3
"""
Database verification and status reconstruction tool.

This script scans the Qdrant database and provides:
1. Database health check
2. Complete list of what's been processed
3. Reconstruction of status.json from actual database contents

Use this whenever you need to:
- Verify what's actually in the database
- Recover from status.json corruption/loss
- Validate that status.json matches database reality
"""

import json
import logging
from pathlib import Path
from qdrant_client import QdrantClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def verify_database():
    """Quick database verification without full rebuild."""
    
    print("\n" + "="*60)
    print("DATABASE VERIFICATION")
    print("="*60 + "\n")
    
    db_path = Path("data/cache/qdrant_db")
    if not db_path.exists():
        print("❌ Database not found at:", db_path)
        return False
    
    try:
        client = QdrantClient(path=str(db_path))
        collection_name = "chromium_complete"
        
        # Get collection info
        collection_info = client.get_collection(collection_name)
        total_docs = collection_info.points_count
        
        print(f"✅ Database Status: HEALTHY")
        print(f"   Location: {db_path}")
        print(f"   Collection: {collection_name}")
        print(f"   Total Documents: {total_docs:,}")
        vectors_count = collection_info.vectors_count if collection_info.vectors_count else 0
        indexed_count = collection_info.indexed_vectors_count if collection_info.indexed_vectors_count else 0
        print(f"   Vectors: {vectors_count:,}")
        print(f"   Indexed Vectors: {indexed_count:,}")
        
        if total_docs == 0:
            print("\n⚠️  Database is empty!")
            return True
        
        # Get sample documents to verify structure
        records, _ = client.scroll(
            collection_name=collection_name,
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        print(f"\n📊 Sample Documents (showing {len(records)}):")
        for i, record in enumerate(records, 1):
            payload = record.payload
            print(f"\n   Document {i}:")
            print(f"   - Commit SHA: {payload.get('commit_sha', 'N/A')[:12]}...")
            print(f"   - Date: {payload.get('commit_date', 'N/A')}")
            print(f"   - Author: {payload.get('author', 'N/A')}")
            print(f"   - Message: {payload.get('message', 'N/A')[:60]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return False


def compare_status_with_database():
    """Compare status.json with actual database contents."""
    
    print("\n" + "="*60)
    print("STATUS FILE VS DATABASE COMPARISON")
    print("="*60 + "\n")
    
    status_file = Path('data/status.json')
    
    # Check if status file exists
    if not status_file.exists():
        print("⚠️  status.json not found!")
        print("   Location checked:", status_file.absolute())
        print("   → Use 'python rebuild_status.py' to create it from database")
        return
    
    # Load status file
    with open(status_file) as f:
        status = json.load(f)
    
    status_docs = status.get('database', {}).get('total_documents', 0)
    status_ranges = len(status.get('processed_ranges', []))
    status_commits = status.get('progress', {}).get('commits_processed', 0)
    
    # Get database info
    db_path = Path("data/cache/qdrant_db")
    try:
        client = QdrantClient(path=str(db_path))
        collection_info = client.get_collection("chromium_complete")
        db_docs = collection_info.points_count
    except Exception as e:
        print(f"❌ Cannot read database: {e}")
        return
    
    # Compare
    print(f"Status File:")
    print(f"   - Documents: {status_docs:,}")
    print(f"   - Commits: {status_commits:,}")
    print(f"   - Ranges: {status_ranges}")
    print(f"   - Last Updated: {status.get('timestamp', 'Unknown')}")
    
    print(f"\nDatabase:")
    print(f"   - Documents: {db_docs:,}")
    
    # Check if they match
    if status_docs == db_docs:
        print(f"\n✅ Status file matches database!")
    else:
        diff = abs(status_docs - db_docs)
        print(f"\n⚠️  MISMATCH detected!")
        print(f"   Difference: {diff:,} documents")
        if status_docs < db_docs:
            print(f"   → Status file is BEHIND (missing {diff:,} documents)")
        else:
            print(f"   → Status file is AHEAD (has {diff:,} extra in count)")
        print(f"\n   💡 Recommendation: Run 'python rebuild_status.py' to sync")


def main():
    """Run full verification and comparison."""
    
    import sys
    
    # Check arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--rebuild':
            print("\n🔄 Running full database scan and status rebuild...")
            from rebuild_status import rebuild_status_file
            rebuild_status_file()
            return
        elif sys.argv[1] == '--help':
            print(__doc__)
            print("\nUsage:")
            print("  python verify_database.py              # Quick verification")
            print("  python verify_database.py --rebuild    # Full rebuild")
            print("  python verify_database.py --help       # Show this help")
            return
    
    # Run verification
    db_ok = verify_database()
    
    if db_ok:
        compare_status_with_database()
    
    print("\n" + "="*60)
    print("\n💡 Commands:")
    print("   python verify_database.py --rebuild    # Rebuild status.json from database")
    print("   python rebuild_status.py               # Same as --rebuild")
    print("\n")


if __name__ == '__main__':
    main()
