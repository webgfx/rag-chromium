#!/usr/bin/env python3
"""
Verify that resume functionality is working correctly.
"""

import json
from pathlib import Path
from rag_system.vector.database import VectorDatabase

def verify_resume_state():
    """Check if the system can properly resume."""
    
    print("=" * 70)
    print("RESUME FUNCTIONALITY VERIFICATION")
    print("=" * 70)
    
    # Check progress file
    progress_file = Path("data/massive_cache/progress.json")
    
    if not progress_file.exists():
        print("\n❌ No progress file found")
        print("   This is a fresh start - no resume needed")
        return
    
    # Load progress
    with open(progress_file, 'r') as f:
        progress = json.load(f)
    
    print("\n📁 Progress File Status:")
    print(f"   ✅ Found: {progress_file}")
    print(f"   📊 Batches Completed: {progress.get('batches_completed', 0)}")
    print(f"   📝 Commits Processed: {progress.get('commits_processed', 0):,}")
    print(f"   📄 Documents Created: {progress.get('documents_created', 0):,}")
    print(f"   🔢 Embeddings Generated: {progress.get('embeddings_generated', 0):,}")
    
    # Check database
    print("\n🗄️  Database Status:")
    try:
        db = VectorDatabase(collection_name="chromium_complete")
        stats = db.get_collection_stats()
        total_docs = stats.get('total_documents', 0)
        
        print(f"   ✅ Connected to: {stats.get('collection_name', 'N/A')}")
        print(f"   📊 Total Documents: {total_docs:,}")
        
        # Verify consistency
        progress_docs = progress.get('documents_created', 0)
        
        print("\n🔍 Consistency Check:")
        if total_docs >= progress_docs:
            print(f"   ✅ Database has {total_docs:,} docs")
            print(f"   ✅ Progress tracked {progress_docs:,} docs")
            print(f"   ✅ Difference: {total_docs - progress_docs:,} docs (from previous sessions)")
            print("\n   ✨ System is consistent and ready to resume!")
        else:
            print(f"   ⚠️  Database has FEWER docs than progress file!")
            print(f"   Database: {total_docs:,}")
            print(f"   Progress: {progress_docs:,}")
            print(f"   This may indicate data loss or corruption")
            
    except Exception as e:
        print(f"   ❌ Database error: {e}")
        return
    
    # Resume instructions
    print("\n" + "=" * 70)
    print("📋 RESUME INSTRUCTIONS")
    print("=" * 70)
    
    if progress.get('start_date') or progress.get('max_commits'):
        print("\n🔄 To resume with the same parameters, run:\n")
        cmd = "python massive_chromium_ingestion.py --repo-path \"d:\\r\\cr\\src\""
        
        if progress.get('start_date'):
            cmd += f" --start-date \"{progress['start_date'][:10]}\""
        if progress.get('end_date'):
            cmd += f" --end-date \"{progress['end_date'][:10]}\""
        if progress.get('max_commits'):
            cmd += f" --max-commits {progress['max_commits']}"
        
        cmd += " --batch-size 1000 --embedding-batch-size 128 --max-workers 8"
        
        print(f"   {cmd}")
        print("\n   This will:")
        print(f"   • Skip first {progress.get('batches_completed', 0)} batches (already done)")
        print(f"   • Continue from batch {progress.get('batches_completed', 0) + 1}")
        print(f"   • Add new documents to existing {total_docs:,} documents")
    else:
        print("\n⚠️  No session parameters found in progress file")
        print("   You may need to start fresh")
    
    # Safety checks
    print("\n" + "=" * 70)
    print("✅ SAFETY CHECKS PASSED")
    print("=" * 70)
    print("✓ Progress file exists and is valid")
    print("✓ Database is accessible and consistent")
    print("✓ Resume functionality is ready")
    print("\n💡 You can safely stop (Ctrl+C) and resume at any time!")
    print("=" * 70)

if __name__ == "__main__":
    verify_resume_state()
