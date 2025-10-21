#!/usr/bin/env python3
"""
Purge database and run controlled test with small date ranges.
Ensures clean state and verifies commit tracking works correctly.
"""

import sys
import shutil
import json
from pathlib import Path
from datetime import datetime
import argparse

from qdrant_client import QdrantClient

def purge_database(db_path: Path, backup: bool = True):
    """Completely purge the Qdrant database."""
    print("\n" + "="*70)
    print("DATABASE PURGE")
    print("="*70)
    
    if not db_path.exists():
        print(f"✓ Database doesn't exist at {db_path}")
        return
    
    # Backup if requested
    if backup:
        backup_path = db_path.parent / f"{db_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n📦 Creating backup: {backup_path}")
        shutil.copytree(db_path, backup_path)
        print(f"✓ Backup created")
    
    # Remove database
    print(f"\n🗑️  Removing database: {db_path}")
    shutil.rmtree(db_path)
    print(f"✓ Database purged")

def purge_cache(cache_dir: Path, backup: bool = True):
    """Purge progress files and cache."""
    print("\n" + "="*70)
    print("CACHE PURGE")
    print("="*70)
    
    if not cache_dir.exists():
        print(f"✓ Cache doesn't exist at {cache_dir}")
        return
    
    files_to_remove = [
        "progress.json",
        "status.json",
        "all_commit_ranges.json",
        "stats.json"
    ]
    
    for filename in files_to_remove:
        file_path = cache_dir / filename
        if file_path.exists():
            if backup:
                backup_path = file_path.parent / f"{file_path.name}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                shutil.copy2(file_path, backup_path)
                print(f"📦 Backed up: {filename} -> {backup_path.name}")
            
            file_path.unlink()
            print(f"🗑️  Removed: {filename}")
    
    # Remove checkpoints
    checkpoint_dir = cache_dir / "checkpoints"
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
        print(f"🗑️  Removed: checkpoints/")
    
    print(f"✓ Cache purged")

def verify_clean_state(db_path: Path, cache_dir: Path):
    """Verify everything is clean."""
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    issues = []
    
    # Check database
    if db_path.exists():
        issues.append(f"Database still exists: {db_path}")
    else:
        print("✓ Database removed")
    
    # Check cache files
    cache_files = ["progress.json", "status.json", "all_commit_ranges.json"]
    for filename in cache_files:
        if (cache_dir / filename).exists():
            issues.append(f"Cache file still exists: {filename}")
    
    if not issues:
        print("✓ All cache files removed")
    
    if issues:
        print("\n❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n✅ CLEAN STATE VERIFIED")
    return True

def create_test_config():
    """Create test configuration with 2 small date ranges."""
    print("\n" + "="*70)
    print("TEST CONFIGURATION")
    print("="*70)
    
    # Two small ranges for testing
    test_ranges = [
        {
            "name": "Test Range 1: Jan 2024",
            "start": "2024-01-01",
            "end": "2024-01-08",  # 1 week
            "expected_commits": "~50-200"
        },
        {
            "name": "Test Range 2: Jul 2024", 
            "start": "2024-07-01",
            "end": "2024-07-08",  # 1 week
            "expected_commits": "~50-200"
        }
    ]
    
    print("\n📋 Test Ranges:")
    for i, range_info in enumerate(test_ranges, 1):
        print(f"\n  Range {i}: {range_info['name']}")
        print(f"    Start: {range_info['start']}")
        print(f"    End:   {range_info['end']}")
        print(f"    Expected: {range_info['expected_commits']} commits")
    
    # Save test config
    config_path = Path("test_config.json")
    with open(config_path, 'w') as f:
        json.dump(test_ranges, f, indent=2)
    
    print(f"\n✓ Test config saved to: {config_path}")
    
    return test_ranges

def generate_test_commands(test_ranges):
    """Generate commands to run tests."""
    print("\n" + "="*70)
    print("TEST EXECUTION COMMANDS")
    print("="*70)
    
    print("\n📝 Run these commands to execute the test:\n")
    
    print("# Terminal 1: Start backup scheduler")
    print("python backup_scheduler.py --interval 1 --keep 5\n")
    
    print("# Terminal 2: Start monitor")
    print("streamlit run ingestion_monitor.py --server.port 8502\n")
    
    print("# Terminal 3: Run test ranges")
    for i, range_info in enumerate(test_ranges, 1):
        print(f"\n# --- Test Range {i}: {range_info['name']} ---")
        cmd = (
            f"python massive_chromium_ingestion.py "
            f"--repo-path \"d:\\r\\cr\\src\" "
            f"--start-date \"{range_info['start']}\" "
            f"--end-date \"{range_info['end']}\" "
            f"--batch-size 100 "
            f"--embedding-batch-size 64 "
            f"--max-workers 4"
        )
        print(cmd)
        
        print(f"\n# Wait for Range {i} to complete, then verify:")
        print("python -c \"from qdrant_client import QdrantClient; "
              "c=QdrantClient(path='data/cache/qdrant_db'); "
              "info=c.get_collection('chromium_complete'); "
              f"print(f'Range {i} Documents: {{info.points_count}}')\"")
        
        if i < len(test_ranges):
            print(f"\n# After Range {i} verification, continue to Range {i+1}...")
    
    print("\n# --- Final Verification ---")
    print("\n# Check total documents (should have both ranges)")
    print("python -c \"from qdrant_client import QdrantClient; "
          "c=QdrantClient(path='data/cache/qdrant_db'); "
          "info=c.get_collection('chromium_complete'); "
          "print(f'Total Documents: {info.points_count}')\"")
    
    print("\n# Check processed ranges (should show 2 ranges)")
    print("Get-Content data\\massive_cache\\all_commit_ranges.json | ConvertFrom-Json")
    
    print("\n# Check monitor status")
    print("Get-Content data\\status.json | ConvertFrom-Json | "
          "Select-Object -ExpandProperty processed_ranges | Format-Table")
    
    print("\n# View monitor dashboard")
    print("# Open: http://localhost:8502")
    print("#   - Should show 2 ranges in timeline")
    print("#   - Should show gap between Jan and Jul")
    print("#   - Should show correct commit counts for each range")

def main():
    parser = argparse.ArgumentParser(description="Purge database and setup test")
    parser.add_argument('--no-backup', action='store_true', help='Skip backup before purge')
    parser.add_argument('--db-path', default='data/cache/qdrant_db', help='Database path')
    parser.add_argument('--cache-dir', default='data/massive_cache', help='Cache directory')
    parser.add_argument('--purge-only', action='store_true', help='Only purge, skip test setup')
    
    args = parser.parse_args()
    
    db_path = Path(args.db_path)
    cache_dir = Path(args.cache_dir)
    backup = not args.no_backup
    
    print("\n" + "="*70)
    print("CHROMIUM RAG DATABASE PURGE & TEST SETUP")
    print("="*70)
    print(f"\nDatabase: {db_path}")
    print(f"Cache: {cache_dir}")
    print(f"Backup before purge: {backup}")
    
    if not args.purge_only:
        response = input("\n⚠️  This will DELETE all data and start fresh. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("\n❌ Cancelled")
            return
    
    # Purge database
    purge_database(db_path, backup)
    
    # Purge cache
    purge_cache(cache_dir, backup)
    
    # Verify clean state
    if not verify_clean_state(db_path, cache_dir):
        print("\n❌ Clean state verification failed!")
        return 1
    
    if args.purge_only:
        print("\n✅ Purge completed")
        return 0
    
    # Create test configuration
    test_ranges = create_test_config()
    
    # Generate test commands
    generate_test_commands(test_ranges)
    
    print("\n" + "="*70)
    print("✅ SETUP COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review the test commands above")
    print("  2. Start backup scheduler in Terminal 1")
    print("  3. Start monitor in Terminal 2")
    print("  4. Run test ranges in Terminal 3")
    print("  5. Verify timeline and ranges in monitor dashboard")
    print("\n" + "="*70 + "\n")

if __name__ == '__main__':
    sys.exit(main() or 0)
