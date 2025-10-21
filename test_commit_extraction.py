#!/usr/bin/env python3
"""
Quick test to verify commit extraction and ordering from Chromium repo.
Tests small date ranges to understand commit structure.
"""

import sys
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rag_system.data.chromium import ChromiumDataExtractor

def test_commit_extraction(repo_path: str, start_date: str, end_date: str, max_commits: int = 100):
    """Test commit extraction from a date range."""
    print("\n" + "="*70)
    print("COMMIT EXTRACTION TEST")
    print("="*70)
    
    print(f"\nRepository: {repo_path}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Max commits: {max_commits}")
    
    # Initialize extractor
    extractor = ChromiumDataExtractor(repo_path)
    
    # Parse dates
    since = datetime.fromisoformat(start_date)
    until = datetime.fromisoformat(end_date)
    
    print(f"\n📊 Extracting commits...")
    
    # Extract commits
    commits = []
    for batch in extractor.extract_commits(
        max_count=max_commits,
        since=since,
        until=until,
        include_diffs=False,
        batch_size=50
    ):
        commits.extend(batch)
        print(f"  Batch: {len(batch)} commits")
    
    print(f"\n✓ Total extracted: {len(commits)} commits")
    
    if not commits:
        print("\n⚠️  No commits found in this range!")
        return
    
    # Analyze commits
    print("\n" + "="*70)
    print("COMMIT ANALYSIS")
    print("="*70)
    
    # Sort by date to check ordering
    sorted_commits = sorted(commits, key=lambda c: c.commit_date)
    
    print(f"\nFirst commit (by date):")
    first = sorted_commits[0]
    print(f"  SHA: {first.sha}")
    print(f"  Date: {first.commit_date}")
    print(f"  Author: {first.author_name}")
    print(f"  Message: {first.message[:80]}")
    
    print(f"\nLast commit (by date):")
    last = sorted_commits[-1]
    print(f"  SHA: {last.sha}")
    print(f"  Date: {last.commit_date}")
    print(f"  Author: {last.author_name}")
    print(f"  Message: {last.message[:80]}")
    
    # Check date distribution
    print(f"\n📅 Date Distribution:")
    date_counts = {}
    for commit in commits:
        date_str = commit.commit_date.strftime('%Y-%m-%d')
        date_counts[date_str] = date_counts.get(date_str, 0) + 1
    
    for date_str in sorted(date_counts.keys())[:10]:
        print(f"  {date_str}: {date_counts[date_str]} commits")
    if len(date_counts) > 10:
        print(f"  ... ({len(date_counts) - 10} more dates)")
    
    # Check ordering
    print(f"\n🔍 Ordering Check:")
    is_chronological = all(
        sorted_commits[i].commit_date <= sorted_commits[i+1].commit_date
        for i in range(len(sorted_commits)-1)
    )
    extraction_order_chronological = all(
        commits[i].commit_date <= commits[i+1].commit_date
        for i in range(len(commits)-1)
    )
    
    print(f"  Commits are chronological when sorted: {is_chronological}")
    print(f"  Commits extracted in chronological order: {extraction_order_chronological}")
    
    if not extraction_order_chronological:
        print("\n  ⚠️  Extraction order is NOT chronological!")
        print("  This means git log is not returning commits in date order.")
        print("  We need to sort commits by date before processing.")
    
    # Sample commits
    print(f"\n📝 Sample Commits (first 5 as extracted):")
    for i, commit in enumerate(commits[:5]):
        print(f"\n  [{i+1}] {commit.sha[:8]}")
        print(f"      Date: {commit.commit_date}")
        print(f"      Message: {commit.message[:60]}")
    
    # Check for duplicates
    print(f"\n🔎 Duplicate Check:")
    sha_set = set(c.sha for c in commits)
    print(f"  Unique commits: {len(sha_set)}")
    print(f"  Total commits: {len(commits)}")
    if len(sha_set) < len(commits):
        print(f"  ⚠️  Found {len(commits) - len(sha_set)} duplicates!")
    else:
        print(f"  ✓ No duplicates")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Commits extracted: {len(commits)}")
    print(f"Date range: {sorted_commits[0].commit_date.date()} to {sorted_commits[-1].commit_date.date()}")
    print(f"Time span: {(sorted_commits[-1].commit_date - sorted_commits[0].commit_date).days} days")
    print(f"Unique commits: {len(sha_set)}")
    print(f"Extraction order: {'Chronological' if extraction_order_chronological else 'NOT chronological (needs sorting)'}")

def run_all_tests():
    """Run tests on multiple date ranges."""
    test_ranges = [
        ("2024-01-01", "2024-01-03", 50),   # 2 days
        ("2024-07-01", "2024-07-03", 50),   # 2 days
        ("2024-10-01", "2024-10-03", 50),   # 2 days
    ]
    
    repo_path = "d:\\r\\cr\\src"
    
    for start, end, max_commits in test_ranges:
        test_commit_extraction(repo_path, start, end, max_commits)
        print("\n" + "="*70 + "\n")
        input("Press Enter to continue to next test range...")

def main():
    parser = argparse.ArgumentParser(description="Test commit extraction")
    parser.add_argument('--repo-path', default='d:\\r\\cr\\src', help='Chromium repo path')
    parser.add_argument('--start-date', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-08', help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-commits', type=int, default=100, help='Max commits to extract')
    parser.add_argument('--run-all', action='store_true', help='Run all test ranges')
    
    args = parser.parse_args()
    
    if args.run_all:
        run_all_tests()
    else:
        test_commit_extraction(args.repo_path, args.start_date, args.end_date, args.max_commits)

if __name__ == '__main__':
    main()
