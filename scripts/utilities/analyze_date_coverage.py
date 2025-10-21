#!/usr/bin/env python3
"""
Analyze actual date coverage in the database to identify gaps.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from qdrant_client import QdrantClient

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def analyze_coverage():
    """Analyze database for actual date coverage and gaps."""
    
    print("\n🔍 Analyzing Database Date Coverage...")
    print("=" * 70)
    
    # Connect to database
    db_path = Path("data/cache/qdrant_db")
    client = QdrantClient(path=str(db_path))
    
    collection_name = "chromium_complete"
    
    # Get total count
    collection_info = client.get_collection(collection_name)
    total_docs = collection_info.points_count
    print(f"\n📊 Total Documents: {total_docs:,}")
    
    # Scan all documents and extract dates
    print("\n📅 Scanning commit dates...")
    dates_by_month = defaultdict(int)
    dates_by_day = defaultdict(int)
    all_dates = []
    
    offset = None
    batch_size = 1000
    scanned = 0
    
    while True:
        records = client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, next_offset = records
        
        if not points:
            break
            
        for point in points:
            payload = point.payload
            commit_date_str = payload.get('commit_date', '')
            
            if commit_date_str:
                try:
                    # Parse date (handle different formats)
                    if 'T' in commit_date_str:
                        commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))
                    else:
                        commit_date = datetime.strptime(commit_date_str, '%Y-%m-%d %H:%M:%S')
                    
                    all_dates.append(commit_date)
                    
                    # Group by month
                    month_key = commit_date.strftime('%Y-%m')
                    dates_by_month[month_key] += 1
                    
                    # Group by day
                    day_key = commit_date.strftime('%Y-%m-%d')
                    dates_by_day[day_key] += 1
                    
                except Exception as e:
                    print(f"⚠️  Error parsing date '{commit_date_str}': {e}")
        
        scanned += len(points)
        if scanned % 5000 == 0:
            print(f"   Scanned {scanned:,}/{total_docs:,} documents...")
        
        offset = next_offset
        if next_offset is None:
            break
    
    print(f"\n✅ Scanned {scanned:,} documents with dates")
    
    # Sort dates
    all_dates.sort()
    
    if not all_dates:
        print("\n❌ No dates found in database!")
        return
    
    earliest = all_dates[0]
    latest = all_dates[-1]
    
    print(f"\n📅 Date Range:")
    print(f"   Earliest: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Latest:   {latest.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Span:     {(latest - earliest).days} days")
    
    # Monthly breakdown
    print(f"\n📊 Commits by Month:")
    print("=" * 70)
    sorted_months = sorted(dates_by_month.keys())
    
    total_commits = 0
    for month in sorted_months:
        count = dates_by_month[month]
        total_commits += count
        bar = "█" * min(50, count // 10)
        print(f"   {month}: {count:>5} commits {bar}")
    
    # Identify gaps (days with 0 commits between earliest and latest)
    print(f"\n🔍 Analyzing Gaps (days with 0 commits)...")
    print("=" * 70)
    
    current_date = earliest.date()
    end_date = latest.date()
    gap_ranges = []
    gap_start = None
    
    while current_date <= end_date:
        day_key = current_date.strftime('%Y-%m-%d')
        
        if day_key not in dates_by_day:
            # Gap found
            if gap_start is None:
                gap_start = current_date
        else:
            # Commits found, check if we were in a gap
            if gap_start is not None:
                gap_end = current_date - timedelta(days=1)
                gap_days = (gap_end - gap_start).days + 1
                
                # Only report gaps of 3+ days
                if gap_days >= 3:
                    gap_ranges.append((gap_start, gap_end, gap_days))
                
                gap_start = None
        
        current_date += timedelta(days=1)
    
    # Check if we ended in a gap
    if gap_start is not None:
        gap_end = end_date
        gap_days = (gap_end - gap_start).days + 1
        if gap_days >= 3:
            gap_ranges.append((gap_start, gap_end, gap_days))
    
    if gap_ranges:
        print(f"\n⚠️  Found {len(gap_ranges)} gaps (3+ days with no commits):\n")
        
        total_gap_days = 0
        for i, (start, end, days) in enumerate(gap_ranges, 1):
            total_gap_days += days
            print(f"   Gap {i}: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')} ({days} days)")
        
        print(f"\n   Total gap days: {total_gap_days}")
        print(f"   Coverage: {((end_date - earliest.date()).days + 1 - total_gap_days) / ((end_date - earliest.date()).days + 1) * 100:.1f}%")
    else:
        print("✅ No significant gaps found!")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print("=" * 70)
    print(f"   Total months with data: {len(sorted_months)}")
    print(f"   Total days with commits: {len(dates_by_day)}")
    print(f"   Average commits/day: {total_commits / len(dates_by_day):.1f}")
    print(f"   Max commits in a day: {max(dates_by_day.values())}")
    
    # Find busiest month
    busiest_month = max(dates_by_month.items(), key=lambda x: x[1])
    print(f"   Busiest month: {busiest_month[0]} ({busiest_month[1]} commits)")
    
    print("\n" + "=" * 70)
    print("✅ Analysis Complete!\n")

if __name__ == "__main__":
    analyze_coverage()
