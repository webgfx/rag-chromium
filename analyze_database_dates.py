#!/usr/bin/env python3
"""Analyze actual commit dates in the database to verify coverage."""

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from rag_system.vector import VectorDatabase

def analyze_database():
    """Analyze what dates are actually in the database."""
    
    # Connect to database
    db = VectorDatabase()
    
    print("Querying database for all documents...")
    
    # Get all points with their metadata
    all_points = []
    offset = None
    
    while True:
        response = db.client.scroll(
            collection_name="chromium_complete",
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        
        points, offset = response
        all_points.extend(points)
        
        if offset is None:
            break
        
        print(f"  Loaded {len(all_points)} documents...", end='\r')
    
    print(f"\n✅ Loaded {len(all_points)} documents total")
    
    # Extract dates
    dates_by_month = defaultdict(int)
    dates_by_year = defaultdict(int)
    commit_dates = []
    
    for point in all_points:
        payload = point.payload
        commit_date_str = payload.get('commit_date')
        
        if commit_date_str:
            try:
                commit_date = datetime.fromisoformat(commit_date_str.replace('Z', '+00:00'))
                commit_dates.append(commit_date)
                
                year_month = commit_date.strftime('%Y-%m')
                year = commit_date.strftime('%Y')
                
                dates_by_month[year_month] += 1
                dates_by_year[year] += 1
            except Exception:
                pass
    
    if not commit_dates:
        print("❌ No commit dates found in database")
        return
    
    commit_dates.sort()
    
    print(f"\n📅 Date Range in Database:")
    print(f"   Earliest: {commit_dates[0].strftime('%Y-%m-%d')}")
    print(f"   Latest:   {commit_dates[-1].strftime('%Y-%m-%d')}")
    
    print(f"\n📊 Commits by Year:")
    for year in sorted(dates_by_year.keys()):
        print(f"   {year}: {dates_by_year[year]:,} commits")
    
    print(f"\n📊 Commits by Month (showing non-zero months):")
    for year_month in sorted(dates_by_month.keys()):
        count = dates_by_month[year_month]
        if count > 0:
            print(f"   {year_month}: {count:,} commits")
    
    # Identify gaps
    print(f"\n🔍 Checking for gaps...")
    
    current_month = datetime(commit_dates[0].year, commit_dates[0].month, 1)
    end_month = datetime(commit_dates[-1].year, commit_dates[-1].month, 1)
    
    gaps = []
    while current_month <= end_month:
        month_key = current_month.strftime('%Y-%m')
        if dates_by_month[month_key] == 0:
            gaps.append(month_key)
        
        # Move to next month
        if current_month.month == 12:
            current_month = datetime(current_month.year + 1, 1, 1)
        else:
            current_month = datetime(current_month.year, current_month.month + 1, 1)
    
    if gaps:
        print(f"\n⚠️  Found {len(gaps)} month(s) with NO commits:")
        for gap in gaps:
            print(f"   {gap}")
    else:
        print("\n✅ No gaps found - all months have commits")
    
    # Check specific ranges mentioned
    print(f"\n🎯 Checking specific date ranges:")
    
    ranges_to_check = [
        ("2024-03-06", "2024-04-10", "March-April 2024"),
        ("2024-04-01", "2024-04-30", "April 2024"),
        ("2024-07-01", "2024-07-31", "July 2024"),
        ("2025-08-01", "2025-09-30", "Aug-Sept 2025"),
        ("2025-10-01", "2025-10-20", "October 2025"),
    ]
    
    for start_str, end_str, label in ranges_to_check:
        start = datetime.fromisoformat(start_str)
        end = datetime.fromisoformat(end_str)
        
        count = sum(1 for d in commit_dates if start <= d <= end)
        print(f"   {label}: {count:,} commits")

if __name__ == '__main__':
    analyze_database()
