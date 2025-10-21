#!/usr/bin/env python3
"""Check for overlapping date ranges and merge if needed."""

import json
from pathlib import Path
from datetime import datetime

status_file = Path('data/status.json')
with open(status_file) as f:
    status = json.load(f)

ranges = status['processed_ranges']

print("Current ranges:")
print("-" * 80)
for r in ranges:
    start = r['start_commit']['date']
    end = r['end_commit']['date']
    print(f"Range {r['range_id']}: {r.get('date_range', 'N/A')}")
    print(f"  Status: {r['status']}")
    print(f"  Start: {start} ({r['start_commit']['sha'][:8]})")
    print(f"  End:   {end} ({r['end_commit']['sha'][:8]})")
    print(f"  Commits: {r['commits_count']}")
    print()

# Check for overlaps
print("\nChecking for overlaps...")
print("-" * 80)

overlaps = []
for i, r1 in enumerate(ranges):
    for j, r2 in enumerate(ranges[i+1:], i+1):
        # Parse dates
        r1_start = datetime.fromisoformat(r1['start_commit']['date'].replace('Z', '+00:00'))
        r1_end = datetime.fromisoformat(r1['end_commit']['date'].replace('Z', '+00:00'))
        r2_start = datetime.fromisoformat(r2['start_commit']['date'].replace('Z', '+00:00'))
        r2_end = datetime.fromisoformat(r2['end_commit']['date'].replace('Z', '+00:00'))
        
        # Check for overlap (ranges overlap if one starts before the other ends)
        if r1_start <= r2_end and r2_start <= r1_end:
            overlaps.append((r1['range_id'], r2['range_id']))
            print(f"⚠️  OVERLAP: Range {r1['range_id']} and Range {r2['range_id']}")
            print(f"    Range {r1['range_id']}: {r1_start.date()} to {r1_end.date()}")
            print(f"    Range {r2['range_id']}: {r2_start.date()} to {r2_end.date()}")
            print()

if not overlaps:
    print("✅ No overlaps found!")
else:
    print(f"\n⚠️  Found {len(overlaps)} overlap(s) that should be merged")
