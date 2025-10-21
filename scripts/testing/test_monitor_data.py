#!/usr/bin/env python3
"""Test what data the monitor loads."""

import json
from pathlib import Path

status_file = Path('data/status.json')
with open(status_file) as f:
    status = json.load(f)

print(f"Total documents: {status['database']['total_documents']}")
print(f"Processed ranges: {len(status['processed_ranges'])}")
print(f"\nRange details:")
for i, r in enumerate(status['processed_ranges'][:3], 1):
    print(f"  Range {i}: {r.get('commits_count', 0)} commits, status: {r.get('status', 'N/A')}")
    print(f"    Dates: {r.get('date_range', 'N/A')}")

print(f"\nCurrent range: {status.get('current_range', {}).get('commits_count', 0)} commits")
print(f"Progress: {status['progress']['commits_processed']} commits, {status['progress']['batches_completed']} batches")
