#!/usr/bin/env python3
"""Reconstruct all commit ranges from the database"""

from qdrant_client import QdrantClient
from datetime import datetime
from collections import defaultdict
import json
from pathlib import Path

# Connect to Qdrant database
db_path = Path('data/cache/qdrant_db')
client = QdrantClient(path=str(db_path))
collection_name = 'chromium_complete'

# Get collection info
try:
    collection_info = client.get_collection(collection_name)
    total_docs = collection_info.points_count
    print(f"Total documents: {total_docs:,}\n")
except Exception as e:
    print(f"Error accessing collection: {e}")
    exit(1)

# Get all documents with metadata
print("Fetching all documents...")
# Qdrant requires scrolling through results
all_points = []
offset = None
batch_size = 1000

while True:
    results = client.scroll(
        collection_name=collection_name,
        limit=batch_size,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    points, offset = results
    all_points.extend(points)
    
    if offset is None:
        break
    print(f"Fetched {len(all_points):,} documents...", end='\r')

print(f"Fetched {len(all_points):,} documents total")

# Convert to format similar to ChromaDB
all_docs = {'metadatas': [p.payload for p in all_points]}

# Extract unique commits with dates
commits = {}
for meta in all_docs['metadatas']:
    sha = meta.get('commit_sha', None)
    date_str = meta.get('commit_date', None)
    
    if sha and date_str and sha not in commits:
        try:
            if 'T' in date_str:
                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            else:
                date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            commits[sha] = date
        except:
            pass

print(f"Found {len(commits):,} unique commits with dates\n")

# Sort by date
sorted_commits = sorted(commits.items(), key=lambda x: x[1])

# Find continuous ranges (gaps > 7 days indicate separate ranges)
ranges = []
current_range = {
    'start_sha': sorted_commits[0][0],
    'start_date': sorted_commits[0][1],
    'end_sha': sorted_commits[0][0],
    'end_date': sorted_commits[0][1],
    'commits': []
}

for sha, date in sorted_commits:
    current_range['commits'].append(sha)
    gap_days = (date - current_range['end_date']).days
    
    # If gap > 7 days, start new range
    if gap_days > 7:
        # Save current range
        ranges.append({
            'start_sha': current_range['start_sha'],
            'start_date': current_range['start_date'].isoformat(),
            'end_sha': current_range['end_sha'],
            'end_date': current_range['end_date'].isoformat(),
            'commits': len(current_range['commits'])
        })
        
        # Start new range
        current_range = {
            'start_sha': sha,
            'start_date': date,
            'end_sha': sha,
            'end_date': date,
            'commits': [sha]
        }
    else:
        # Continue current range
        current_range['end_sha'] = sha
        current_range['end_date'] = date

# Add final range
ranges.append({
    'start_sha': current_range['start_sha'],
    'start_date': current_range['start_date'].isoformat(),
    'end_sha': current_range['end_sha'],
    'end_date': current_range['end_date'].isoformat(),
    'commits': len(current_range['commits'])
})

# Merge consecutive ranges (where gap between end of one and start of next is ≤ 1 day)
merged_ranges = []
if ranges:
    current_merged = ranges[0].copy()
    current_merged['commits_count'] = current_merged['commits']
    
    for i in range(1, len(ranges)):
        prev_end = datetime.fromisoformat(current_merged['end_date'].replace('Z', '+00:00'))
        curr_start = datetime.fromisoformat(ranges[i]['start_date'].replace('Z', '+00:00'))
        gap_days = (curr_start - prev_end).days
        
        # If consecutive (gap ≤ 1 day), merge them
        if gap_days <= 1:
            current_merged['end_sha'] = ranges[i]['end_sha']
            current_merged['end_date'] = ranges[i]['end_date']
            current_merged['commits_count'] += ranges[i]['commits']
        else:
            # Save current merged range and start new one
            merged_ranges.append(current_merged)
            current_merged = ranges[i].copy()
            current_merged['commits_count'] = current_merged['commits']
    
    # Add final merged range
    merged_ranges.append(current_merged)
    ranges = merged_ranges

print(f"Found {len(ranges)} continuous ranges (after merging consecutive):\n")
for i, r in enumerate(ranges, 1):
    print(f"Range {i}:")
    print(f"  Start: {r['start_sha'][:8]} ({r['start_date']})")
    print(f"  End:   {r['end_sha'][:8]} ({r['end_date']})")
    print(f"  Commits: {r.get('commits_count', r['commits']):,}")
    print()

# Clean up the ranges data for saving
for r in ranges:
    if 'commits_count' in r:
        r['commits'] = r.pop('commits_count')

# Save to a new file for visualization
output_file = 'data/massive_cache/all_commit_ranges.json'
with open(output_file, 'w') as f:
    json.dump(ranges, f, indent=2)

print(f"Saved all ranges to: {output_file}")
