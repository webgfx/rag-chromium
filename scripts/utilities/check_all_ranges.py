#!/usr/bin/env python3
"""Check all processed commit ranges in the database"""

import chromadb
from collections import defaultdict
from datetime import datetime

# Connect to database
client = chromadb.PersistentClient(path='data/cache/vector_db')
collection = client.get_collection('chromium_complete')

print(f"Total documents: {collection.count():,}\n")

# Get all documents with metadata
print("Fetching all documents (this may take a moment)...")
all_docs = collection.get(include=['metadatas'])

# Group by commit SHA to find unique commits
commits = defaultdict(lambda: {'date': None, 'doc_count': 0})

for meta in all_docs['metadatas']:
    sha = meta.get('commit_sha', 'unknown')
    date_str = meta.get('commit_date', None)
    
    if sha != 'unknown':
        commits[sha]['doc_count'] += 1
        if date_str and not commits[sha]['date']:
            # Parse date
            try:
                # Handle different date formats
                if 'T' in date_str:
                    date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                else:
                    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                commits[sha]['date'] = date
            except:
                pass

print(f"\nUnique commits: {len(commits):,}")

# Find date range
dates = [info['date'] for info in commits.values() if info['date']]
if dates:
    earliest = min(dates)
    latest = max(dates)
    print(f"\nDate range:")
    print(f"  Earliest: {earliest.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Latest:   {latest.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Find commits in different time periods to identify potential gaps
    print("\nCommits by year:")
    by_year = defaultdict(int)
    for sha, info in commits.items():
        if info['date']:
            by_year[info['date'].year] += 1
    
    for year in sorted(by_year.keys()):
        print(f"  {year}: {by_year[year]:,} commits")
else:
    print("\nNo date information found")

print("\nSample commits (first 10 with dates):")
dated_commits = [(sha, info) for sha, info in commits.items() if info['date']]
dated_commits.sort(key=lambda x: x[1]['date'], reverse=True)
for sha, info in dated_commits[:10]:
    print(f"  {sha[:8]} - {info['date'].strftime('%Y-%m-%d %H:%M:%S')} ({info['doc_count']} docs)")
