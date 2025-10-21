#!/usr/bin/env python3
"""
Optimize processed_ranges in status.json by:
1. Combining adjacent/overlapping ranges
2. Removing unnecessary fields (date, message, author)
3. Keeping only essential fields: range_id, start.index, start.sha, end.index, end.sha, commits_count
"""

import json
from pathlib import Path
from typing import List, Dict, Any


def load_status() -> Dict[str, Any]:
    """Load status.json file."""
    status_file = Path("data/status.json")
    if not status_file.exists():
        print("❌ status.json not found")
        return None
    
    with open(status_file) as f:
        return json.load(f)


def save_status(status: Dict[str, Any]) -> None:
    """Save status.json file."""
    status_file = Path("data/status.json")
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    print(f"✓ Saved optimized status to {status_file}")


def clean_range_fields(range_data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove unnecessary fields from a range, keeping only essential data including dates."""
    # Handle both 'range_id' and 'id'
    range_id = range_data.get('id', range_data.get('range_id', 0))
    
    return {
        'id': range_id,
        'start': {
            'index': range_data['start']['index'],
            'sha': range_data['start']['sha'],
            'date': range_data['start'].get('date')
        },
        'end': {
            'index': range_data['end']['index'],
            'sha': range_data['end']['sha'],
            'date': range_data['end'].get('date')
        }
    }


def can_combine_ranges(range1: Dict[str, Any], range2: Dict[str, Any]) -> bool:
    """
    Check if two ranges can be combined.
    Ranges can be combined if they are adjacent or overlapping.
    """
    # Get indices
    r1_start = range1['start']['index']
    r1_end = range1['end']['index']
    r2_start = range2['start']['index']
    r2_end = range2['end']['index']
    
    # Check if adjacent (allowing for +/- 1 gap due to inclusive/exclusive boundaries)
    if r1_end >= r2_start - 1 and r1_end <= r2_start + 1:
        return True
    
    if r2_end >= r1_start - 1 and r2_end <= r1_start + 1:
        return True
    
    # Check if overlapping
    if (r1_start <= r2_start <= r1_end) or (r1_start <= r2_end <= r1_end):
        return True
    
    if (r2_start <= r1_start <= r2_end) or (r2_start <= r1_end <= r2_end):
        return True
    
    return False


def combine_ranges(range1: Dict[str, Any], range2: Dict[str, Any]) -> Dict[str, Any]:
    """Combine two ranges into one."""
    r1_start = range1['start']['index']
    r1_end = range1['end']['index']
    r2_start = range2['start']['index']
    r2_end = range2['end']['index']
    
    # Determine overall start and end (preserve dates)
    if r1_start < r2_start:
        start_index = r1_start
        start_sha = range1['start']['sha']
        start_date = range1['start'].get('date')
    else:
        start_index = r2_start
        start_sha = range2['start']['sha']
        start_date = range2['start'].get('date')
    
    if r1_end > r2_end:
        end_index = r1_end
        end_sha = range1['end']['sha']
        end_date = range1['end'].get('date')
    else:
        end_index = r2_end
        end_sha = range2['end']['sha']
        end_date = range2['end'].get('date')
    
    return {
        'id': range1.get('id', range1.get('range_id', 0)),
        'start': {
            'index': start_index,
            'sha': start_sha,
            'date': start_date
        },
        'end': {
            'index': end_index,
            'sha': end_sha,
            'date': end_date
        }
    }


def optimize_ranges(ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Optimize ranges by:
    1. Cleaning unnecessary fields
    2. Sorting by start index
    3. Combining adjacent/overlapping ranges
    """
    if not ranges:
        return []
    
    print(f"\n📊 Initial state: {len(ranges)} ranges")
    
    # Clean all ranges first
    cleaned_ranges = [clean_range_fields(r) for r in ranges]
    
    # Sort by start index
    cleaned_ranges.sort(key=lambda r: r['start']['index'])
    
    # Combine adjacent/overlapping ranges
    optimized = []
    current = cleaned_ranges[0]
    combined_count = 0
    
    for next_range in cleaned_ranges[1:]:
        if can_combine_ranges(current, next_range):
            current_id = current.get('id', current.get('range_id', 0))
            next_id = next_range.get('id', next_range.get('range_id', 0))
            print(f"  → Combining range {current_id} (idx {current['start']['index']}-{current['end']['index']}) "
                  f"with range {next_id} (idx {next_range['start']['index']}-{next_range['end']['index']})")
            current = combine_ranges(current, next_range)
            combined_count += 1
        else:
            optimized.append(current)
            current = next_range
    
    # Add the last range
    optimized.append(current)
    
    # Renumber ids sequentially
    for i, range_data in enumerate(optimized, start=1):
        range_data['id'] = i
    
    print(f"\n✓ Optimized: {len(ranges)} ranges → {len(optimized)} ranges ({combined_count} combinations)")
    print(f"\n📋 Final ranges:")
    for r in optimized:
        commits_count = r['end']['index'] - r['start']['index'] + 1
        print(f"  Range {r['id']}: index {r['start']['index']:,} - {r['end']['index']:,} ({commits_count:,} commits)")
    
    return optimized


def main():
    print("=" * 80)
    print("RANGE OPTIMIZER")
    print("=" * 80)
    
    # Load status
    status = load_status()
    if not status:
        return
    
    # Get processed ranges
    ranges = status.get('processed_ranges', [])
    if not ranges:
        print("No processed ranges found")
        return
    
    # Optimize ranges
    optimized_ranges = optimize_ranges(ranges)
    
    # Update status
    status['processed_ranges'] = optimized_ranges
    
    # Save
    save_status(status)
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
