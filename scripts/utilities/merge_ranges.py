#!/usr/bin/env python3
"""Merge overlapping ranges in status.json."""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

def merge_overlapping_ranges(ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge overlapping ranges, keeping completed ranges intact."""
    if not ranges:
        return []
    
    # Separate processing and completed ranges
    processing_ranges = [r for r in ranges if r.get('status') == 'processing']
    completed_ranges = [r for r in ranges if r.get('status') == 'completed']
    
    if not processing_ranges:
        return ranges  # No processing ranges, nothing to merge
    
    if len(processing_ranges) > 1:
        print(f"⚠️  Warning: Multiple processing ranges found ({len(processing_ranges)})")
    
    # For each processing range, check if it overlaps with completed ranges
    merged_ranges = []
    ranges_to_remove = set()
    
    for proc_range in processing_ranges:
        proc_start = datetime.fromisoformat(proc_range['start_commit']['date'].replace('Z', '+00:00'))
        proc_end = datetime.fromisoformat(proc_range['end_commit']['date'].replace('Z', '+00:00'))
        
        overlapping_completed = []
        for i, comp_range in enumerate(completed_ranges):
            comp_start = datetime.fromisoformat(comp_range['start_commit']['date'].replace('Z', '+00:00'))
            comp_end = datetime.fromisoformat(comp_range['end_commit']['date'].replace('Z', '+00:00'))
            
            # Check if ranges overlap
            if proc_start <= comp_end and comp_start <= proc_end:
                overlapping_completed.append((i, comp_range))
                print(f"Range {proc_range['range_id']} overlaps with Range {comp_range['range_id']}")
        
        if overlapping_completed:
            print(f"\n🔄 Merging Range {proc_range['range_id']} (processing) with {len(overlapping_completed)} completed range(s)")
            
            # Mark overlapping completed ranges for removal
            for idx, _ in overlapping_completed:
                ranges_to_remove.add(idx)
            
            # Merge: Keep the processing range since it contains the superset of data
            # Just update the date_range to reflect the full span
            all_dates = [proc_start, proc_end]
            for _, comp_range in overlapping_completed:
                all_dates.append(datetime.fromisoformat(comp_range['start_commit']['date'].replace('Z', '+00:00')))
                all_dates.append(datetime.fromisoformat(comp_range['end_commit']['date'].replace('Z', '+00:00')))
            
            earliest = min(all_dates)
            latest = max(all_dates)
            
            # Update the processing range
            merged_range = proc_range.copy()
            merged_range['date_range'] = f"{earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}"
            
            print(f"  ✅ Merged range spans: {merged_range['date_range']}")
            print(f"  📊 Contains {merged_range['commits_count']} commits")
            
            merged_ranges.append(merged_range)
        else:
            # No overlap, keep as is
            merged_ranges.append(proc_range)
    
    # Add back completed ranges that don't overlap
    for i, comp_range in enumerate(completed_ranges):
        if i not in ranges_to_remove:
            merged_ranges.append(comp_range)
    
    # Sort by start date
    merged_ranges.sort(key=lambda r: datetime.fromisoformat(r['start_commit']['date'].replace('Z', '+00:00')))
    
    # Reassign range IDs
    for i, r in enumerate(merged_ranges, 1):
        r['range_id'] = i
    
    return merged_ranges


def main():
    status_file = Path('data/status.json')
    backup_file = Path('data/status_before_merge.json')
    
    # Load current status
    with open(status_file) as f:
        status = json.load(f)
    
    # Backup
    with open(backup_file, 'w') as f:
        json.dump(status, f, indent=2)
    print(f"✅ Backup saved to {backup_file}")
    
    original_ranges = status['processed_ranges']
    print(f"\n📊 Original: {len(original_ranges)} ranges")
    
    # Merge overlapping ranges
    merged_ranges = merge_overlapping_ranges(original_ranges)
    print(f"📊 After merge: {len(merged_ranges)} ranges")
    
    if len(merged_ranges) == len(original_ranges):
        print("\n✅ No changes needed - ranges don't overlap or already merged")
        return
    
    # Update status
    status['processed_ranges'] = merged_ranges
    
    # Save
    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2, default=str)
    
    print(f"\n✅ Updated status file saved")
    print(f"\nFinal ranges:")
    for r in merged_ranges:
        status_icon = "🔄" if r['status'] == 'processing' else "✅"
        print(f"  {status_icon} Range {r['range_id']}: {r['date_range']} ({r['commits_count']} commits)")


if __name__ == '__main__':
    main()
