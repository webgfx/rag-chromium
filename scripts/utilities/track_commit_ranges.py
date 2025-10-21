#!/usr/bin/env python3
"""
Track and manage processed commit ranges for incremental ingestion.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


class CommitRangeTracker:
    """Track which commit ranges have been processed."""
    
    def __init__(self, tracking_file: str = "data/massive_cache/commit_ranges.json"):
        self.tracking_file = Path(tracking_file)
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        self.ranges = self.load_ranges()
    
    def load_ranges(self) -> List[Dict]:
        """Load previously processed commit ranges."""
        if self.tracking_file.exists():
            with open(self.tracking_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_ranges(self):
        """Save commit ranges to file."""
        with open(self.tracking_file, 'w') as f:
            json.dump(self.ranges, f, indent=2, default=str)
    
    def add_range(self, 
                  start_date: str,
                  end_date: str,
                  first_commit_sha: str,
                  last_commit_sha: str,
                  first_commit_date: str,
                  last_commit_date: str,
                  commits_processed: int,
                  documents_created: int,
                  phase_name: str = None):
        """Record a processed commit range."""
        range_entry = {
            "phase_name": phase_name,
            "query_start_date": start_date,
            "query_end_date": end_date,
            "actual_first_commit_sha": first_commit_sha,
            "actual_last_commit_sha": last_commit_sha,
            "actual_first_commit_date": first_commit_date,
            "actual_last_commit_date": last_commit_date,
            "commits_processed": commits_processed,
            "documents_created": documents_created,
            "processed_at": datetime.now().isoformat(),
        }
        
        self.ranges.append(range_entry)
        self.save_ranges()
        return range_entry
    
    def get_latest_commit_date(self) -> Optional[str]:
        """Get the most recent commit date that has been processed."""
        if not self.ranges:
            return None
        
        # Find the most recent actual_last_commit_date
        latest = max(self.ranges, key=lambda r: r.get('actual_last_commit_date', ''))
        return latest.get('actual_last_commit_date')
    
    def get_oldest_commit_date(self) -> Optional[str]:
        """Get the oldest commit date that has been processed."""
        if not self.ranges:
            return None
        
        # Find the oldest actual_first_commit_date
        oldest = min(self.ranges, key=lambda r: r.get('actual_first_commit_date', 'Z'))
        return oldest.get('actual_first_commit_date')
    
    def find_gaps(self) -> List[Tuple[str, str]]:
        """Find date gaps in processed ranges."""
        if len(self.ranges) < 2:
            return []
        
        # Sort ranges by first commit date
        sorted_ranges = sorted(self.ranges, key=lambda r: r.get('actual_first_commit_date', ''))
        
        gaps = []
        for i in range(len(sorted_ranges) - 1):
            current_last = sorted_ranges[i]['actual_last_commit_date']
            next_first = sorted_ranges[i + 1]['actual_first_commit_date']
            
            # If there's a gap (dates don't overlap)
            if current_last < next_first:
                gaps.append((current_last, next_first))
        
        return gaps
    
    def get_summary(self) -> Dict:
        """Get summary of all processed ranges."""
        if not self.ranges:
            return {
                "total_ranges": 0,
                "total_commits": 0,
                "total_documents": 0,
                "oldest_commit": None,
                "newest_commit": None,
                "gaps": []
            }
        
        return {
            "total_ranges": len(self.ranges),
            "total_commits": sum(r.get('commits_processed', 0) for r in self.ranges),
            "total_documents": sum(r.get('documents_created', 0) for r in self.ranges),
            "oldest_commit": self.get_oldest_commit_date(),
            "newest_commit": self.get_latest_commit_date(),
            "gaps": self.find_gaps(),
            "ranges": self.ranges
        }
    
    def suggest_next_range(self, target_direction: str = "recent") -> Optional[Tuple[str, str]]:
        """Suggest the next date range to process.
        
        Args:
            target_direction: "recent" to process newer commits, "historic" for older
        """
        latest = self.get_latest_commit_date()
        oldest = self.get_oldest_commit_date()
        
        if target_direction == "recent":
            if latest:
                # Suggest processing from latest forward to now
                return (latest, datetime.now().isoformat())
            else:
                # No data yet, suggest recent 6 months
                return ("2024-04-01", datetime.now().isoformat())
        else:
            if oldest:
                # Suggest processing older data
                # Go back 2 years from oldest
                return ("2020-01-01", oldest)
            else:
                return None


def main():
    """CLI for commit range tracking."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Track processed commit ranges")
    parser.add_argument("--summary", action="store_true", help="Show summary of processed ranges")
    parser.add_argument("--gaps", action="store_true", help="Show gaps in processed ranges")
    parser.add_argument("--suggest", choices=["recent", "historic"], help="Suggest next range to process")
    
    args = parser.parse_args()
    
    tracker = CommitRangeTracker()
    
    if args.summary or (not args.gaps and not args.suggest):
        summary = tracker.get_summary()
        print("\n=== Commit Range Summary ===")
        print(f"Total ranges processed: {summary['total_ranges']}")
        print(f"Total commits: {summary['total_commits']:,}")
        print(f"Total documents: {summary['total_documents']:,}")
        print(f"Oldest commit: {summary['oldest_commit']}")
        print(f"Newest commit: {summary['newest_commit']}")
        
        if summary['ranges']:
            print("\n=== Processed Ranges ===")
            for i, r in enumerate(summary['ranges'], 1):
                print(f"\n{i}. {r.get('phase_name', 'Unknown Phase')}")
                print(f"   Query: {r['query_start_date']} to {r['query_end_date']}")
                print(f"   Actual: {r['actual_first_commit_date']} to {r['actual_last_commit_date']}")
                print(f"   Commits: {r['commits_processed']:,}, Documents: {r['documents_created']:,}")
                print(f"   Processed: {r['processed_at']}")
    
    if args.gaps:
        gaps = tracker.find_gaps()
        if gaps:
            print("\n=== Date Gaps Found ===")
            for i, (start, end) in enumerate(gaps, 1):
                print(f"{i}. Gap from {start} to {end}")
        else:
            print("\nNo gaps found in processed ranges.")
    
    if args.suggest:
        suggestion = tracker.suggest_next_range(args.suggest)
        if suggestion:
            start, end = suggestion
            print(f"\n=== Suggested Next Range ({args.suggest}) ===")
            print(f"Start date: {start}")
            print(f"End date: {end}")
            print(f"\nCommand:")
            print(f'python massive_chromium_ingestion.py --repo-path "d:\\r\\cr\\src" --start-date "{start}" --end-date "{end}" --max-commits 100000')
        else:
            print("\nNo suggestion available.")


if __name__ == "__main__":
    main()
