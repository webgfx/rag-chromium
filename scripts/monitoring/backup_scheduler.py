#!/usr/bin/env python3
"""
Automated backup scheduler for Qdrant database.
Runs periodic backups and monitors ingestion for automatic backups.
"""

import time
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import argparse
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backup_database import QdrantBackupManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackupScheduler:
    """Automated backup scheduler with multiple strategies."""
    
    def __init__(self, db_path: Path, backup_dir: Path, 
                 interval_hours: int = 6, keep_backups: int = 10):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.interval_hours = interval_hours
        self.keep_backups = keep_backups
        self.manager = QdrantBackupManager(db_path, backup_dir)
        self.last_backup_time = None
        self.last_document_count = 0
        
    def should_backup_time_based(self) -> bool:
        """Check if it's time for a scheduled backup."""
        if self.last_backup_time is None:
            return True
        
        elapsed = datetime.now() - self.last_backup_time
        return elapsed >= timedelta(hours=self.interval_hours)
    
    def should_backup_milestone(self, current_count: int, milestone: int = 10000) -> bool:
        """Check if we've reached a document count milestone."""
        if self.last_document_count == 0:
            self.last_document_count = current_count
            return False
        
        # Check if we crossed a milestone boundary
        last_milestone = (self.last_document_count // milestone) * milestone
        current_milestone = (current_count // milestone) * milestone
        
        return current_milestone > last_milestone
    
    def get_current_document_count(self) -> int:
        """Get current document count from status file or database."""
        try:
            # Try status file first (faster)
            status_file = Path('data/status.json')
            if status_file.exists():
                with open(status_file) as f:
                    status = json.load(f)
                    return status.get('database', {}).get('total_documents', 0)
            
            # Fall back to database query
            self.manager.connect()
            stats = self.manager.get_collection_stats('chromium_complete')
            return stats.get('points_count', 0)
            
        except Exception as e:
            logger.error(f"Failed to get document count: {e}")
            return 0
    
    def perform_backup(self, reason: str = "scheduled") -> bool:
        """Perform a backup with specified reason."""
        try:
            logger.info(f"Starting backup (reason: {reason})...")
            
            # Use compressed backup for scheduled/milestone backups (saves space)
            # Use full backup for critical milestones
            backup_type = "compressed" if reason != "critical" else "all"
            
            backup_info = self.manager.create_comprehensive_backup(
                collection_name='chromium_complete',
                backup_type=backup_type
            )
            
            if 'error' not in backup_info:
                self.last_backup_time = datetime.now()
                self.last_document_count = backup_info.get('collection_stats', {}).get('points_count', 0)
                
                # Clean up old backups
                self.manager.cleanup_old_backups(keep_last_n=self.keep_backups)
                
                logger.info(f"✅ Backup completed successfully ({reason})")
                return True
            else:
                logger.error(f"❌ Backup failed: {backup_info.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}", exc_info=True)
            return False
    
    def monitor_and_backup(self, check_interval_minutes: int = 30):
        """
        Monitor ingestion and perform backups based on:
        1. Time interval (every N hours)
        2. Document milestones (every 10K documents)
        3. Manual trigger (check for flag file)
        """
        logger.info("Starting backup scheduler...")
        logger.info(f"  Time-based interval: every {self.interval_hours} hours")
        logger.info(f"  Milestone-based: every 10,000 documents")
        logger.info(f"  Check interval: every {check_interval_minutes} minutes")
        logger.info(f"  Keeping last {self.keep_backups} backups")
        
        trigger_file = Path('data/massive_cache/trigger_backup.flag')
        
        while True:
            try:
                # Check for manual trigger
                if trigger_file.exists():
                    logger.info("Manual backup trigger detected")
                    self.perform_backup(reason="manual")
                    trigger_file.unlink()
                
                # Get current state
                current_count = self.get_current_document_count()
                
                # Check time-based backup
                if self.should_backup_time_based():
                    self.perform_backup(reason=f"scheduled ({self.interval_hours}h)")
                
                # Check milestone-based backup
                elif self.should_backup_milestone(current_count, milestone=10000):
                    milestone_reached = (current_count // 10000) * 10000
                    self.perform_backup(reason=f"milestone ({milestone_reached} docs)")
                
                # Status update
                if self.last_backup_time:
                    elapsed = datetime.now() - self.last_backup_time
                    next_backup = timedelta(hours=self.interval_hours) - elapsed
                    logger.info(f"Documents: {current_count:,} | Last backup: {elapsed.total_seconds()/3600:.1f}h ago | Next: {next_backup.total_seconds()/3600:.1f}h")
                else:
                    logger.info(f"Documents: {current_count:,} | No backups yet")
                
                # Sleep until next check
                time.sleep(check_interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Backup scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}", exc_info=True)
                time.sleep(60)  # Wait a bit before retrying


def main():
    parser = argparse.ArgumentParser(description="Automated backup scheduler for Qdrant")
    parser.add_argument('--db-path', default='data/cache/qdrant_db', help='Path to Qdrant database')
    parser.add_argument('--backup-dir', default='data/backups', help='Backup destination directory')
    parser.add_argument('--interval', type=int, default=6, help='Backup interval in hours (default: 6)')
    parser.add_argument('--keep', type=int, default=10, help='Number of backups to keep (default: 10)')
    parser.add_argument('--check-interval', type=int, default=30, help='Check interval in minutes (default: 30)')
    parser.add_argument('--run-once', action='store_true', help='Run one backup and exit')
    
    args = parser.parse_args()
    
    scheduler = BackupScheduler(
        db_path=Path(args.db_path),
        backup_dir=Path(args.backup_dir),
        interval_hours=args.interval,
        keep_backups=args.keep
    )
    
    if args.run_once:
        # Run single backup
        success = scheduler.perform_backup(reason="manual")
        sys.exit(0 if success else 1)
    else:
        # Run continuous monitoring
        scheduler.monitor_and_backup(check_interval_minutes=args.check_interval)


if __name__ == '__main__':
    main()
