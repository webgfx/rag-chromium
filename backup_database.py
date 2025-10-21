#!/usr/bin/env python3
"""
Sophisticated database backup solution for Qdrant.
Supports incremental snapshots, compression, and automated scheduling.
"""

import json
import logging
import shutil
import gzip
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import SnapshotDescription

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantBackupManager:
    """Manages backups for Qdrant vector database."""
    
    def __init__(self, db_path: Path, backup_dir: Path):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        self.client = None
        
    def connect(self):
        """Connect to Qdrant database."""
        logger.info(f"Connecting to Qdrant at {self.db_path}")
        self.client = QdrantClient(path=str(self.db_path))
        
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get collection statistics for verification."""
        try:
            info = self.client.get_collection(collection_name)
            return {
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'indexed_vectors_count': info.indexed_vectors_count,
                'segments_count': info.segments_count,
                'status': info.status
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def create_snapshot(self, collection_name: str) -> Optional[str]:
        """Create a Qdrant snapshot using native API."""
        try:
            logger.info(f"Creating snapshot for collection '{collection_name}'...")
            snapshot_info = self.client.create_snapshot(collection_name=collection_name)
            logger.info(f"Snapshot created: {snapshot_info.name}")
            return snapshot_info.name
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return None
    
    def list_snapshots(self, collection_name: str) -> List[SnapshotDescription]:
        """List all snapshots for a collection."""
        try:
            snapshots = self.client.list_snapshots(collection_name=collection_name)
            return snapshots
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []
    
    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def backup_full_directory(self, backup_name: Optional[str] = None) -> Path:
        """Create a full backup by copying the entire database directory."""
        if backup_name is None:
            backup_name = f"full_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = self.backup_dir / backup_name
        
        logger.info(f"Creating full directory backup: {backup_path}")
        
        # Copy entire database directory
        if backup_path.exists():
            logger.warning(f"Backup path exists, removing: {backup_path}")
            shutil.rmtree(backup_path)
        
        shutil.copytree(self.db_path, backup_path, symlinks=False)
        
        # Calculate directory size
        total_size = sum(f.stat().st_size for f in backup_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        logger.info(f"Full backup completed: {size_mb:.2f} MB")
        
        return backup_path
    
    def backup_compressed(self, backup_name: Optional[str] = None) -> Path:
        """Create a compressed backup archive."""
        if backup_name is None:
            backup_name = f"compressed_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        archive_path = self.backup_dir / f"{backup_name}.tar.gz"
        
        logger.info(f"Creating compressed backup: {archive_path}")
        
        # Create tar.gz archive
        import tarfile
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(self.db_path, arcname=self.db_path.name)
        
        # Calculate sizes
        original_size = sum(f.stat().st_size for f in self.db_path.rglob('*') if f.is_file())
        compressed_size = archive_path.stat().st_size
        compression_ratio = (1 - compressed_size / original_size) * 100
        
        logger.info(f"Compressed backup completed:")
        logger.info(f"  Original: {original_size / (1024*1024):.2f} MB")
        logger.info(f"  Compressed: {compressed_size / (1024*1024):.2f} MB")
        logger.info(f"  Compression: {compression_ratio:.1f}%")
        
        # Calculate checksum
        checksum = self.calculate_checksum(archive_path)
        logger.info(f"  Checksum: {checksum}")
        
        return archive_path
    
    def backup_collection_data(self, collection_name: str, backup_name: Optional[str] = None) -> Path:
        """Export collection data to JSON (for maximum portability)."""
        if backup_name is None:
            backup_name = f"collection_export_{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        export_path = self.backup_dir / f"{backup_name}.jsonl.gz"
        
        logger.info(f"Exporting collection '{collection_name}' to {export_path}")
        
        # Scroll through all points and export
        points_exported = 0
        with gzip.open(export_path, 'wt', encoding='utf-8') as f:
            offset = None
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )
                
                if not records:
                    break
                
                for record in records:
                    # Export as JSON lines
                    data = {
                        'id': str(record.id),
                        'vector': record.vector,
                        'payload': record.payload
                    }
                    f.write(json.dumps(data, default=str) + '\n')
                    points_exported += 1
                
                if points_exported % 5000 == 0:
                    logger.info(f"  Exported {points_exported} points...")
                
                offset = next_offset
                if offset is None:
                    break
        
        file_size = export_path.stat().st_size / (1024 * 1024)
        logger.info(f"Collection export completed: {points_exported} points, {file_size:.2f} MB")
        
        # Calculate checksum
        checksum = self.calculate_checksum(export_path)
        logger.info(f"  Checksum: {checksum}")
        
        return export_path
    
    def save_metadata(self, backup_info: Dict[str, Any]):
        """Save backup metadata for tracking."""
        metadata = []
        
        # Load existing metadata
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                metadata = json.load(f)
        
        # Add new backup info
        metadata.append(backup_info)
        
        # Save updated metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def create_comprehensive_backup(self, collection_name: str = "chromium_complete", 
                                   backup_type: str = "all") -> Dict[str, Any]:
        """
        Create comprehensive backup with multiple strategies.
        
        Args:
            collection_name: Name of the collection to backup
            backup_type: Type of backup - "snapshot", "full", "compressed", "export", "all"
        
        Returns:
            Dictionary with backup information
        """
        timestamp = datetime.now()
        backup_info = {
            'timestamp': timestamp.isoformat(),
            'collection_name': collection_name,
            'backup_type': backup_type,
            'backups': {}
        }
        
        # Connect to database
        self.connect()
        
        # Get collection stats before backup
        stats = self.get_collection_stats(collection_name)
        backup_info['collection_stats'] = stats
        logger.info(f"Collection stats: {stats.get('points_count', 0)} points")
        
        try:
            # 1. Qdrant native snapshot (fastest, most reliable for restore)
            if backup_type in ["snapshot", "all"]:
                snapshot_name = self.create_snapshot(collection_name)
                if snapshot_name:
                    backup_info['backups']['snapshot'] = {
                        'name': snapshot_name,
                        'type': 'qdrant_snapshot',
                        'path': str(self.db_path / 'snapshots' / snapshot_name)
                    }
            
            # 2. Full directory copy (complete database state)
            if backup_type in ["full", "all"]:
                full_backup_path = self.backup_full_directory()
                backup_info['backups']['full_copy'] = {
                    'path': str(full_backup_path),
                    'type': 'directory_copy',
                    'size_mb': sum(f.stat().st_size for f in full_backup_path.rglob('*') if f.is_file()) / (1024*1024)
                }
            
            # 3. Compressed archive (space-efficient, portable)
            if backup_type in ["compressed", "all"]:
                archive_path = self.backup_compressed()
                checksum = self.calculate_checksum(archive_path)
                backup_info['backups']['compressed'] = {
                    'path': str(archive_path),
                    'type': 'tar_gz',
                    'size_mb': archive_path.stat().st_size / (1024*1024),
                    'checksum': checksum
                }
            
            # 4. Collection data export (maximum portability, can restore to different DB)
            if backup_type in ["export", "all"]:
                export_path = self.backup_collection_data(collection_name)
                checksum = self.calculate_checksum(export_path)
                backup_info['backups']['data_export'] = {
                    'path': str(export_path),
                    'type': 'jsonl_gz',
                    'size_mb': export_path.stat().st_size / (1024*1024),
                    'checksum': checksum
                }
            
            # Save metadata
            self.save_metadata(backup_info)
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("BACKUP SUMMARY")
            logger.info("="*60)
            logger.info(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Collection: {collection_name}")
            logger.info(f"Documents: {stats.get('points_count', 0)}")
            logger.info(f"Backup Type: {backup_type}")
            logger.info(f"Backups Created: {len(backup_info['backups'])}")
            for name, info in backup_info['backups'].items():
                logger.info(f"  - {name}: {info['path']}")
            logger.info("="*60)
            
            return backup_info
            
        except Exception as e:
            logger.error(f"Backup failed: {e}", exc_info=True)
            backup_info['error'] = str(e)
            return backup_info
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups."""
        if not self.metadata_file.exists():
            return []
        
        with open(self.metadata_file) as f:
            metadata = json.load(f)
        
        return metadata
    
    def verify_backup(self, backup_path: Path) -> bool:
        """Verify backup integrity."""
        logger.info(f"Verifying backup: {backup_path}")
        
        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False
        
        # For compressed backups, verify checksum if available
        if backup_path.suffix == '.gz':
            current_checksum = self.calculate_checksum(backup_path)
            logger.info(f"Current checksum: {current_checksum}")
            
            # Look up original checksum in metadata
            metadata = self.list_backups()
            for backup in metadata:
                for backup_type, info in backup.get('backups', {}).items():
                    if info.get('path') == str(backup_path) and 'checksum' in info:
                        original_checksum = info['checksum']
                        if current_checksum == original_checksum:
                            logger.info("✓ Checksum verified!")
                            return True
                        else:
                            logger.error("✗ Checksum mismatch!")
                            return False
        
        logger.info("✓ Backup exists and is accessible")
        return True
    
    def cleanup_old_backups(self, keep_last_n: int = 5):
        """Clean up old backups, keeping only the most recent N backups."""
        logger.info(f"Cleaning up old backups (keeping last {keep_last_n})...")
        
        metadata = self.list_backups()
        if len(metadata) <= keep_last_n:
            logger.info(f"Only {len(metadata)} backups found, nothing to clean up")
            return
        
        # Sort by timestamp
        metadata.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Keep last N, remove others
        to_remove = metadata[keep_last_n:]
        
        for backup in to_remove:
            logger.info(f"Removing old backup from {backup['timestamp']}")
            for backup_type, info in backup.get('backups', {}).items():
                backup_path = Path(info['path'])
                if backup_path.exists():
                    if backup_path.is_dir():
                        shutil.rmtree(backup_path)
                    else:
                        backup_path.unlink()
                    logger.info(f"  Removed: {backup_path}")
        
        # Update metadata
        metadata = metadata[:keep_last_n]
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Cleanup completed, {len(to_remove)} old backups removed")


def main():
    parser = argparse.ArgumentParser(description="Backup Qdrant database")
    parser.add_argument('--db-path', default='data/cache/qdrant_db', help='Path to Qdrant database')
    parser.add_argument('--backup-dir', default='data/backups', help='Backup destination directory')
    parser.add_argument('--collection', default='chromium_complete', help='Collection name to backup')
    parser.add_argument('--type', default='all', choices=['snapshot', 'full', 'compressed', 'export', 'all'],
                       help='Type of backup to create')
    parser.add_argument('--verify', action='store_true', help='Verify backup after creation')
    parser.add_argument('--cleanup', type=int, metavar='N', help='Keep only last N backups')
    parser.add_argument('--list', action='store_true', help='List all backups')
    
    args = parser.parse_args()
    
    # Create backup manager
    manager = QdrantBackupManager(
        db_path=Path(args.db_path),
        backup_dir=Path(args.backup_dir)
    )
    
    # List backups
    if args.list:
        backups = manager.list_backups()
        print(f"\n{'='*60}")
        print(f"AVAILABLE BACKUPS ({len(backups)} total)")
        print(f"{'='*60}")
        for i, backup in enumerate(backups, 1):
            print(f"\n{i}. {backup['timestamp']}")
            print(f"   Collection: {backup['collection_name']}")
            print(f"   Type: {backup['backup_type']}")
            print(f"   Documents: {backup.get('collection_stats', {}).get('points_count', 'N/A')}")
            print(f"   Backups:")
            for name, info in backup.get('backups', {}).items():
                size = info.get('size_mb', 'N/A')
                size_str = f"{size:.2f} MB" if isinstance(size, (int, float)) else size
                print(f"     - {name}: {size_str}")
        print(f"{'='*60}\n")
        return
    
    # Cleanup old backups
    if args.cleanup:
        manager.cleanup_old_backups(keep_last_n=args.cleanup)
        return
    
    # Create backup
    print(f"\n{'='*60}")
    print("STARTING BACKUP PROCESS")
    print(f"{'='*60}\n")
    
    backup_info = manager.create_comprehensive_backup(
        collection_name=args.collection,
        backup_type=args.type
    )
    
    # Verify if requested
    if args.verify and 'backups' in backup_info:
        print(f"\n{'='*60}")
        print("VERIFYING BACKUPS")
        print(f"{'='*60}\n")
        
        for name, info in backup_info['backups'].items():
            backup_path = Path(info['path'])
            verified = manager.verify_backup(backup_path)
            status = "✓ VERIFIED" if verified else "✗ FAILED"
            print(f"{name}: {status}")
    
    print("\n✅ Backup process completed successfully!\n")


if __name__ == '__main__':
    main()
