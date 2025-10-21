#!/usr/bin/env python3
"""
Restore Qdrant database from backup.
Supports multiple backup formats with verification.
"""

import json
import logging
import shutil
import gzip
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import argparse

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QdrantRestoreManager:
    """Manages restoration of Qdrant database from backups."""
    
    def __init__(self, db_path: Path, backup_dir: Path):
        self.db_path = Path(db_path)
        self.backup_dir = Path(backup_dir)
        self.metadata_file = self.backup_dir / "backup_metadata.json"
        
    def list_available_backups(self) -> list:
        """List all available backups with details."""
        if not self.metadata_file.exists():
            logger.warning("No backup metadata found")
            return []
        
        with open(self.metadata_file) as f:
            backups = json.load(f)
        
        return backups
    
    def restore_from_directory(self, backup_path: Path, force: bool = False):
        """Restore from a full directory backup."""
        logger.info(f"Restoring from directory: {backup_path}")
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Check if current database exists
        if self.db_path.exists() and not force:
            raise RuntimeError(
                f"Database already exists at {self.db_path}. "
                f"Use --force to overwrite, or manually backup/remove it first."
            )
        
        # Remove current database if force is enabled
        if self.db_path.exists():
            logger.warning(f"Removing existing database: {self.db_path}")
            shutil.rmtree(self.db_path)
        
        # Copy backup to database location
        logger.info(f"Copying backup to {self.db_path}")
        shutil.copytree(backup_path, self.db_path, symlinks=False)
        
        logger.info("✅ Restoration from directory completed")
        
    def restore_from_compressed(self, archive_path: Path, force: bool = False):
        """Restore from compressed tar.gz backup."""
        logger.info(f"Restoring from compressed archive: {archive_path}")
        
        if not archive_path.exists():
            raise FileNotFoundError(f"Backup archive not found: {archive_path}")
        
        # Check if current database exists
        if self.db_path.exists() and not force:
            raise RuntimeError(
                f"Database already exists at {self.db_path}. "
                f"Use --force to overwrite, or manually backup/remove it first."
            )
        
        # Remove current database if force is enabled
        if self.db_path.exists():
            logger.warning(f"Removing existing database: {self.db_path}")
            shutil.rmtree(self.db_path)
        
        # Extract archive
        logger.info(f"Extracting archive to {self.db_path.parent}")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.db_path.parent)
        
        logger.info("✅ Restoration from compressed archive completed")
    
    def restore_from_snapshot(self, snapshot_name: str, collection_name: str = "chromium_complete"):
        """Restore from Qdrant snapshot."""
        logger.info(f"Restoring from snapshot: {snapshot_name}")
        
        # Connect to Qdrant
        client = QdrantClient(path=str(self.db_path))
        
        # Restore snapshot
        client.recover_snapshot(
            collection_name=collection_name,
            snapshot_name=snapshot_name
        )
        
        logger.info("✅ Restoration from snapshot completed")
    
    def restore_from_export(self, export_path: Path, collection_name: str = "chromium_complete",
                           vector_size: int = 1024, force: bool = False):
        """Restore from JSON Lines export."""
        logger.info(f"Restoring from export: {export_path}")
        
        if not export_path.exists():
            raise FileNotFoundError(f"Export file not found: {export_path}")
        
        # Connect to Qdrant
        client = QdrantClient(path=str(self.db_path))
        
        # Check if collection exists
        try:
            existing = client.get_collection(collection_name)
            if not force:
                raise RuntimeError(
                    f"Collection '{collection_name}' already exists. "
                    f"Use --force to overwrite."
                )
            logger.warning(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        except Exception:
            pass  # Collection doesn't exist, which is fine
        
        # Create collection
        logger.info(f"Creating collection: {collection_name}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        
        # Import data
        logger.info("Importing data from export...")
        batch = []
        batch_size = 100
        points_imported = 0
        
        with gzip.open(export_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                
                # Create point
                point = PointStruct(
                    id=data['id'],
                    vector=data['vector'],
                    payload=data['payload']
                )
                batch.append(point)
                
                # Upload batch when full
                if len(batch) >= batch_size:
                    client.upsert(
                        collection_name=collection_name,
                        points=batch
                    )
                    points_imported += len(batch)
                    batch = []
                    
                    if points_imported % 5000 == 0:
                        logger.info(f"  Imported {points_imported} points...")
            
            # Upload remaining batch
            if batch:
                client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                points_imported += len(batch)
        
        logger.info(f"✅ Restoration from export completed: {points_imported} points imported")
    
    def verify_restoration(self, collection_name: str = "chromium_complete",
                          expected_count: Optional[int] = None) -> bool:
        """Verify restored database."""
        logger.info("Verifying restored database...")
        
        try:
            client = QdrantClient(path=str(self.db_path))
            info = client.get_collection(collection_name)
            
            logger.info(f"Collection: {collection_name}")
            logger.info(f"  Points: {info.points_count}")
            logger.info(f"  Vectors: {info.vectors_count}")
            logger.info(f"  Indexed: {info.indexed_vectors_count}")
            logger.info(f"  Status: {info.status}")
            
            if expected_count is not None:
                if info.points_count != expected_count:
                    logger.warning(
                        f"Point count mismatch! Expected {expected_count}, got {info.points_count}"
                    )
                    return False
            
            logger.info("✓ Database verification passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Database verification failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Restore Qdrant database from backup")
    parser.add_argument('--db-path', default='data/cache/qdrant_db', help='Path to Qdrant database')
    parser.add_argument('--backup-dir', default='data/backups', help='Backup directory')
    parser.add_argument('--list', action='store_true', help='List available backups')
    parser.add_argument('--backup-id', type=int, help='Backup ID to restore (from --list)')
    parser.add_argument('--backup-path', help='Direct path to backup (directory or archive)')
    parser.add_argument('--type', choices=['directory', 'compressed', 'snapshot', 'export'],
                       help='Type of backup to restore')
    parser.add_argument('--collection', default='chromium_complete', help='Collection name')
    parser.add_argument('--force', action='store_true', help='Overwrite existing database')
    parser.add_argument('--verify', action='store_true', help='Verify after restoration')
    
    args = parser.parse_args()
    
    manager = QdrantRestoreManager(
        db_path=Path(args.db_path),
        backup_dir=Path(args.backup_dir)
    )
    
    # List backups
    if args.list:
        backups = manager.list_available_backups()
        
        print(f"\n{'='*70}")
        print(f"AVAILABLE BACKUPS ({len(backups)} total)")
        print(f"{'='*70}")
        
        for i, backup in enumerate(backups, 1):
            print(f"\nID: {i}")
            print(f"Timestamp: {backup['timestamp']}")
            print(f"Collection: {backup['collection_name']}")
            print(f"Type: {backup['backup_type']}")
            print(f"Documents: {backup.get('collection_stats', {}).get('points_count', 'N/A')}")
            print(f"Available formats:")
            
            for name, info in backup.get('backups', {}).items():
                backup_type = info['type']
                path = info['path']
                size = info.get('size_mb', 'N/A')
                size_str = f"{size:.2f} MB" if isinstance(size, (int, float)) else size
                
                # Check if backup file/directory still exists
                exists = Path(path).exists()
                status = "✓" if exists else "✗ (missing)"
                
                print(f"  [{i}.{name}] {backup_type}: {size_str} {status}")
                print(f"     Path: {path}")
        
        print(f"\n{'='*70}")
        print("To restore, use: python restore_database.py --backup-id <ID> --type <TYPE>")
        print(f"{'='*70}\n")
        return
    
    # Restore from backup
    if args.backup_id or args.backup_path:
        try:
            backup_path = None
            expected_count = None
            
            # Get backup path from ID
            if args.backup_id:
                backups = manager.list_available_backups()
                if args.backup_id < 1 or args.backup_id > len(backups):
                    print(f"Error: Invalid backup ID {args.backup_id}")
                    return
                
                backup = backups[args.backup_id - 1]
                expected_count = backup.get('collection_stats', {}).get('points_count')
                
                # Get specific backup type
                if args.type:
                    backup_info = None
                    for name, info in backup.get('backups', {}).items():
                        if args.type in name or info['type'] == args.type:
                            backup_info = info
                            break
                    
                    if not backup_info:
                        print(f"Error: Backup type '{args.type}' not found in backup {args.backup_id}")
                        return
                    
                    backup_path = Path(backup_info['path'])
                else:
                    print("Error: --type is required when using --backup-id")
                    return
            
            # Use direct path
            elif args.backup_path:
                backup_path = Path(args.backup_path)
                if not args.type:
                    print("Error: --type is required when using --backup-path")
                    return
            
            # Perform restoration
            print(f"\n{'='*70}")
            print("STARTING RESTORATION")
            print(f"{'='*70}\n")
            print(f"Backup: {backup_path}")
            print(f"Type: {args.type}")
            print(f"Destination: {args.db_path}")
            print(f"Force: {args.force}")
            print()
            
            if not args.force:
                response = input("This will restore the database. Continue? [y/N]: ")
                if response.lower() != 'y':
                    print("Restoration cancelled")
                    return
            
            # Restore based on type
            if args.type == 'directory':
                manager.restore_from_directory(backup_path, force=args.force)
            elif args.type == 'compressed' or args.type == 'tar_gz':
                manager.restore_from_compressed(backup_path, force=args.force)
            elif args.type == 'snapshot' or args.type == 'qdrant_snapshot':
                snapshot_name = backup_path.name if backup_path.is_file() else str(backup_path)
                manager.restore_from_snapshot(snapshot_name, args.collection)
            elif args.type == 'export' or args.type == 'jsonl_gz':
                manager.restore_from_export(backup_path, args.collection, force=args.force)
            else:
                print(f"Error: Unknown backup type: {args.type}")
                return
            
            # Verify if requested
            if args.verify:
                print(f"\n{'='*70}")
                print("VERIFYING RESTORATION")
                print(f"{'='*70}\n")
                
                verified = manager.verify_restoration(args.collection, expected_count)
                if not verified:
                    print("\n⚠️  Verification found issues - please check the logs")
                    return
            
            print(f"\n{'='*70}")
            print("✅ RESTORATION COMPLETED SUCCESSFULLY")
            print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"\n❌ Restoration failed: {e}")
            logger.error("Restoration failed", exc_info=True)
    
    else:
        print("Error: Either --list, --backup-id, or --backup-path is required")
        print("Use --help for usage information")


if __name__ == '__main__':
    main()
