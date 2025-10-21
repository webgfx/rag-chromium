#!/usr/bin/env python3
"""
Autonomous monitoring and error handling for Chromium ingestion.
Monitors progress, detects errors, and restarts with fixes as needed.
"""

import time
import subprocess
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent))

from rag_system.core.logger import setup_logger
from rag_system.vector.qdrant_database import VectorDatabase


def monitor_ingestion():
    """Monitor ingestion progress and handle errors automatically."""
    logger = setup_logger("Autonomous_Monitor")
    
    logger.info("=" * 80)
    logger.info("Starting autonomous ingestion monitoring")
    logger.info("User has gone to sleep - will handle all issues automatically")
    logger.info("=" * 80)
    
    check_interval = 60  # Check every minute
    last_document_count = 0
    no_progress_count = 0
    max_no_progress = 10  # Restart if no progress for 10 minutes
    
    try:
        # Initialize Qdrant to check progress
        db = VectorDatabase(
            collection_name="chromium_complete",
            persist_directory="data/cache/qdrant_db",
            vector_size=1024
        )
        
        while True:
            time.sleep(check_interval)
            
            # Get current stats
            try:
                stats = db.get_collection_stats()
                current_count = stats.get('total_documents', 0)
                
                logger.info(f"Current status: {current_count:,} documents")
                
                # Check for progress
                if current_count > last_document_count:
                    documents_added = current_count - last_document_count
                    rate = documents_added / (check_interval / 60)  # docs per minute
                    logger.info(f"Progress: +{documents_added:,} documents ({rate:.1f}/min)")
                    last_document_count = current_count
                    no_progress_count = 0
                else:
                    no_progress_count += 1
                    logger.warning(f"No progress detected ({no_progress_count}/{max_no_progress})")
                    
                    if no_progress_count >= max_no_progress:
                        logger.error("No progress for 10 minutes - process may be stuck")
                        logger.info("Will check process status...")
                        
                        # Check if process is running
                        result = subprocess.run(
                            ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue"],
                            capture_output=True,
                            text=True
                        )
                        
                        if "python" not in result.stdout.lower():
                            logger.error("Ingestion process not running!")
                            logger.info("Restarting ingestion...")
                            restart_ingestion()
                            no_progress_count = 0
                
                # Check progress.json for errors
                progress_file = Path("data/massive_cache/progress.json")
                if progress_file.exists():
                    with open(progress_file) as f:
                        progress = json.load(f)
                    
                    if progress.get('error_count', 0) > 100:
                        logger.error(f"Too many errors: {progress['error_count']}")
                        logger.info("Restarting with adjusted parameters...")
                        restart_ingestion(reduce_batch=True)
                        no_progress_count = 0
                
            except Exception as e:
                logger.error(f"Error checking stats: {e}")
                no_progress_count += 1
            
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitor crashed: {e}")
        import traceback
        traceback.print_exc()


def restart_ingestion(reduce_batch=False):
    """Restart ingestion process with optional parameter adjustments."""
    logger = setup_logger("Ingestion_Restarter")
    
    logger.info("Stopping current ingestion process...")
    subprocess.run(["powershell", "-Command", "Stop-Process -Name python -Force"], check=False)
    time.sleep(5)
    
    batch_size = 250 if reduce_batch else 500
    logger.info(f"Restarting with batch_size={batch_size}...")
    
    cmd = [
        "python", "massive_chromium_ingestion.py",
        "--repo-path", "d:\\r\\cr\\src",
        "--start-date", "2020-01-01",
        "--batch-size", str(batch_size),
        "--embedding-batch-size", "128",
        "--max-workers", "8"
    ]
    
    subprocess.Popen(cmd, shell=False)
    logger.info("Ingestion restarted successfully")


if __name__ == "__main__":
    print("=" * 80)
    print("AUTONOMOUS MONITORING MODE")
    print("Will monitor ingestion and handle all issues automatically")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 80)
    print()
    
    monitor_ingestion()
