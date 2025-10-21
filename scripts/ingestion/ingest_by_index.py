#!/usr/bin/env python3
"""
Ingest Chromium commits by commit index range.
Skips indices already in processed ranges.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# CRITICAL: Preload heavy modules to prevent KeyboardInterrupt during import
# This fixes the issue where scipy.optimize and qdrant_client imports get interrupted
print("Preloading heavy modules to prevent import interrupts...")

import time

max_retries = 3
for attempt in range(max_retries):
    try:
        import warnings
        warnings.filterwarnings('ignore')
        
        print(f"  Attempt {attempt + 1}/{max_retries}...")
        
        # These imports take several seconds and can be interrupted on Windows
        print("    Loading numpy...", end=" ")
        import numpy as np
        print("✓")
        
        print("    Loading scipy.optimize...", end=" ")
        import scipy.optimize
        print("✓")
        
        print("    Loading sklearn.metrics...", end=" ")
        import sklearn.metrics
        print("✓")
        
        print("    Loading transformers...", end=" ")
        from transformers import AutoModel, AutoTokenizer
        print("✓")
        
        print("    Loading qdrant_client...", end=" ")
        import signal
        # Ignore SIGINT during qdrant import (pydantic schema generation is slow)
        original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            from qdrant_client import QdrantClient
            print("✓")
        finally:
            signal.signal(signal.SIGINT, original_handler)
        
        print("✓ All heavy modules preloaded successfully!")
        break
        
    except KeyboardInterrupt:
        if attempt < max_retries - 1:
            print(f"\n⚠ Module preload interrupted on attempt {attempt + 1}, retrying...")
            time.sleep(1)
        else:
            print("\n✗ Module preload interrupted after all retries - this is the root cause!")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Module preload failed: {e}")
        sys.exit(1)

from rag_system.core.logger import setup_logger
from massive_chromium_ingestion import MassiveIngestionPipeline, IngestionConfig

logger = setup_logger("index_ingestion")


def get_processed_indices():
    """Load processed index ranges from status.json."""
    status_file = Path("data/status.json")
    if not status_file.exists():
        return []
    
    status = json.loads(status_file.read_text())
    ranges = []
    for r in status.get('processed_ranges', []):
        ranges.append((r['start']['index'], r['end']['index']))
    return ranges


def is_index_processed(index: int, processed_ranges):
    """Check if an index is already in processed ranges."""
    for start, end in processed_ranges:
        if start <= index <= end:
            return True
    return False


def get_commits_by_index_range(repo_path: str, start_index: int, end_index: int):
    """Get commits by index range from git repository."""
    count = end_index - start_index + 1
    logger.info(f"Fetching {count:,} commits starting at index {start_index:,}...")
    
    # Check if we have a cached file first
    cache_file = Path(f"data/massive_cache/commits_{start_index}_{end_index}.txt")
    if cache_file.exists():
        logger.info(f"Using cached commit list from {cache_file}")
        target_shas = cache_file.read_text().strip().split('\n')
        logger.info(f"Loaded {len(target_shas):,} commits from cache")
    else:
        logger.info(f"Note: git operations may take 1-2 minutes for large ranges...")
        
        try:
            logger.info(f"Starting git subprocess...")
            # Use CREATE_NEW_PROCESS_GROUP on Windows to prevent Ctrl+C propagation
            import platform
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if platform.system() == 'Windows' else 0
            
            # Temporarily ignore SIGINT during git operation (can take 10+ seconds)
            import signal
            original_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
            
            try:
                result = subprocess.run(
                    ['git', 'rev-list', '--all', '--reverse', f'--skip={start_index}', f'--max-count={count}'],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise on non-zero exit
                    creationflags=creation_flags
                    # No timeout - let it run as long as needed
                )
            finally:
                # Restore original handler
                signal.signal(signal.SIGINT, original_handler)
            
            logger.info(f"Git subprocess completed with exit code {result.returncode}")
            
            if result.returncode != 0:
                logger.error(f"❌ Git command failed with exit code {result.returncode}")
                logger.error(f"stderr: {result.stderr}")
                raise RuntimeError(f"Git command failed with exit code {result.returncode}")
            
            target_shas = result.stdout.strip().split('\n')
            logger.info(f"Fetched {len(target_shas):,} commits from git")
        except subprocess.TimeoutExpired as e:
            logger.error(f"❌ Git operation timed out after 3 minutes")
            logger.error(f"This usually means the git repository is very large or the range is too wide")
            raise RuntimeError(f"Git timeout fetching {count} commits") from e
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Git command failed: {e}")
            logger.error(f"stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
            raise RuntimeError(f"Git command failed") from e
        except Exception as e:
            logger.error(f"❌ Unexpected error during git operation: {e}")
            raise
    
    # Load processed ranges and filter
    processed_ranges = get_processed_indices()
    logger.info(f"Found {len(processed_ranges)} processed ranges")
    
    # Filter out already processed commits
    commits_to_process = []
    skipped = 0
    
    for i, sha in enumerate(target_shas):
        index = start_index + i
        if is_index_processed(index, processed_ranges):
            skipped += 1
        else:
            commits_to_process.append({
                'index': index,
                'sha': sha
            })
    
    logger.info(f"Filtered: {len(commits_to_process):,} new commits, {skipped:,} already processed")
    
    return commits_to_process


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest Chromium commits by index range")
    parser.add_argument("--repo-path", required=True, help="Path to Chromium repository")
    parser.add_argument("--start-index", type=int, required=True, help="Start commit index")
    parser.add_argument("--end-index", type=int, required=True, help="End commit index")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size")
    parser.add_argument("--embedding-batch-size", type=int, default=256, help="Embedding batch size")
    parser.add_argument("--max-workers", type=int, default=16, help="Max workers")
    
    args = parser.parse_args()
    
    logger.info("="*60)
    logger.info("CHROMIUM COMMIT INGESTION BY INDEX")
    logger.info("="*60)
    logger.info(f"Repository: {args.repo_path}")
    logger.info(f"Index range: {args.start_index:,} to {args.end_index:,}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Embedding batch: {args.embedding_batch_size}")
    logger.info(f"Workers: {args.max_workers}")
    logger.info("="*60)
    
    # Get commits to process
    commits = get_commits_by_index_range(
        args.repo_path,
        args.start_index,
        args.end_index
    )
    
    if not commits:
        logger.info("No new commits to process!")
        return
    
    logger.info(f"Processing {len(commits):,} commits...")
    
    # Write commit SHAs to temporary file for the pipeline to use
    commit_file = Path("data/massive_cache/commits_to_process.txt")
    commit_file.parent.mkdir(parents=True, exist_ok=True)
    commit_file.write_text('\n'.join([c['sha'] for c in commits]))
    
    logger.info(f"Wrote {len(commits):,} commit SHAs to {commit_file}")
    
    # Create configuration (no date filtering - we already filtered by index)
    config = IngestionConfig(
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        max_workers=args.max_workers
    )
    
    # Run ingestion with specific commits
    pipeline = MassiveIngestionPipeline(config)
    
    # Store the commit index mapping in progress
    progress_file = Path("data/massive_cache/progress.json")
    if progress_file.exists():
        progress = json.loads(progress_file.read_text())
    else:
        progress = {}
    
    progress['start_index'] = args.start_index
    progress['end_index'] = args.end_index
    progress['commit_index_current'] = args.start_index
    progress['commit_file'] = str(commit_file)
    
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    progress_file.write_text(json.dumps(progress, indent=2))
    
    # Process commits from our filtered list
    pipeline.run_ingestion_from_commit_list(
        repo_path=args.repo_path,
        commit_shas=[c['sha'] for c in commits],
        commit_indices={c['sha']: c['index'] for c in commits}
    )
    
    logger.info("="*60)
    logger.info("INGESTION COMPLETE")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        sys.exit(1)
