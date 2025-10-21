#!/usr/bin/env python3
"""
Massive Chromium commit ingestion pipeline for complete RAG system.
Designed to handle millions of commits with optimizations for scale.
"""

import os
import sys
import json
import time
import hashlib
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import pickle
import sqlite3
import subprocess
from tqdm import tqdm
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Backup trigger flag file
BACKUP_TRIGGER_FLAG = Path("data/massive_cache/trigger_backup.flag")

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.data.chromium import ChromiumDataExtractor
from rag_system.data.preprocessor import DataPreprocessor
from rag_system.data.chunker import TextChunker
from rag_system.embeddings.generator import EmbeddingGenerator
from rag_system.vector import VectorDatabase, VectorDocument
from track_commit_ranges import CommitRangeTracker


@dataclass
class IngestionConfig:
    """Configuration for massive ingestion."""
    # Processing parameters
    batch_size: int = 2000  # Increased for GPU
    max_workers: int = 8
    embedding_batch_size: int = 2048  # Much larger for GPU acceleration
    max_memory_gb: float = 32.0  # Increased for powerful system
    
    # Filtering parameters
    min_commit_date: Optional[datetime] = None
    max_commit_date: Optional[datetime] = None
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    max_files_per_commit: int = 100
    min_message_length: int = 10
    
    # Quality filters
    priority_keywords: List[str] = None
    skip_merge_commits: bool = False  # Changed to False for initial testing
    skip_bot_commits: bool = False    # Changed to False for initial testing
    language_filter: List[str] = None
    path_filter: List[str] = None
    
    # Storage parameters
    cache_dir: Path = Path("data/massive_cache")
    checkpoint_interval: int = 1000
    enable_compression: bool = True
    
    def __post_init__(self):
        """Set defaults for complex fields."""
        if self.include_patterns is None:
            self.include_patterns = [
                r".*\.(cc|cpp|c|h|hpp|js|py|html|css)$",
                r".*/src/.*",
                r".*/chrome/.*",
                r".*/content/.*"
            ]
        
        if self.priority_keywords is None:
            self.priority_keywords = [
                "fix", "bug", "crash", "security", "performance", "optimize",
                "implement", "feature", "api", "breaking", "memory", "leak"
            ]
        
        if self.language_filter is None:
            self.language_filter = []  # Empty list = no filtering
        
        if self.path_filter is None:
            self.path_filter = []  # Empty list = no filtering


class MassiveIngestionPipeline:
    """Pipeline for ingesting all Chromium commits efficiently."""
    
    def __init__(self, config: IngestionConfig):
        """Initialize the massive ingestion pipeline."""
        self.config = config
        self.logger = setup_logger(f"{__name__}.MassiveIngestionPipeline")
        
        # Create cache directories
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.config.cache_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize progress tracking
        self.progress_file = self.config.cache_dir / "progress.json"
        self.stats_file = self.config.cache_dir / "stats.json"
        self.error_log = self.config.cache_dir / "errors.log"
        self.range_tracker = CommitRangeTracker()
        
        # Initialize components
        self.extractor = None
        self.preprocessor = None
        self.chunker = None
        self.embedding_generator = None
        self.vector_db = None
        
        # Progress tracking
        self.total_commits_processed = 0
        self.total_documents_created = 0
        self.total_embeddings_generated = 0
        self.start_time = None
        self.stats = {
            "commits_processed": 0,
            "documents_created": 0,
            "embeddings_generated": 0,
            "errors": 0,
            "avg_processing_rate": 0.0,
            "estimated_completion": None
        }
    
    def initialize_components(self, repo_path: str):
        """Initialize all pipeline components."""
        self.logger.info("Initializing massive ingestion components...")
        
        # Initialize extractor
        self.logger.info("Step 1: Initializing extractor...")
        self.extractor = ChromiumDataExtractor(repo_path)
        
        # Initialize preprocessor and chunker
        self.logger.info("Step 2: Initializing preprocessor and chunker...")
        self.preprocessor = DataPreprocessor()
        self.chunker = TextChunker()
        
        # Initialize embedding generator with optimizations
        self.logger.info("Step 3: Initializing embedding generator...")
        self.embedding_generator = EmbeddingGenerator(
            model_name="BAAI/bge-large-en-v1.5",  # Use BGE-large instead of BGE-M3
            batch_size=self.config.embedding_batch_size
        )
        
        # Initialize vector database
        self.logger.info("Step 4: Initializing vector database...")
        self.vector_db = VectorDatabase(collection_name="chromium_complete")
        
        # Load processed ranges for skipping
        self.logger.info("Step 5: Loading processed ranges...")
        self.processed_ranges = self._load_processed_ranges()
        
        self.logger.info("Components initialized successfully")
    
    def _load_processed_ranges(self) -> List[Tuple[datetime, datetime]]:
        """Load processed date ranges from database."""
        try:
            import json
            from pathlib import Path
            ranges_file = Path('data/massive_cache/all_commit_ranges.json')
            if ranges_file.exists():
                with open(ranges_file) as f:
                    ranges_data = json.load(f)
                processed = []
                for r in ranges_data:
                    try:
                        # Handle both old and new format
                        start_key = 'start_date' if 'start_date' in r else 'actual_last_commit_date'
                        end_key = 'end_date' if 'end_date' in r else 'actual_first_commit_date'
                        
                        # Skip if dates are missing or invalid
                        start_date = r.get(start_key)
                        end_date = r.get(end_key)
                        if not start_date or not end_date or start_date == 'N/A' or end_date == 'N/A':
                            continue
                        
                        start = datetime.fromisoformat(str(start_date).replace('Z', '+00:00'))
                        end = datetime.fromisoformat(str(end_date).replace('Z', '+00:00'))
                        processed.append((start, end))
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"Skipping invalid range entry: {e}")
                        continue
                
                self.logger.info(f"Loaded {len(processed)} processed ranges for skipping:")
                for i, (s, e) in enumerate(processed, 1):
                    self.logger.info(f"  Range {i}: {s.date()} to {e.date()}")
                return processed
        except Exception as e:
            self.logger.warning(f"Could not load processed ranges: {e}")
        return []
    
    def is_commit_in_processed_range(self, commit_date: datetime) -> bool:
        """Check if a commit date falls within any processed range."""
        for start, end in self.processed_ranges:
            if start <= commit_date <= end:
                return True
        return False
    
    def load_progress(self) -> Dict[str, Any]:
        """Load progress from checkpoint."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                self.logger.info(f"Loaded progress: {progress['commits_processed']} commits processed")
                return progress
        return {"commits_processed": 0, "last_commit_sha": None}
    
    def save_progress(self, progress: Dict[str, Any]):
        """Save progress to checkpoint."""
        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
        
        # Update stats
        self.stats.update(progress)
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.stats["avg_processing_rate"] = progress["commits_processed"] / elapsed
            
            # Estimate completion time
            if progress["commits_processed"] > 0:
                total_estimated = self.estimate_total_commits()
                remaining = total_estimated - progress["commits_processed"]
                eta_seconds = remaining / self.stats["avg_processing_rate"]
                eta = datetime.now() + timedelta(seconds=eta_seconds)
                self.stats["estimated_completion"] = eta.isoformat()
        
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2, default=str)
        
        # Write comprehensive status file for monitor (includes DB stats)
        self._write_monitor_status(progress)
    
    def _write_monitor_status(self, progress: Dict[str, Any]):
        """Write comprehensive status for monitor dashboard with real-time timeline updates."""
        try:
            # Get database stats
            db_stats = self.vector_db.get_collection_stats()
            
            # Load existing processed ranges and update current range in real-time
            processed_ranges_data = []
            status_file = Path('data/status.json')
            
            try:
                if status_file.exists():
                    with open(status_file) as f:
                        existing_status = json.load(f)
                        processed_ranges_data = existing_status.get('processed_ranges', [])
                        self.logger.debug(f"Loaded {len(processed_ranges_data)} existing ranges from monitor status")
            except Exception as e:
                self.logger.debug(f"Could not load existing ranges from monitor status: {e}")
            
            # Do NOT add processing ranges to processed_ranges array
            # Only completed ranges belong in processed_ranges
            # In-progress ranges are tracked separately in current_range field
            
            # Build current processing range info from progress
            # Show actual progress: start is the initial commit, end is the latest processed commit
            current_range = None
            if progress.get('start_commit_sha') and progress.get('end_commit_sha'):
                # Use the actual index of the last processed commit
                current_index = progress.get('end_commit_index', progress.get('start_index'))
                
                current_range = {
                    'status': 'processing',
                    'start': {
                        'index': progress.get('start_commit_index', progress.get('start_index')),
                        'sha': progress['start_commit_sha'],
                        'date': progress.get('start_commit_date')
                    },
                    'end': {
                        'index': current_index,
                        'sha': progress['end_commit_sha'],
                        'date': progress.get('end_commit_date')
                    },
                    'target_index': progress.get('end_index')  # Show the target end for reference
                }
            
            # Filter stats to remove first_commit/last_commit, avg_commits_per_range, rebuild_timestamp
            filtered_stats = {k: v for k, v in self.stats.items() 
                            if not k.startswith('first_commit_') and not k.startswith('last_commit_')
                            and k not in ['avg_commits_per_range', 'rebuild_timestamp']}
            
            # Filter progress to remove first_commit_*, last_commit_*, batches_completed
            filtered_progress = {k: v for k, v in progress.items()
                               if not k.startswith('first_commit_') and not k.startswith('last_commit_')
                               and k != 'batches_completed'}
            
            # Combine with progress data
            status = {
                'progress': filtered_progress,
                'processed_ranges': processed_ranges_data,
                'current_range': current_range,
                'stats': filtered_stats
            }
            
            status_file = Path('data/status.json')
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to write monitor status: {e}")
    
    def _clean_range_fields(self, range_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove unnecessary fields from a range, keeping only essential data including dates."""
        # Handle both old format (start_commit/end_commit) and new format (start/end)
        # Also handle both 'range_id' and 'id'
        range_id = range_data.get('id', range_data.get('range_id', 0))
        
        if 'start_commit' in range_data:
            return {
                'id': range_id,
                'start': {
                    'index': range_data.get('start_commit', {}).get('index', range_data.get('start', {}).get('index')),
                    'sha': range_data['start_commit']['sha'],
                    'date': range_data['start_commit'].get('date')
                },
                'end': {
                    'index': range_data.get('end_commit', {}).get('index', range_data.get('end', {}).get('index')),
                    'sha': range_data['end_commit']['sha'],
                    'date': range_data['end_commit'].get('date')
                }
            }
        else:
            # Already in new format, just keep essential fields
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
    
    def _can_combine_ranges(self, range1: Dict[str, Any], range2: Dict[str, Any]) -> bool:
        """Check if two ranges can be combined (adjacent or overlapping)."""
        r1_start = range1['start']['index']
        r1_end = range1['end']['index']
        r2_start = range2['start']['index']
        r2_end = range2['end']['index']
        
        # Check if adjacent (allowing for +/- 1 gap)
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
    
    def _combine_ranges(self, range1: Dict[str, Any], range2: Dict[str, Any]) -> Dict[str, Any]:
        """Combine two ranges into one."""
        r1_start = range1['start']['index']
        r1_end = range1['end']['index']
        r2_start = range2['start']['index']
        r2_end = range2['end']['index']
        
        # Determine overall start and end (preserve dates from appropriate range)
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
    
    def _optimize_ranges(self, ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize ranges by cleaning fields and combining adjacent/overlapping ranges."""
        if not ranges:
            return []
        
        # Clean all ranges first
        cleaned_ranges = [self._clean_range_fields(r) for r in ranges]
        
        # Sort by start index
        cleaned_ranges.sort(key=lambda r: r['start']['index'])
        
        # Combine adjacent/overlapping ranges
        optimized = []
        current = cleaned_ranges[0]
        combined_count = 0
        
        for next_range in cleaned_ranges[1:]:
            if self._can_combine_ranges(current, next_range):
                current_id = current.get('id', current.get('range_id', 0))
                next_id = next_range.get('id', next_range.get('range_id', 0))
                self.logger.info(f"  Combining range {current_id} (idx {current['start']['index']}-{current['end']['index']}) "
                               f"with range {next_id} (idx {next_range['start']['index']}-{next_range['end']['index']})")
                current = self._combine_ranges(current, next_range)
                combined_count += 1
            else:
                optimized.append(current)
                current = next_range
        
        # Add the last range
        optimized.append(current)
        
        # Renumber ids sequentially
        for i, range_data in enumerate(optimized, start=1):
            range_data['id'] = i
        
        if combined_count > 0:
            self.logger.info(f"Optimized: {len(ranges)} ranges → {len(optimized)} ranges ({combined_count} combinations)")
        
        return optimized
    
    def _add_completed_range_to_monitor(self, progress: Dict[str, Any]) -> None:
        """Mark current range as completed in status.json and optimize ranges."""
        try:
            status_file = Path('data/status.json')
            
            if not status_file.exists():
                self.logger.warning("Status file doesn't exist, skipping completion update")
                return
            
            # Load existing status
            with open(status_file) as f:
                existing_status = json.load(f)
            
            processed_ranges = existing_status.get('processed_ranges', [])
            
            # Find and update the current range
            range_marked = False
            for range_data in processed_ranges:
                if range_data.get('start_commit', {}).get('sha') == progress.get('first_commit_sha'):
                    # Calculate accurate commits count for this session
                    commits_in_session = progress.get('commits_processed', 0) - progress.get('commits_processed_before_session', 0)
                    range_data['commits_count'] = commits_in_session
                    
                    # Update final end commit
                    range_data['end_commit'] = {
                        'sha': progress['last_commit_sha'],
                        'date': progress['last_commit_date'],
                        'message': progress.get('last_commit_message', ''),
                        'author': progress.get('last_commit_author', '')
                    }
                    
                    range_marked = True
                    self.logger.info(f"Marked range {range_data['range_id']} as completed ({commits_in_session} commits)")
                    break
            
            if not range_marked:
                # Range doesn't exist yet, create it as completed with new format
                commits_in_session = progress.get('commits_processed', 0) - progress.get('commits_processed_before_session', 0)
                
                new_range = {
                    'id': len(processed_ranges) + 1,
                    'start': {
                        'index': progress.get('start_index'),
                        'sha': progress['first_commit_sha'],
                        'date': progress.get('first_commit_date')
                    },
                    'end': {
                        'index': progress.get('end_index'),
                        'sha': progress['last_commit_sha'],
                        'date': progress.get('last_commit_date')
                    }
                }
                processed_ranges.append(new_range)
                self.logger.info(f"Added completed range {new_range['id']} ({commits_in_session} commits)")
            
            # Optimize ranges: clean fields and combine adjacent/overlapping ranges
            self.logger.info("Optimizing processed ranges...")
            optimized_ranges = self._optimize_ranges(processed_ranges)
            
            # Update the status file with optimized ranges
            existing_status['processed_ranges'] = optimized_ranges
            with open(status_file, 'w') as f:
                json.dump(existing_status, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"Failed to mark range as completed in monitor status: {e}")
    
    def estimate_total_commits(self) -> int:
        """Estimate total number of commits to process."""
        try:
            # Quick count from git
            repo = self.extractor.repo
            if self.config.min_commit_date:
                commits = list(repo.iter_commits(since=self.config.min_commit_date))
                return len(commits)
            else:
                # Rough estimation for full history
                return 1200000  # Chromium has ~1.2M commits
        except Exception as e:
            self.logger.warning(f"Could not estimate total commits: {e}")
            return 1000000  # Conservative estimate
    
    def filter_commit(self, commit_data) -> bool:
        """Apply intelligent filtering to commits."""
        try:
            # Date filtering
            if self.config.min_commit_date and commit_data.commit_date < self.config.min_commit_date:
                return False
            if self.config.max_commit_date and commit_data.commit_date > self.config.max_commit_date:
                return False
            
            # Skip merge commits
            if self.config.skip_merge_commits and len(commit_data.parents) > 1:
                return False
            
            # Skip bot commits
            if self.config.skip_bot_commits:
                bot_indicators = ["autoroll", "bot", "automatic", "roll", "update dependencies"]
                if any(indicator in commit_data.author_email.lower() or 
                       indicator in commit_data.message.lower() 
                       for indicator in bot_indicators):
                    return False
            
            # File count filtering
            if len(commit_data.files_changed) > self.config.max_files_per_commit:
                return False
            
            # Message length filtering
            if len(commit_data.message.strip()) < self.config.min_message_length:
                return False
            
            # Path filtering
            if self.config.path_filter:
                relevant_files = [f for f in commit_data.files_changed 
                                if any(path in f for path in self.config.path_filter)]
                if not relevant_files:
                    return False
            
            # Priority keyword boost (always include if present)
            if any(keyword in commit_data.message.lower() 
                   for keyword in self.config.priority_keywords):
                return True
            
            # Regular inclusion criteria
            return True
            
        except Exception as e:
            self.logger.warning(f"Error filtering commit {getattr(commit_data, 'sha', 'unknown')}: {e}")
            return False
    
    def process_commit_batch(self, commits_batch: List, start_index: int = 0) -> Tuple[List[VectorDocument], int]:
        """Process a batch of commits into vector documents."""
        vector_documents = []
        processed_count = 0
        
        for commit in commits_batch:
            try:
                if not self.filter_commit(commit):
                    continue
                
                # Create commit content
                commit_content = self.create_commit_content(commit)
                
                # Preprocess content
                processed_content = self.preprocessor.clean_commit_message(commit_content)
                
                # Create chunks
                chunks = self.chunker.chunk(
                    content=processed_content,
                    document_id=f"commit_{commit.sha}"
                )
                
                # Add metadata to chunks
                for chunk in chunks:
                    chunk.metadata.update({
                        'commit_sha': commit.sha,
                        'commit_index': start_index + commits_batch.index(commit),
                        'author': commit.author_name,
                        'commit_date': commit.commit_date.isoformat(),
                        'files_changed': len(commit.files_changed),
                        'additions': commit.additions,
                        'deletions': commit.deletions,
                        'source_type': 'chromium_commit_complete',
                        'priority_score': self.calculate_priority_score(commit)
                    })
                
                # Convert to vector documents (embeddings added later)
                for chunk in chunks:
                    vector_doc = VectorDocument(
                        id=chunk.id,
                        content=chunk.content,
                        embedding=None,  # Will be filled by embedding generation
                        metadata=chunk.metadata
                    )
                    vector_documents.append(vector_doc)
                
                processed_count += 1
                
            except Exception as e:
                self.logger.error(f"Error processing commit {commit.sha}: {e}")
                with open(self.error_log, 'a') as f:
                    f.write(f"{datetime.now()}: Error processing {commit.sha}: {e}\n")
                continue
        
        return vector_documents, processed_count
    
    def create_commit_content(self, commit) -> str:
        """Create formatted content for a commit."""
        return f"""
Commit: {commit.sha}
Author: {commit.author_name} <{commit.author_email}>
Date: {commit.commit_date}
Message: {commit.message}

Files Changed ({len(commit.files_changed)}):
{chr(10).join(f"  - {file}" for file in commit.files_changed[:20])}
{"  ..." if len(commit.files_changed) > 20 else ""}

Changes: +{commit.additions} -{commit.deletions}
""".strip()
    
    def calculate_priority_score(self, commit) -> float:
        """Calculate priority score for a commit (0.0 to 1.0)."""
        score = 0.0
        
        # Keywords in message
        for keyword in self.config.priority_keywords:
            if keyword in commit.message.lower():
                score += 0.1
        
        # Recent commits get higher priority
        days_old = (datetime.now() - commit.commit_date).days
        if days_old < 30:
            score += 0.3
        elif days_old < 365:
            score += 0.2
        elif days_old < 365 * 3:
            score += 0.1
        
        # Reasonable file count (not too many, not too few)
        file_count = len(commit.files_changed)
        if 3 <= file_count <= 20:
            score += 0.2
        elif 1 <= file_count <= 50:
            score += 0.1
        
        return min(score, 1.0)
    
    def generate_embeddings_batch(self, documents: List[Dict]) -> List[VectorDocument]:
        """Generate embeddings for a batch of document dictionaries."""
        if not documents:
            return []
        
        # Extract texts from dict format
        texts = [doc['content'] if isinstance(doc, dict) else doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_generator.encode_texts(texts)
        
        # PERFORMANCE FIX: Convert all embeddings at once (vectorized operation)
        # This is 10-20x faster than calling .tolist() in a loop
        import numpy as np
        if isinstance(embeddings, np.ndarray):
            embeddings_list = embeddings.tolist()  # Single vectorized conversion
        else:
            embeddings_list = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        
        self.logger.info(f"Converting {len(documents)} documents to VectorDocument objects...")
        convert_start = time.time()
        
        # Convert to VectorDocument objects with embeddings
        vector_documents = []
        for doc, embedding in zip(documents, embeddings_list):
            if isinstance(doc, dict):
                # Convert dict to VectorDocument
                vector_doc = VectorDocument(
                    id=doc['commit_sha'],
                    content=doc['content'],
                    embedding=embedding,
                    metadata=doc
                )
            else:
                # Already a VectorDocument, just add embedding
                doc.embedding = embedding
                vector_doc = doc
            
            vector_documents.append(vector_doc)
        
        convert_time = time.time() - convert_start
        self.logger.info(f"Document conversion completed in {convert_time:.2f}s")
        
        return vector_documents
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor system resources."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
    
    def trigger_backup(self, reason: str = "manual"):
        """Trigger a backup by creating flag file for backup scheduler."""
        try:
            BACKUP_TRIGGER_FLAG.parent.mkdir(parents=True, exist_ok=True)
            with open(BACKUP_TRIGGER_FLAG, 'w') as f:
                f.write(f"{datetime.now().isoformat()}|{reason}\n")
            self.logger.info(f"✓ Backup triggered: {reason}")
        except Exception as e:
            self.logger.warning(f"Failed to trigger backup: {e}")
    
    def run_massive_ingestion(
        self,
        repo_path: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_commits: Optional[int] = None
    ):
        """Run the massive ingestion pipeline."""
        self.start_time = time.time()
        self.logger.info(f"Starting massive Chromium ingestion from {repo_path}")
        
        # Initialize components
        self.initialize_components(repo_path)
        
        # Load previous progress
        progress = self.load_progress()
        
        # Restore totals from progress for accurate tracking
        self.total_commits_processed = progress.get('commits_processed', 0)
        self.total_documents_created = progress.get('documents_created', 0)
        self.total_embeddings_generated = progress.get('embeddings_generated', 0)
        
        # Configure date range
        if start_date:
            self.config.min_commit_date = start_date
        if end_date:
            self.config.max_commit_date = end_date
        
        # Get initial database stats
        initial_stats = self.vector_db.get_collection_stats()
        initial_count = initial_stats.get('total_documents', 0)
        
        self.logger.info(f"Initial database size: {initial_count} documents")
        self.logger.info(f"Resuming from: {progress['commits_processed']} commits processed in previous sessions")
        self.logger.info(f"Previous session created: {progress.get('documents_created', 0)} documents")
        
        # Store commits before this session for accurate range tracking
        commits_before_session = progress.get('commits_processed', 0)
        progress['commits_processed_before_session'] = commits_before_session
        
        # Trigger backup at start of ingestion session
        self.trigger_backup(f"session_start_{initial_count}_docs")
        
        processed_in_session = 0
        batch_count = 0
        last_backup_milestone = (initial_count // 10000) * 10000  # Track last 10K milestone
        total_batches = 0  # Will be incremented as batches are processed (streaming)
        
        try:
            # Extract commits in batches
            commit_batches = self.extractor.extract_commits(
                max_count=max_commits,
                since=self.config.min_commit_date,
                until=self.config.max_commit_date,
                include_diffs=False,  # Skip diffs for memory efficiency
                batch_size=self.config.batch_size
            )
            
            self.logger.info(f"Starting to iterate over commit batches...")
            
            for commit_batch in commit_batches:
                batch_count += 1
                batch_start_time = time.time()
                
                self.logger.info(f"Received batch {batch_count} with {len(commit_batch)} commits")
                
                # Skip if we've already processed this batch
                if batch_count <= progress.get('batches_completed', 0):
                    self.logger.info(f"Skipping already processed batch {batch_count}")
                    continue
                
                # Filter out commits that are already in processed ranges
                filtered_batch = []
                skipped_count = 0
                for commit in commit_batch:
                    commit_date = commit.commit_date
                    if not hasattr(commit_date, 'replace'):  # Already datetime
                        pass
                    elif isinstance(commit_date, str):
                        commit_date = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                    
                    if self.is_commit_in_processed_range(commit_date):
                        skipped_count += 1
                    else:
                        filtered_batch.append(commit)
                
                if skipped_count > 0:
                    self.logger.info(f"Skipped {skipped_count} commits already in processed ranges")
                
                if not filtered_batch:
                    self.logger.info(f"Batch {batch_count} fully processed, skipping")
                    continue
                
                self.logger.info(f"Processing batch {batch_count} ({len(filtered_batch)} commits after filtering)")
                
                # Track first and last commit info from filtered batch
                # NOTE: Git log returns commits in reverse chronological order (newest first)
                # So batch[0] is the LATEST (newest) commit, batch[-1] is EARLIEST (oldest)
                if filtered_batch:
                    latest_commit = filtered_batch[0]   # Newest commit (most recent date)
                    earliest_commit = filtered_batch[-1]  # Oldest commit (earliest date)
                    
                    # Validate date order
                    latest_date = latest_commit.commit_date if hasattr(latest_commit.commit_date, 'isoformat') else datetime.fromisoformat(str(latest_commit.commit_date))
                    earliest_date = earliest_commit.commit_date if hasattr(earliest_commit.commit_date, 'isoformat') else datetime.fromisoformat(str(earliest_commit.commit_date))
                    
                    # Safety check: ensure earliest <= latest
                    if earliest_date > latest_date:
                        self.logger.warning(f"Date order violation detected: earliest={earliest_date} > latest={latest_date}, swapping")
                        latest_commit, earliest_commit = earliest_commit, latest_commit
                        latest_date, earliest_date = earliest_date, latest_date
                    
                    # On first batch: set the END (latest/newest commit) - this won't change
                    if batch_count == 1 or not progress.get('end_commit_sha'):
                        progress['end_commit_sha'] = latest_commit.sha
                        progress['end_commit_date'] = latest_commit.commit_date.isoformat() if hasattr(latest_commit.commit_date, 'isoformat') else str(latest_commit.commit_date)
                        progress['end_commit_message'] = latest_commit.message[:100]
                        progress['end_commit_author'] = latest_commit.author_name
                    
                    # Always update the START (earliest commit so far) - this moves backward as we process more
                    progress['start_commit_sha'] = earliest_commit.sha
                    progress['start_commit_date'] = earliest_commit.commit_date.isoformat() if hasattr(earliest_commit.commit_date, 'isoformat') else str(earliest_commit.commit_date)
                    progress['start_commit_message'] = earliest_commit.message[:100]
                    progress['start_commit_author'] = earliest_commit.author_name
                    
                    # Legacy fields for backward compatibility (but using correct semantics)
                    if batch_count == 1 or not progress.get('first_commit_sha'):
                        # 'first' means first processed (latest date)
                        progress['first_commit_sha'] = latest_commit.sha
                        progress['first_commit_date'] = latest_commit.commit_date.isoformat() if hasattr(latest_commit.commit_date, 'isoformat') else str(latest_commit.commit_date)
                        progress['first_commit_message'] = latest_commit.message[:100]
                        progress['first_commit_author'] = latest_commit.author_name
                    
                    # 'last' means last processed so far (going backward = earliest date so far)
                    progress['last_commit_sha'] = earliest_commit.sha
                    progress['last_commit_date'] = earliest_commit.commit_date.isoformat() if hasattr(earliest_commit.commit_date, 'isoformat') else str(earliest_commit.commit_date)
                    progress['last_commit_message'] = earliest_commit.message[:100]
                    progress['last_commit_author'] = earliest_commit.author_name
                
                # Process commits into vector documents
                current_commit_index = progress.get('commit_index_current', 0)
                vector_documents, commits_processed = self.process_commit_batch(filtered_batch, current_commit_index)
                
                if vector_documents:
                    # Generate embeddings
                    self.logger.info(f"Generating embeddings for {len(vector_documents)} documents...")
                    vector_documents = self.generate_embeddings_batch(vector_documents)
                    
                    # Store in vector database
                    self.logger.info(f"Storing {len(vector_documents)} documents in vector database...")
                    added_count = self.vector_db.add_documents(vector_documents)
                    
                    self.total_documents_created += added_count
                    self.total_embeddings_generated += len(vector_documents)
                
                # Update progress
                self.total_commits_processed += commits_processed
                processed_in_session += commits_processed
                progress['commit_index_current'] = current_commit_index + len(filtered_batch)
                
                batch_time = time.time() - batch_start_time
                
                # Update and save progress
                progress.update({
                    "commits_processed": self.total_commits_processed,
                    "documents_created": self.total_documents_created,
                    "embeddings_generated": self.total_embeddings_generated,
                    "last_batch_time": batch_time,
                    "session_start": self.start_time,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "max_commits": max_commits
                })
                
                self.save_progress(progress)
                
                # Reload processed ranges every 10 batches to stay current
                if batch_count % 10 == 0:
                    self.processed_ranges = self._load_processed_ranges()
                
                # Trigger backup every 10 batches
                if batch_count % 10 == 0:
                    self.trigger_backup(f"batch_checkpoint_{batch_count}_batches_{self.total_documents_created}_docs")
                
                # Trigger backup at 10K document milestones
                current_milestone = (self.total_documents_created // 10000) * 10000
                if current_milestone > last_backup_milestone and current_milestone > 0:
                    self.trigger_backup(f"milestone_{current_milestone}_documents")
                    last_backup_milestone = current_milestone
                    self.logger.info(f"🎯 Milestone reached: {current_milestone:,} documents")
                
                # Log progress (for streaming batches, total is unknown)
                resources = self.monitor_resources()
                self.logger.info(
                    f"Batch {batch_count} completed: "
                    f"{commits_processed} commits, {len(vector_documents)} documents, "
                    f"{batch_time:.2f}s, Memory: {resources['memory_percent']:.1f}%"
                )
                
                # Memory management
                if resources['memory_percent'] > 85:
                    self.logger.warning("High memory usage detected, forcing garbage collection")
                    import gc
                    gc.collect()
                
                # Check if we should stop
                if max_commits and self.total_commits_processed >= max_commits:
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Ingestion interrupted by user")
        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}", exc_info=True)
        finally:
            # Final statistics
            total_time = time.time() - self.start_time
            final_stats = self.vector_db.get_collection_stats()
            final_count = final_stats.get('total_documents', 0)
            
            # Record the processed commit range
            if progress.get('first_commit_sha') and progress.get('last_commit_sha'):
                phase_name = f"Phase ({start_date.strftime('%Y-%m-%d') if start_date else 'earliest'} to {end_date.strftime('%Y-%m-%d') if end_date else 'latest'})"
                self.range_tracker.add_range(
                    start_date=start_date.isoformat() if start_date else "earliest",
                    end_date=end_date.isoformat() if end_date else "latest",
                    first_commit_sha=progress['first_commit_sha'],
                    last_commit_sha=progress['last_commit_sha'],
                    first_commit_date=progress['first_commit_date'],
                    last_commit_date=progress['last_commit_date'],
                    commits_processed=self.total_commits_processed,
                    documents_created=self.total_documents_created,
                    phase_name=phase_name
                )
                self.logger.info(f"Recorded commit range: {progress['first_commit_date']} to {progress['last_commit_date']}")
                
                # Add completed range to status.json processed_ranges
                self._add_completed_range_to_monitor(progress)
            
            # Trigger final backup on completion
            self.trigger_backup(f"session_complete_{final_count}_docs_{processed_in_session}_new")
            
            self.logger.info("Massive ingestion session completed:")
            self.logger.info(f"  Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
            self.logger.info(f"  Commits processed this session: {processed_in_session}")
            self.logger.info(f"  Total commits processed: {self.total_commits_processed}")
            self.logger.info(f"  Documents created: {self.total_documents_created}")
            self.logger.info(f"  Database size: {initial_count} → {final_count}")
            self.logger.info(f"  Processing rate: {processed_in_session/total_time:.2f} commits/sec")

    def run_ingestion_from_commit_list(
        self,
        repo_path: str,
        commit_shas: List[str],
        commit_indices: Dict[str, int]
    ):
        """
        Run ingestion from a pre-filtered list of commit SHAs.
        This is used by index-based ingestion to process specific commits.
        
        Args:
            repo_path: Path to the git repository
            commit_shas: List of commit SHAs to process
            commit_indices: Mapping of commit SHA to its git log index
        """
        self.start_time = time.time()
        self.logger.info(f"Starting ingestion of {len(commit_shas):,} pre-filtered commits")
        
        # Initialize components
        self.initialize_components(repo_path)
        
        # Load previous progress
        progress = self.load_progress()
        
        # Preserve start_index and end_index if they exist
        start_index = progress.get('start_index')
        end_index = progress.get('end_index')
        
        # Restore totals from progress
        self.total_commits_processed = progress.get('commits_processed', 0)
        self.total_documents_created = progress.get('documents_created', 0)
        self.total_embeddings_generated = progress.get('embeddings_generated', 0)
        
        # Get initial database stats
        initial_stats = self.vector_db.get_collection_stats()
        initial_count = initial_stats.get('total_documents', 0)
        
        self.logger.info(f"Initial database size: {initial_count} documents")
        self.logger.info(f"Resuming from: {progress['commits_processed']} commits processed in previous sessions")
        
        # Trigger backup at start
        self.trigger_backup(f"session_start_{initial_count}_docs")
        
        processed_in_session = 0
        batch_count = 0
        last_backup_milestone = (initial_count // 10000) * 10000
        total_batches = (len(commit_shas) + self.config.batch_size - 1) // self.config.batch_size
        
        self.logger.info(f"Will process {len(commit_shas)} commits in {total_batches} batches of {self.config.batch_size}")
        
        try:
            # Process commits in batches
            for i in range(0, len(commit_shas), self.config.batch_size):
                batch_shas = commit_shas[i:i + self.config.batch_size]
                batch_count += 1
                batch_start_time = time.time()
                
                self.logger.info(f"Fetching batch {batch_count} ({len(batch_shas)} commits)...")
                
                # Extract commit data for this batch using git log
                commits_data = []
                
                # Use git log to fetch commit details efficiently
                format_str = '%H%n%an%n%ae%n%ai%n%B%n--END--'
                
                self.logger.info(f"CHECKPOINT 1: About to fetch git log for {len(batch_shas)} SHAs")
                
                try:
                    # Windows-specific: Use CREATE_NEW_PROCESS_GROUP to prevent signal propagation
                    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
                    
                    self.logger.info(f"CHECKPOINT 2: Starting subprocess.run with creationflags={creationflags}")
                    
                    result = subprocess.run(
                        ['git', 'log', '--format=' + format_str, '--no-walk'] + batch_shas,
                        cwd=self.extractor.repo_path,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=60,
                        creationflags=creationflags
                    )
                    
                    self.logger.info(f"CHECKPOINT 3: subprocess.run completed successfully, {len(result.stdout):,} bytes")
                    
                    # Parse the output
                    commit_texts = result.stdout.split('--END--\n')
                    for i, commit_text in enumerate(commit_texts):
                        if not commit_text.strip():
                            continue
                        
                        lines = commit_text.strip().split('\n')
                        if len(lines) < 4:
                            continue
                        
                        sha = lines[0]
                        author = lines[1]
                        author_email = lines[2]
                        date_str = lines[3]
                        message = '\n'.join(lines[4:])
                        
                        # Get stats for this commit
                        try:
                            if i == 0:  # Log first commit stats fetch
                                self.logger.info(f"CHECKPOINT 4: Starting git stats loop (500 commits)")
                            
                            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == 'win32' else 0
                            stats_result = subprocess.run(
                                ['git', 'show', '--stat', '--format=', sha],
                                cwd=self.extractor.repo_path,
                                capture_output=True,
                                text=True,
                                check=True,
                                timeout=10,
                                creationflags=creationflags
                            )
                            
                            if i % 100 == 0:  # Log progress every 100 commits
                                self.logger.info(f"CHECKPOINT 5: Processed {i}/{len(commit_texts)} commit stats")
                            
                            # Parse stats
                            files_changed = 0
                            additions = 0
                            deletions = 0
                            for line in stats_result.stdout.split('\n'):
                                if '|' in line:
                                    files_changed += 1
                                if 'insertion' in line or 'deletion' in line:
                                    # Parse summary line like: "5 files changed, 123 insertions(+), 45 deletions(-)"
                                    parts = line.split(',')
                                    for part in parts:
                                        part = part.strip()
                                        if 'insertion' in part:
                                            try:
                                                # Extract number before "insertion"
                                                num_str = part.split()[0]
                                                additions = int(num_str)
                                            except (ValueError, IndexError):
                                                pass
                                        if 'deletion' in part:
                                            try:
                                                # Extract number before "deletion"
                                                num_str = part.split()[0]
                                                deletions = int(num_str)
                                            except (ValueError, IndexError):
                                                pass
                            
                            commit_data = {
                                'sha': sha,
                                'index': commit_indices.get(sha, -1),
                                'author': author,
                                'author_email': author_email,
                                'date': datetime.fromisoformat(date_str.replace(' ', 'T')),
                                'message': message,
                                'files_changed': files_changed,
                                'additions': additions,
                                'deletions': deletions
                            }
                            commits_data.append(commit_data)
                        except Exception as e:
                            self.logger.warning(f"Failed to get stats for commit {sha[:8]}: {e}")
                            continue
                
                except subprocess.TimeoutExpired:
                    self.logger.error(f"Git operation timed out for batch {batch_count}")
                    continue
                except Exception as e:
                    self.logger.error(f"Failed to fetch batch {batch_count}: {e}")
                    continue
                
                self.logger.info(f"CHECKPOINT 6: Completed stats for all commits, got {len(commits_data)} valid commits")
                
                if not commits_data:
                    self.logger.warning(f"No valid commits in batch {batch_count}, skipping")
                    continue
                
                self.logger.info(f"Processing batch {batch_count} ({len(commits_data)} commits after filtering)")
                
                # Convert to documents
                documents = [self._commit_to_document(c) for c in commits_data]
                
                # Generate embeddings
                self.logger.info(f"Generating embeddings for {len(documents)} documents...")
                vector_documents = self.generate_embeddings_batch(documents)
                
                # Store in vector database
                self.logger.info(f"Storing {len(vector_documents)} documents in vector database...")
                added_count = self.vector_db.add_documents(vector_documents)
                
                # Update progress tracking
                if commits_data:
                    if not progress.get('first_commit_sha'):
                        progress['first_commit_sha'] = commits_data[0]['sha']
                        progress['first_commit_date'] = commits_data[0]['date'].isoformat()
                    
                    progress['last_commit_sha'] = commits_data[-1]['sha']
                    progress['last_commit_date'] = commits_data[-1]['date'].isoformat()
                
                # Update counters
                commits_processed = len(commits_data)
                self.total_commits_processed += commits_processed
                self.total_documents_created += added_count
                self.total_embeddings_generated += len(vector_documents)
                processed_in_session += commits_processed
                
                batch_time = time.time() - batch_start_time
                
                # Set start_commit fields for current_range tracking (first batch only)
                if batch_count == 1 and commits_data:
                    progress['start_commit_sha'] = commits_data[0]['sha']
                    progress['start_commit_date'] = commits_data[0]['date'].isoformat()
                    progress['start_commit_message'] = commits_data[0]['message'][:100]
                    progress['start_commit_author'] = commits_data[0]['author']
                    progress['start_commit_index'] = commits_data[0].get('index', progress.get('start_index'))
                
                # Always update end_commit to track real-time progress
                if commits_data:
                    progress['end_commit_sha'] = commits_data[-1]['sha']
                    progress['end_commit_date'] = commits_data[-1]['date'].isoformat()
                    progress['end_commit_message'] = commits_data[-1]['message'][:100]
                    progress['end_commit_author'] = commits_data[-1]['author']
                    progress['end_commit_index'] = commits_data[-1].get('index', -1)
                
                # Update and save progress (preserve index fields)
                progress.update({
                    "commits_processed": self.total_commits_processed,
                    "documents_created": self.total_documents_created,
                    "embeddings_generated": self.total_embeddings_generated,
                    "last_batch_time": batch_time,
                    "session_start": self.start_time,
                    "start_index": start_index,
                    "end_index": end_index,
                    "commits_processed_before_session": progress.get('commits_processed_before_session', self.total_commits_processed - processed_in_session)
                })
                
                self.save_progress(progress)
                
                # Trigger backup every 10 batches
                if batch_count % 10 == 0:
                    self.trigger_backup(f"batch_checkpoint_{batch_count}_batches_{self.total_documents_created}_docs")
                
                # Trigger backup at 10K milestones
                current_milestone = (self.total_documents_created // 10000) * 10000
                if current_milestone > last_backup_milestone and current_milestone > 0:
                    self.trigger_backup(f"milestone_{current_milestone}_documents")
                    last_backup_milestone = current_milestone
                    self.logger.info(f"🎯 Milestone reached: {current_milestone:,} documents")
                
                # Log progress with batches remaining
                resources = self.monitor_resources()
                batches_remaining = total_batches - batch_count
                self.logger.info(
                    f"Batch {batch_count}/{total_batches} completed ({batches_remaining} batches left): "
                    f"{commits_processed} commits, {len(vector_documents)} documents, "
                    f"{batch_time:.2f}s, Memory: {resources['memory_percent']:.1f}%"
                )
                
                # Memory management
                if resources['memory_percent'] > 85:
                    self.logger.warning("High memory usage detected, forcing garbage collection")
                    import gc
                    gc.collect()
        
        except KeyboardInterrupt:
            self.logger.info("Ingestion interrupted by user")
        except Exception as e:
            self.logger.error(f"Ingestion failed: {e}", exc_info=True)
        finally:
            # Final statistics
            total_time = time.time() - self.start_time
            final_stats = self.vector_db.get_collection_stats()
            final_count = final_stats.get('total_documents', 0)
            
            # Record the processed commit range if we have data
            if progress.get('first_commit_sha') and progress.get('last_commit_sha'):
                self._add_completed_range_to_monitor(progress)
                self.logger.info(f"Recorded commit range: {progress['first_commit_date']} to {progress['last_commit_date']}")
            
            # Trigger final backup
            self.trigger_backup(f"session_complete_{final_count}_docs_{processed_in_session}_new")
            
            self.logger.info("Ingestion session completed:")
            self.logger.info(f"  Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
            self.logger.info(f"  Commits processed this session: {processed_in_session}")
            self.logger.info(f"  Total commits processed: {self.total_commits_processed}")
            self.logger.info(f"  Documents created: {self.total_documents_created}")
            self.logger.info(f"  Database size: {initial_count} → {final_count}")
            if total_time > 0:
                self.logger.info(f"  Processing rate: {processed_in_session/total_time:.2f} commits/sec")
    
    def _commit_to_document(self, commit_data: Dict) -> Dict:
        """Convert commit data dictionary to document format."""
        return {
            'commit_sha': commit_data['sha'],
            'commit_index': commit_data.get('index', -1),
            'commit_date': commit_data['date'].isoformat() if hasattr(commit_data['date'], 'isoformat') else str(commit_data['date']),
            'author': commit_data['author'],
            'author_email': commit_data['author_email'],
            'message': commit_data['message'],
            'files_changed': commit_data['files_changed'],
            'additions': commit_data['additions'],
            'deletions': commit_data['deletions'],
            'content': f"Commit {commit_data['sha'][:8]} by {commit_data['author']}\n\n{commit_data['message']}\n\nFiles changed: {commit_data['files_changed']}, +{commit_data['additions']} -{commit_data['deletions']}"
        }


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Massive Chromium commit ingestion")
    parser.add_argument("--repo-path", default="d:\\r\\cr\\src",
                       help="Path to Chromium repository")
    parser.add_argument("--start-date", type=str,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--max-commits", type=int,
                       help="Maximum commits to process")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for processing")
    parser.add_argument("--embedding-batch-size", type=int, default=512,
                       help="Batch size for embedding generation")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum worker processes")
    parser.add_argument("--max-memory-gb", type=float, default=16.0,
                       help="Maximum memory usage in GB")
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    # Create configuration
    config = IngestionConfig(
        batch_size=args.batch_size,
        embedding_batch_size=args.embedding_batch_size,
        max_workers=args.max_workers,
        max_memory_gb=args.max_memory_gb,
        min_commit_date=start_date,
        max_commit_date=end_date
    )
    
    # Run massive ingestion
    pipeline = MassiveIngestionPipeline(config)
    pipeline.run_massive_ingestion(
        repo_path=args.repo_path,
        start_date=start_date,
        end_date=end_date,
        max_commits=args.max_commits
    )


if __name__ == "__main__":
    main()