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
from tqdm import tqdm
import psutil

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.data.chromium import ChromiumDataExtractor
from rag_system.data.preprocessor import DataPreprocessor
from rag_system.data.chunker import TextChunker
from rag_system.embeddings.generator import EmbeddingGenerator
from rag_system.vector.database import VectorDatabase, VectorDocument


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
            "batches_completed": 0,
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
        self.extractor = ChromiumDataExtractor(repo_path)
        
        # Initialize preprocessor and chunker
        self.preprocessor = DataPreprocessor()
        self.chunker = TextChunker()
        
        # Initialize embedding generator with optimizations
        self.embedding_generator = EmbeddingGenerator(
            model_name="BAAI/bge-large-en-v1.5",  # Use BGE-large instead of BGE-M3
            batch_size=self.config.embedding_batch_size
        )
        
        # Initialize vector database
        self.vector_db = VectorDatabase(collection_name="chromium_complete")
        
        self.logger.info("Components initialized successfully")
    
    def load_progress(self) -> Dict[str, Any]:
        """Load progress from checkpoint."""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                progress = json.load(f)
                self.logger.info(f"Loaded progress: {progress['commits_processed']} commits processed")
                return progress
        return {"commits_processed": 0, "last_commit_sha": None, "batches_completed": 0}
    
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
    
    def process_commit_batch(self, commits_batch: List) -> Tuple[List[VectorDocument], int]:
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
    
    def generate_embeddings_batch(self, documents: List[VectorDocument]) -> List[VectorDocument]:
        """Generate embeddings for a batch of documents."""
        if not documents:
            return documents
        
        # Extract texts
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_generator.encode_texts(texts)
        
        # Assign embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        
        return documents
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor system resources."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
    
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
        
        processed_in_session = 0
        batch_count = 0
        
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
                
                self.logger.info(f"Processing batch {batch_count} ({len(commit_batch)} commits)")
                
                # Track first and last commit info
                if commit_batch:
                    first_commit = commit_batch[0]
                    last_commit = commit_batch[-1]
                    
                    # Update progress with commit range info (CommitData objects, not dicts)
                    if batch_count == 1 or not progress.get('first_commit_sha'):
                        progress['first_commit_sha'] = first_commit.sha
                        progress['first_commit_date'] = first_commit.commit_date.isoformat() if hasattr(first_commit.commit_date, 'isoformat') else str(first_commit.commit_date)
                        progress['first_commit_message'] = first_commit.message[:100]  # First 100 chars
                    
                    progress['last_commit_sha'] = last_commit.sha
                    progress['last_commit_date'] = last_commit.commit_date.isoformat() if hasattr(last_commit.commit_date, 'isoformat') else str(last_commit.commit_date)
                    progress['last_commit_message'] = last_commit.message[:100]  # First 100 chars
                
                # Process commits into vector documents
                vector_documents, commits_processed = self.process_commit_batch(commit_batch)
                
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
                
                batch_time = time.time() - batch_start_time
                
                # Update and save progress
                progress.update({
                    "commits_processed": self.total_commits_processed,
                    "batches_completed": batch_count,
                    "documents_created": self.total_documents_created,
                    "embeddings_generated": self.total_embeddings_generated,
                    "last_batch_time": batch_time,
                    "session_start": self.start_time,
                    "start_date": start_date.isoformat() if start_date else None,
                    "end_date": end_date.isoformat() if end_date else None,
                    "max_commits": max_commits
                })
                
                self.save_progress(progress)
                
                # Log progress
                resources = self.monitor_resources()
                self.logger.info(
                    f"Batch {batch_count} completed: {commits_processed} commits, "
                    f"{len(vector_documents)} documents, {batch_time:.2f}s, "
                    f"Memory: {resources['memory_percent']:.1f}%"
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
            
            self.logger.info("Massive ingestion session completed:")
            self.logger.info(f"  Total time: {total_time:.2f}s ({total_time/3600:.2f}h)")
            self.logger.info(f"  Commits processed this session: {processed_in_session}")
            self.logger.info(f"  Total commits processed: {self.total_commits_processed}")
            self.logger.info(f"  Documents created: {self.total_documents_created}")
            self.logger.info(f"  Database size: {initial_count} → {final_count}")
            self.logger.info(f"  Processing rate: {processed_in_session/total_time:.2f} commits/sec")


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