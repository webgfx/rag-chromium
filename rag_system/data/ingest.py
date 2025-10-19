"""
Main data ingestion pipeline script.
Coordinates the extraction, preprocessing, and chunking of Chromium data.
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import sys

from .chromium import ChromiumDataExtractor, CommitData
from .preprocessor import DataPreprocessor, ProcessedDocument
from .chunker import TextChunker, CodeChunker, DiffChunker, Chunk
from ..core.config import get_config
from ..core.logger import setup_logger, PerformanceLogger


class DataIngestionPipeline:
    """
    Main pipeline for ingesting and processing Chromium data.
    Coordinates extraction, preprocessing, and chunking.
    """
    
    def __init__(self):
        """Initialize the data ingestion pipeline."""
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.DataIngestionPipeline")
        
        # Initialize components
        self.extractor = ChromiumDataExtractor()
        self.preprocessor = DataPreprocessor()
        
        # Initialize chunkers
        self.text_chunker = TextChunker(
            chunk_size=self.config.data.max_chunk_size,
            overlap=self.config.data.chunk_overlap
        )
        self.code_chunker = CodeChunker(
            chunk_size=self.config.data.max_chunk_size,
            overlap=self.config.data.chunk_overlap
        )
        self.diff_chunker = DiffChunker(
            chunk_size=self.config.data.max_chunk_size,
            overlap=self.config.data.chunk_overlap
        )
        
        # Output directory
        self.output_dir = Path(self.config.data.cache_dir) / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def ingest_recent_commits(self, days: int = 30, batch_size: int = 1000) -> Dict[str, Any]:
        """
        Ingest and process recent commits.
        
        Args:
            days: Number of recent days to process
            batch_size: Batch size for processing
        
        Returns:
            Summary of ingestion results
        """
        with PerformanceLogger(self.logger, f"recent commits ingestion ({days} days)"):
            total_commits = 0
            total_chunks = 0
            all_chunks = []
            
            # Extract commits in batches
            for commit_batch in self.extractor.extract_recent_commits(days=days, batch_size=batch_size):
                self.logger.info(f"Processing batch of {len(commit_batch)} commits")
                
                # Process commits
                processed_docs = self._process_commits(commit_batch)
                
                # Chunk documents
                batch_chunks = self._chunk_documents(processed_docs)
                
                all_chunks.extend(batch_chunks)
                total_commits += len(commit_batch)
                total_chunks += len(batch_chunks)
                
                self.logger.info(f"Processed {len(commit_batch)} commits into {len(batch_chunks)} chunks")
            
            # Save results
            results_file = self.output_dir / f"recent_commits_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self._save_chunks(all_chunks, results_file)
            
            summary = {
                'total_commits': total_commits,
                'total_chunks': total_chunks,
                'output_file': str(results_file),
                'ingestion_date': datetime.now().isoformat(),
                'days_processed': days
            }
            
            self.logger.info(f"Ingestion complete: {summary}")
            return summary
    
    def ingest_commit_range(
        self,
        max_count: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        paths: Optional[List[str]] = None,
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Ingest commits from a specific range.
        
        Args:
            max_count: Maximum number of commits
            since: Start date
            until: End date
            paths: Specific file paths to filter
            batch_size: Batch size for processing
        
        Returns:
            Summary of ingestion results
        """
        with PerformanceLogger(self.logger, "commit range ingestion"):
            total_commits = 0
            total_chunks = 0
            all_chunks = []
            
            # Extract commits in batches
            for commit_batch in self.extractor.extract_commits(
                max_count=max_count,
                since=since,
                until=until,
                paths=paths,
                batch_size=batch_size
            ):
                # Process commits
                processed_docs = self._process_commits(commit_batch)
                
                # Chunk documents
                batch_chunks = self._chunk_documents(processed_docs)
                
                all_chunks.extend(batch_chunks)
                total_commits += len(commit_batch)
                total_chunks += len(batch_chunks)
                
                self.logger.info(f"Processed {len(commit_batch)} commits into {len(batch_chunks)} chunks")
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = self.output_dir / f"commit_range_{timestamp}.json"
            self._save_chunks(all_chunks, results_file)
            
            summary = {
                'total_commits': total_commits,
                'total_chunks': total_chunks,
                'output_file': str(results_file),
                'ingestion_date': datetime.now().isoformat(),
                'parameters': {
                    'max_count': max_count,
                    'since': since.isoformat() if since else None,
                    'until': until.isoformat() if until else None,
                    'paths': paths
                }
            }
            
            self.logger.info(f"Ingestion complete: {summary}")
            return summary
    
    def _process_commits(self, commits: List[CommitData]) -> List[ProcessedDocument]:
        """Process a batch of commits into documents."""
        documents = []
        
        for commit in commits:
            # Create documents for different parts of the commit
            
            # 1. Commit message document
            if commit.message.strip():
                doc_id = f"{commit.sha}_message"
                processed_doc = self.preprocessor.process_document(
                    content=commit.message,
                    document_id=doc_id,
                    document_type="commit",
                    additional_metadata={
                        'commit_sha': commit.sha,
                        'author': commit.author_name,
                        'author_email': commit.author_email,
                        'commit_date': commit.commit_date.isoformat(),
                        'files_changed': commit.files_changed[:10],  # First 10 files
                        'additions': commit.additions,
                        'deletions': commit.deletions
                    }
                )
                documents.append(processed_doc)
            
            # 2. Diff document (if available and not too large)
            if commit.diff and len(commit.diff) < 100000:  # Skip very large diffs
                doc_id = f"{commit.sha}_diff"
                processed_doc = self.preprocessor.process_document(
                    content=commit.diff,
                    document_id=doc_id,
                    document_type="diff",
                    additional_metadata={
                        'commit_sha': commit.sha,
                        'author': commit.author_name,
                        'commit_date': commit.commit_date.isoformat(),
                        'files_changed': commit.files_changed,
                        'additions': commit.additions,
                        'deletions': commit.deletions
                    }
                )
                documents.append(processed_doc)
        
        return documents
    
    def _chunk_documents(self, documents: List[ProcessedDocument]) -> List[Chunk]:
        """Chunk processed documents."""
        chunks = []
        
        for doc in documents:
            try:
                content = doc.preprocessed_content or doc.content
                
                if doc.document_type == "commit":
                    # Use text chunker for commit messages
                    doc_chunks = self.text_chunker.chunk(content, doc.id)
                
                elif doc.document_type == "diff":
                    # Use diff chunker for diffs
                    doc_chunks = self.diff_chunker.chunk(content, doc.id)
                
                elif doc.document_type == "code":
                    # Use code chunker for code files
                    doc_chunks = self.code_chunker.chunk(content, doc.id, language=doc.language)
                
                else:
                    # Default to text chunker
                    doc_chunks = self.text_chunker.chunk(content, doc.id)
                
                # Add document metadata to chunks
                for chunk in doc_chunks:
                    chunk.metadata.update(doc.metadata)
                    if doc.language:
                        chunk.language = doc.language
                
                chunks.extend(doc_chunks)
                
            except Exception as e:
                self.logger.error(f"Failed to chunk document {doc.id}: {e}")
                continue
        
        return chunks
    
    def _save_chunks(self, chunks: List[Chunk], output_file: Path) -> None:
        """Save chunks to JSON file."""
        serializable_chunks = []
        
        for chunk in chunks:
            chunk_dict = {
                'id': chunk.id,
                'content': chunk.content,
                'start_idx': chunk.start_idx,
                'end_idx': chunk.end_idx,
                'metadata': chunk.metadata,
                'chunk_type': chunk.chunk_type,
                'parent_id': chunk.parent_id,
                'language': chunk.language
            }
            serializable_chunks.append(chunk_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'chunks': serializable_chunks,
                'total_count': len(chunks),
                'created_at': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(chunks)} chunks to {output_file}")
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """Get statistics about ingested data."""
        stats = {
            'repository_stats': self.extractor.get_repository_stats(),
            'output_files': list(self.output_dir.glob("*.json")),
            'total_output_files': len(list(self.output_dir.glob("*.json")))
        }
        
        # Count total chunks across all files
        total_chunks = 0
        for file_path in self.output_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    total_chunks += data.get('total_count', 0)
            except Exception as e:
                self.logger.warning(f"Failed to read {file_path}: {e}")
        
        stats['total_chunks_ingested'] = total_chunks
        return stats


def main():
    """Main CLI entry point for data ingestion."""
    parser = argparse.ArgumentParser(description="Chromium RAG Data Ingestion Pipeline")
    parser.add_argument("--mode", choices=["recent", "range", "stats"], default="recent",
                       help="Ingestion mode")
    parser.add_argument("--days", type=int, default=30,
                       help="Number of recent days to process (recent mode)")
    parser.add_argument("--max-count", type=int,
                       help="Maximum number of commits to process")
    parser.add_argument("--since", type=str,
                       help="Start date (YYYY-MM-DD)")
    parser.add_argument("--until", type=str,
                       help="End date (YYYY-MM-DD)")
    parser.add_argument("--paths", nargs="+",
                       help="Specific file paths to filter")
    parser.add_argument("--batch-size", type=int, default=1000,
                       help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = DataIngestionPipeline()
    
    if args.mode == "recent":
        # Ingest recent commits
        summary = pipeline.ingest_recent_commits(
            days=args.days,
            batch_size=args.batch_size
        )
        print(f"Ingestion complete: {json.dumps(summary, indent=2)}")
    
    elif args.mode == "range":
        # Parse dates
        since = datetime.fromisoformat(args.since) if args.since else None
        until = datetime.fromisoformat(args.until) if args.until else None
        
        summary = pipeline.ingest_commit_range(
            max_count=args.max_count,
            since=since,
            until=until,
            paths=args.paths,
            batch_size=args.batch_size
        )
        print(f"Ingestion complete: {json.dumps(summary, indent=2)}")
    
    elif args.mode == "stats":
        # Show ingestion statistics
        stats = pipeline.get_ingestion_stats()
        print(f"Ingestion statistics: {json.dumps(stats, indent=2, default=str)}")


if __name__ == "__main__":
    main()