"""
Advanced embedding cache system with persistence and optimization.
"""

import os
import sys
import sqlite3
import pickle
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import threading
from contextlib import contextmanager

from ..core.config import get_config
from ..core.logger import setup_logger


@dataclass
class CacheEntry:
    """Represents a cached embedding entry."""
    key: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    created_at: float
    last_accessed: float
    access_count: int


class EmbeddingCache:
    """
    High-performance embedding cache with SQLite backend.
    Supports persistence, LRU eviction, and efficient lookup.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_size: int = 100000,
        max_memory_mb: int = 1024
    ):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory for cache files
            max_size: Maximum number of cached embeddings
            max_memory_mb: Maximum memory usage in MB
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.EmbeddingCache")
        
        self.cache_dir = Path(cache_dir or self.config.data.cache_dir) / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        
        # Database file
        self.db_file = self.cache_dir / "embeddings.db"
        
        # In-memory cache for frequently accessed embeddings
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.current_memory_usage = 0
        
        # Thread lock for concurrent access
        self.lock = threading.RLock()
        
        # Initialize database
        self._init_database()
        
        # Load frequently accessed embeddings into memory
        self._load_hot_cache()
        
        self.logger.info(f"Initialized embedding cache with max_size={max_size}, "
                        f"max_memory={max_memory_mb}MB")
    
    def _init_database(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(str(self.db_file)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    metadata TEXT,
                    model_name TEXT,
                    embedding_dim INTEGER,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Create indices for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_last_accessed ON embeddings(last_accessed)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_access_count ON embeddings(access_count)")
            
            conn.commit()
    
    def _load_hot_cache(self) -> None:
        """Load frequently accessed embeddings into memory."""
        with sqlite3.connect(str(self.db_file)) as conn:
            # Load most frequently accessed embeddings
            cursor = conn.execute("""
                SELECT key, embedding, metadata, created_at, last_accessed, access_count
                FROM embeddings
                ORDER BY access_count DESC, last_accessed DESC
                LIMIT ?
            """, (min(1000, self.max_size // 10),))
            
            loaded_count = 0
            for row in cursor:
                key, embedding_blob, metadata_json, created_at, last_accessed, access_count = row
                
                # Check memory limit
                embedding_size = len(embedding_blob)
                if self.current_memory_usage + embedding_size > self.max_memory_bytes:
                    break
                
                # Deserialize
                embedding = pickle.loads(embedding_blob)
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Add to memory cache
                entry = CacheEntry(
                    key=key,
                    embedding=embedding,
                    metadata=metadata,
                    created_at=created_at,
                    last_accessed=last_accessed,
                    access_count=access_count
                )
                
                self.memory_cache[key] = entry
                self.current_memory_usage += embedding_size
                loaded_count += 1
            
            self.logger.info(f"Loaded {loaded_count} embeddings into hot cache "
                           f"({self.current_memory_usage / 1024 / 1024:.1f} MB)")
    
    def _make_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model name."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def put(
        self,
        text: str,
        embedding: np.ndarray,
        model_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store an embedding in the cache.
        
        Args:
            text: Original text
            embedding: Embedding vector
            model_name: Name of the model that generated the embedding
            metadata: Additional metadata
        """
        key = self._make_key(text, model_name)
        current_time = time.time()
        
        metadata = metadata or {}
        metadata.update({
            'text_length': len(text),
            'model_name': model_name
        })
        
        with self.lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                embedding=embedding,
                metadata=metadata,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1
            )
            
            # Store in database
            embedding_blob = pickle.dumps(embedding)
            metadata_json = json.dumps(metadata)
            
            with sqlite3.connect(str(self.db_file)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO embeddings 
                    (key, embedding, metadata, model_name, embedding_dim, created_at, last_accessed, access_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, embedding_blob, metadata_json, model_name,
                    len(embedding), current_time, current_time, 1
                ))
                conn.commit()
            
            # Add to memory cache if there's space
            embedding_size = len(embedding_blob)
            if self.current_memory_usage + embedding_size <= self.max_memory_bytes:
                self.memory_cache[key] = entry
                self.current_memory_usage += embedding_size
            else:
                # Evict least recently used items from memory cache
                self._evict_memory_cache(embedding_size)
                self.memory_cache[key] = entry
                self.current_memory_usage += embedding_size
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Retrieve an embedding from the cache.
        
        Args:
            text: Original text
            model_name: Name of the model
        
        Returns:
            Cached embedding or None if not found
        """
        key = self._make_key(text, model_name)
        
        with self.lock:
            # Check memory cache first
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                entry.last_accessed = time.time()
                entry.access_count += 1
                return entry.embedding.copy()
            
            # Check database
            with sqlite3.connect(str(self.db_file)) as conn:
                cursor = conn.execute("""
                    SELECT embedding, metadata, created_at, access_count
                    FROM embeddings WHERE key = ?
                """, (key,))
                
                row = cursor.fetchone()
                if row is None:
                    return None
                
                embedding_blob, metadata_json, created_at, access_count = row
                
                # Update access statistics
                current_time = time.time()
                conn.execute("""
                    UPDATE embeddings 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE key = ?
                """, (current_time, key))
                conn.commit()
                
                # Deserialize embedding
                embedding = pickle.loads(embedding_blob)
                metadata = json.loads(metadata_json) if metadata_json else {}
                
                # Add to memory cache if there's space and it's frequently accessed
                if access_count >= 2:  # Only cache frequently accessed items
                    embedding_size = len(embedding_blob)
                    if self.current_memory_usage + embedding_size <= self.max_memory_bytes:
                        entry = CacheEntry(
                            key=key,
                            embedding=embedding,
                            metadata=metadata,
                            created_at=created_at,
                            last_accessed=current_time,
                            access_count=access_count + 1
                        )
                        self.memory_cache[key] = entry
                        self.current_memory_usage += embedding_size
                
                return embedding
    
    def get_batch(
        self,
        texts: List[str],
        model_name: str
    ) -> Tuple[List[Optional[np.ndarray]], List[int]]:
        """
        Retrieve multiple embeddings efficiently.
        
        Args:
            texts: List of texts
            model_name: Name of the model
        
        Returns:
            Tuple of (embeddings list, indices of missing embeddings)
        """
        embeddings = [None] * len(texts)
        missing_indices = []
        
        keys = [self._make_key(text, model_name) for text in texts]
        
        with self.lock:
            # Check memory cache
            memory_hits = 0
            for i, key in enumerate(keys):
                if key in self.memory_cache:
                    entry = self.memory_cache[key]
                    entry.last_accessed = time.time()
                    entry.access_count += 1
                    embeddings[i] = entry.embedding.copy()
                    memory_hits += 1
                else:
                    missing_indices.append(i)
            
            # Check database for missing embeddings
            if missing_indices:
                missing_keys = [keys[i] for i in missing_indices]
                placeholders = ','.join(['?'] * len(missing_keys))
                
                with sqlite3.connect(str(self.db_file)) as conn:
                    cursor = conn.execute(f"""
                        SELECT key, embedding, metadata, created_at, access_count
                        FROM embeddings WHERE key IN ({placeholders})
                    """, missing_keys)
                    
                    db_hits = {}
                    current_time = time.time()
                    
                    for row in cursor:
                        key, embedding_blob, metadata_json, created_at, access_count = row
                        embedding = pickle.loads(embedding_blob)
                        metadata = json.loads(metadata_json) if metadata_json else {}
                        
                        db_hits[key] = (embedding, metadata, created_at, access_count)
                    
                    # Update access statistics for found embeddings
                    if db_hits:
                        update_keys = list(db_hits.keys())
                        placeholders = ','.join(['?'] * len(update_keys))
                        conn.execute(f"""
                            UPDATE embeddings 
                            SET last_accessed = ?, access_count = access_count + 1
                            WHERE key IN ({placeholders})
                        """, [current_time] + update_keys)
                        conn.commit()
                    
                    # Fill in found embeddings and update missing indices
                    still_missing = []
                    for idx in missing_indices:
                        key = keys[idx]
                        if key in db_hits:
                            embedding, metadata, created_at, access_count = db_hits[key]
                            embeddings[idx] = embedding
                            
                            # Add to memory cache if frequently accessed
                            if access_count >= 2:
                                embedding_size = sys.getsizeof(embedding)
                                if self.current_memory_usage + embedding_size <= self.max_memory_bytes:
                                    entry = CacheEntry(
                                        key=key,
                                        embedding=embedding,
                                        metadata=metadata,
                                        created_at=created_at,
                                        last_accessed=current_time,
                                        access_count=access_count + 1
                                    )
                                    self.memory_cache[key] = entry
                                    self.current_memory_usage += embedding_size
                        else:
                            still_missing.append(idx)
                    
                    missing_indices = still_missing
            
            self.logger.debug(f"Cache batch lookup: {memory_hits} memory hits, "
                            f"{len(texts) - len(missing_indices) - memory_hits} DB hits, "
                            f"{len(missing_indices)} misses")
        
        return embeddings, missing_indices
    
    def _evict_memory_cache(self, space_needed: int) -> None:
        """Evict least recently used items from memory cache."""
        if not self.memory_cache:
            return
        
        # Sort by last accessed time (LRU)
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        space_freed = 0
        for key, entry in sorted_entries:
            if space_freed >= space_needed:
                break
            
            embedding_size = entry.embedding.nbytes
            del self.memory_cache[key]
            self.current_memory_usage -= embedding_size
            space_freed += embedding_size
        
        self.logger.debug(f"Evicted {space_freed} bytes from memory cache")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with sqlite3.connect(str(self.db_file)) as conn:
            # Total embeddings in database
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            total_db_count = cursor.fetchone()[0]
            
            # Database size
            db_size = os.path.getsize(self.db_file) if self.db_file.exists() else 0
            
            # Model distribution
            cursor = conn.execute("""
                SELECT model_name, COUNT(*) 
                FROM embeddings 
                GROUP BY model_name
            """)
            model_counts = dict(cursor.fetchall())
        
        return {
            'total_embeddings': total_db_count,
            'memory_cache_size': len(self.memory_cache),
            'memory_usage_mb': self.current_memory_usage / 1024 / 1024,
            'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
            'database_size_mb': db_size / 1024 / 1024,
            'model_distribution': model_counts,
            'cache_dir': str(self.cache_dir)
        }
    
    def clear(self, model_name: Optional[str] = None) -> int:
        """
        Clear cache entries.
        
        Args:
            model_name: If specified, only clear entries for this model
        
        Returns:
            Number of entries cleared
        """
        with self.lock:
            if model_name:
                # Clear specific model
                with sqlite3.connect(str(self.db_file)) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM embeddings WHERE model_name = ?", (model_name,))
                    count = cursor.fetchone()[0]
                    
                    conn.execute("DELETE FROM embeddings WHERE model_name = ?", (model_name,))
                    conn.commit()
                
                # Clear from memory cache
                keys_to_remove = [
                    key for key, entry in self.memory_cache.items()
                    if entry.metadata.get('model_name') == model_name
                ]
                for key in keys_to_remove:
                    entry = self.memory_cache.pop(key)
                    self.current_memory_usage -= entry.embedding.nbytes
            
            else:
                # Clear all
                with sqlite3.connect(str(self.db_file)) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                    count = cursor.fetchone()[0]
                    
                    conn.execute("DELETE FROM embeddings")
                    conn.commit()
                
                self.memory_cache.clear()
                self.current_memory_usage = 0
        
        self.logger.info(f"Cleared {count} cache entries" + (f" for model {model_name}" if model_name else ""))
        return count
    
    def optimize(self) -> None:
        """Optimize the cache by removing old or infrequently accessed entries."""
        with self.lock:
            current_time = time.time()
            one_week_ago = current_time - (7 * 24 * 60 * 60)  # 1 week
            
            with sqlite3.connect(str(self.db_file)) as conn:
                # Remove entries older than 1 week with low access count
                cursor = conn.execute("""
                    DELETE FROM embeddings 
                    WHERE created_at < ? AND access_count <= 2
                """, (one_week_ago,))
                
                removed_count = cursor.rowcount
                
                # Vacuum the database to reclaim space
                conn.execute("VACUUM")
                conn.commit()
            
            # Reload hot cache
            self.memory_cache.clear()
            self.current_memory_usage = 0
            self._load_hot_cache()
            
            self.logger.info(f"Cache optimization complete: removed {removed_count} old entries")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass