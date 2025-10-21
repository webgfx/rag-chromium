"""
Advanced embedding generation with batch processing and GPU optimization.
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Iterator, Tuple
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import json

from .models import EmbeddingModelManager, ModelConfig
from ..core.config import get_config
from ..core.logger import setup_logger, PerformanceLogger
from ..data.chunker import Chunk


class EmbeddingGenerator:
    """
    High-performance embedding generator with GPU acceleration.
    Supports batch processing, caching, and multiple embedding models.
    """
    
    def __init__(
        self,
        model_name: str = "bge-m3",
        batch_size: Optional[int] = None,
        cache_embeddings: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the embedding model to use
            batch_size: Batch size for processing (auto-detected if None)
            cache_embeddings: Whether to cache generated embeddings
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.EmbeddingGenerator")
        
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        
        # Initialize model manager
        self.model_manager = EmbeddingModelManager()
        self.model = None
        self.model_config = None
        
        # Set batch size
        self.batch_size = batch_size or self._auto_detect_batch_size()
        
        # Query embedding cache for fast repeated queries
        self.query_cache = {}  # Separate cache for query embeddings
        self.max_query_cache_size = 1000  # Keep last 1000 queries
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_file = Path(self.config.data.embeddings_dir) / f"{model_name}_embeddings.json"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        if self.cache_embeddings and self.cache_file.exists():
            self._load_cache()
    
    def _auto_detect_batch_size(self) -> int:
        """Auto-detect optimal batch size based on available GPU memory."""
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            # Aggressive batch sizes for high-end GPUs (RTX 5080 has 16GB)
            if gpu_memory >= 24:  # RTX 4090, RTX 5080, etc.
                return 256  # 2x larger for better GPU utilization
            elif gpu_memory >= 16:  # RTX 4080, RTX 5080 16GB, etc.
                return 192  # 2x larger
            elif gpu_memory >= 12:  # RTX 4070 Ti, etc.
                return 128
            elif gpu_memory >= 8:   # RTX 4060 Ti, etc.
                return 64
            else:
                return 32
        else:
            return 8  # Conservative for CPU
    
    def _load_model(self) -> None:
        """Load the embedding model if not already loaded."""
        if self.model is None:
            self.model = self.model_manager.load_model(self.model_name)
            self.model_config = self.model_manager.model_configs[self.model_name]
            
            # Enable FP16 inference for 2x speedup on RTX GPUs
            if torch.cuda.is_available() and hasattr(self.model, 'half'):
                try:
                    # For sentence transformers, convert the underlying model
                    if hasattr(self.model, '_first_module'):
                        self.model = self.model.half()
                        self.logger.info(f"Enabled FP16 inference for {self.model_name}")
                except Exception as e:
                    self.logger.warning(f"Could not enable FP16: {e}")
            
            self.logger.info(f"Loaded embedding model: {self.model_name}")
    
    def _load_cache(self) -> None:
        """Load embeddings cache from disk."""
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                self.embedding_cache = cache_data.get('embeddings', {})
            
            self.logger.info(f"Loaded {len(self.embedding_cache)} cached embeddings")
        except Exception as e:
            self.logger.warning(f"Failed to load embedding cache: {e}")
            self.embedding_cache = {}
    
    def _save_cache(self) -> None:
        """Save embeddings cache to disk."""
        if not self.cache_embeddings:
            return
        
        try:
            cache_data = {
                'model_name': self.model_name,
                'created_at': time.time(),
                'embeddings': self.embedding_cache
            }
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f)
            
            self.logger.debug(f"Saved {len(self.embedding_cache)} embeddings to cache")
        except Exception as e:
            self.logger.warning(f"Failed to save embedding cache: {e}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def encode_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        normalize: bool = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings with caching.
        
        Args:
            texts: List of texts to encode
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings (uses model default if None)
            use_cache: Whether to use query cache (for repeated queries)
        
        Returns:
            Array of embeddings
        """
        if not texts:
            return np.array([])
        
        self._load_model()
        
        # Check query cache first for single-text queries (typical for search)
        if use_cache and len(texts) == 1:
            cache_key = self._get_cache_key(texts[0])
            if cache_key in self.query_cache:
                self.logger.debug("Query cache hit")
                return np.array([self.query_cache[cache_key]])
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        uncached_texts = []
        uncached_indices = []
        
        if self.cache_embeddings:
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self.embedding_cache:
                    cached_embeddings[i] = np.array(self.embedding_cache[cache_key])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        self.logger.info(f"Found {len(cached_embeddings)} cached embeddings, "
                        f"generating {len(uncached_texts)} new embeddings")
        
        # Generate embeddings for uncached texts
        new_embeddings = {}
        if uncached_texts:
            with PerformanceLogger(self.logger, f"embedding generation for {len(uncached_texts)} texts"):
                if self.model_config.model_type == "sentence_transformer":
                    embeddings = self._encode_sentence_transformer(uncached_texts, show_progress, normalize)
                else:
                    embeddings = self._encode_huggingface(uncached_texts, show_progress, normalize)
                
                # Cache new embeddings
                if self.cache_embeddings:
                    for i, (text, embedding) in enumerate(zip(uncached_texts, embeddings)):
                        cache_key = self._get_cache_key(text)
                        self.embedding_cache[cache_key] = embedding.tolist()
                        new_embeddings[uncached_indices[i]] = embedding
                    
                    # Save cache periodically
                    if len(self.embedding_cache) % 1000 == 0:
                        self._save_cache()
                else:
                    for i, embedding in enumerate(embeddings):
                        new_embeddings[uncached_indices[i]] = embedding
        
        # Combine cached and new embeddings in original order
        all_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                all_embeddings.append(cached_embeddings[i])
            else:
                all_embeddings.append(new_embeddings[i])
        
        # Save cache at the end
        if self.cache_embeddings and new_embeddings:
            self._save_cache()
        
        result = np.array(all_embeddings)
        
        # Cache single query for fast repeated access
        if use_cache and len(texts) == 1:
            cache_key = self._get_cache_key(texts[0])
            self.query_cache[cache_key] = result[0]
            
            # Evict oldest if cache is full
            if len(self.query_cache) > self.max_query_cache_size:
                # Remove 10% oldest entries
                keys_to_remove = list(self.query_cache.keys())[:100]
                for key in keys_to_remove:
                    del self.query_cache[key]
        
        return result
    
    def _encode_sentence_transformer(
        self,
        texts: List[str],
        show_progress: bool,
        normalize: bool = None
    ) -> np.ndarray:
        """Encode texts using Sentence Transformer model."""
        normalize = normalize if normalize is not None else self.model_config.normalize
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def _encode_huggingface(
        self,
        texts: List[str],
        show_progress: bool,
        normalize: bool = None
    ) -> np.ndarray:
        """Encode texts using Hugging Face model."""
        model, tokenizer = self.model
        normalize = normalize if normalize is not None else self.model_config.normalize
        
        embeddings = []
        
        # Process in batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        
        if show_progress:
            batches = tqdm(batches, desc="Generating embeddings")
        
        with torch.no_grad():
            for batch in batches:
                # Tokenize
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.model_config.max_seq_length
                ).to(model.device)
                
                # Get model outputs
                outputs = model(**inputs)
                
                # Apply pooling
                if self.model_config.pooling_method == "cls":
                    batch_embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
                elif self.model_config.pooling_method == "mean":
                    # Mean pooling with attention mask
                    attention_mask = inputs["attention_mask"]
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                elif self.model_config.pooling_method == "max":
                    # Max pooling
                    token_embeddings = outputs.last_hidden_state
                    attention_mask = inputs["attention_mask"]
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
                    batch_embeddings = torch.max(token_embeddings, 1)[0]
                else:
                    raise ValueError(f"Unknown pooling method: {self.model_config.pooling_method}")
                
                # Normalize if requested
                if normalize:
                    batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_chunks(
        self,
        chunks: List[Chunk],
        content_field: str = "content",
        show_progress: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Encode a list of chunks into embeddings.
        
        Args:
            chunks: List of Chunk objects
            content_field: Field to use for embedding content
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (embeddings array, chunk IDs)
        """
        if not chunks:
            return np.array([]), []
        
        # Extract texts and IDs
        texts = []
        chunk_ids = []
        
        for chunk in chunks:
            if content_field == "content":
                text = chunk.content
            elif content_field == "preprocessed_content" and hasattr(chunk, 'preprocessed_content'):
                text = chunk.preprocessed_content or chunk.content
            else:
                text = getattr(chunk, content_field, chunk.content)
            
            texts.append(text)
            chunk_ids.append(chunk.id)
        
        # Generate embeddings
        embeddings = self.encode_texts(texts, show_progress=show_progress)
        
        return embeddings, chunk_ids
    
    def encode_chunks_with_metadata(
        self,
        chunks: List[Chunk],
        content_field: str = "content",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Encode chunks and return with metadata.
        
        Args:
            chunks: List of Chunk objects
            content_field: Field to use for embedding content
            show_progress: Whether to show progress bar
        
        Returns:
            List of dictionaries with embeddings and metadata
        """
        embeddings, chunk_ids = self.encode_chunks(chunks, content_field, show_progress)
        
        results = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            result = {
                'chunk_id': chunk.id,
                'embedding': embedding.tolist(),
                'content': getattr(chunk, content_field, chunk.content),
                'metadata': chunk.metadata,
                'chunk_type': chunk.chunk_type,
                'parent_id': chunk.parent_id,
                'language': chunk.language,
                'embedding_model': self.model_name,
                'embedding_dim': len(embedding)
            }
            results.append(result)
        
        return results
    
    def save_embeddings(
        self,
        embeddings_data: List[Dict[str, Any]],
        output_file: Path
    ) -> None:
        """
        Save embeddings to file.
        
        Args:
            embeddings_data: List of embedding dictionaries
            output_file: Output file path
        """
        output_data = {
            'model_name': self.model_name,
            'model_config': {
                'embedding_dim': self.model_config.embedding_dim,
                'max_seq_length': self.model_config.max_seq_length,
                'pooling_method': self.model_config.pooling_method,
                'normalize': self.model_config.normalize
            },
            'created_at': time.time(),
            'total_embeddings': len(embeddings_data),
            'embeddings': embeddings_data
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        
        self.logger.info(f"Saved {len(embeddings_data)} embeddings to {output_file}")
    
    def benchmark(self, sample_texts: List[str], batch_sizes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark the embedding generator performance.
        
        Args:
            sample_texts: Sample texts for benchmarking
            batch_sizes: List of batch sizes to test
        
        Returns:
            Benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128]
        
        self._load_model()
        
        results = {
            'model_name': self.model_name,
            'gpu_memory': self.model_manager.get_memory_usage(),
            'batch_size_results': []
        }
        
        for batch_size in batch_sizes:
            if batch_size > len(sample_texts):
                continue
            
            original_batch_size = self.batch_size
            self.batch_size = batch_size
            
            try:
                start_time = time.time()
                embeddings = self.encode_texts(sample_texts[:batch_size * 5], show_progress=False)
                end_time = time.time()
                
                duration = end_time - start_time
                throughput = len(embeddings) / duration
                
                results['batch_size_results'].append({
                    'batch_size': batch_size,
                    'duration': duration,
                    'throughput': throughput,
                    'samples_processed': len(embeddings)
                })
                
            except Exception as e:
                self.logger.warning(f"Benchmark failed for batch size {batch_size}: {e}")
            
            finally:
                self.batch_size = original_batch_size
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding generator."""
        stats = {
            'model_name': self.model_name,
            'batch_size': self.batch_size,
            'cache_enabled': self.cache_embeddings,
            'cached_embeddings': len(self.embedding_cache),
            'model_loaded': self.model is not None
        }
        
        if self.model is not None:
            stats.update(self.model_manager.get_model_info(self.model_name))
            stats['memory_usage'] = self.model_manager.get_memory_usage()
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        if self.cache_file.exists():
            self.cache_file.unlink()
        self.logger.info("Cleared embedding cache")
    
    def __del__(self):
        """Cleanup when the generator is destroyed."""
        if self.cache_embeddings and self.embedding_cache:
            self._save_cache()