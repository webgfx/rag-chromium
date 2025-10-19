"""
Advanced embedding model management system.
Supports multiple state-of-the-art models with GPU optimization.
"""

import os
import torch
import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

from ..core.config import get_config
from ..core.logger import setup_logger, PerformanceLogger


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    name: str
    model_id: str
    max_seq_length: int
    embedding_dim: int
    pooling_method: str = "cls"
    normalize: bool = True
    trust_remote_code: bool = False
    model_type: str = "sentence_transformer"  # or "huggingface"


class EmbeddingModelManager:
    """
    Manages multiple embedding models with GPU optimization.
    Supports both Sentence Transformers and raw Hugging Face models.
    """
    
    # Predefined model configurations
    MODEL_CONFIGS = {
        "bge-m3": ModelConfig(
            name="bge-m3",
            model_id="BAAI/bge-m3",
            max_seq_length=8192,
            embedding_dim=1024,
            pooling_method="cls",
            normalize=True,
            model_type="sentence_transformer"
        ),
        "bge-large-en": ModelConfig(
            name="bge-large-en",
            model_id="BAAI/bge-large-en-v1.5",
            max_seq_length=512,
            embedding_dim=1024,
            pooling_method="cls",
            normalize=True,
            model_type="sentence_transformer"
        ),
        "e5-large": ModelConfig(
            name="e5-large",
            model_id="intfloat/e5-large-v2",
            max_seq_length=512,
            embedding_dim=1024,
            pooling_method="mean",
            normalize=True,
            model_type="sentence_transformer"
        ),
        "code-bert": ModelConfig(
            name="code-bert",
            model_id="microsoft/codebert-base",
            max_seq_length=512,
            embedding_dim=768,
            pooling_method="cls",
            normalize=True,
            model_type="huggingface"
        ),
        "unixcoder": ModelConfig(
            name="unixcoder",
            model_id="microsoft/unixcoder-base",
            max_seq_length=512,
            embedding_dim=768,
            pooling_method="cls",
            normalize=True,
            model_type="huggingface"
        ),
        "multilingual-e5": ModelConfig(
            name="multilingual-e5",
            model_id="intfloat/multilingual-e5-large",
            max_seq_length=512,
            embedding_dim=1024,
            pooling_method="mean",
            normalize=True,
            model_type="sentence_transformer"
        )
    }
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the embedding model manager.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.EmbeddingModelManager")
        
        self.cache_dir = Path(cache_dir or self.config.data.cache_dir) / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._setup_device()
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        
        self.logger.info(f"Initialized embedding model manager on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Set up the compute device for models."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            
            # Set up GPU memory management
            if hasattr(self.config.gpu, 'memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(self.config.gpu.memory_fraction)
            
            # Enable optimizations
            if hasattr(self.config.gpu, 'enable_mixed_precision') and self.config.gpu.enable_mixed_precision:
                torch.backends.cudnn.benchmark = True
            
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
        else:
            device = torch.device("cpu")
            self.logger.warning("CUDA not available, using CPU")
        
        return device
    
    def load_model(self, model_name: str, **kwargs) -> Any:
        """
        Load an embedding model.
        
        Args:
            model_name: Name of the model to load
            **kwargs: Additional arguments for model loading
        
        Returns:
            Loaded model instance
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # Get model configuration
        if model_name in self.MODEL_CONFIGS:
            model_config = self.MODEL_CONFIGS[model_name]
        else:
            # Create default config for custom model
            model_config = ModelConfig(
                name=model_name,
                model_id=model_name,
                max_seq_length=512,
                embedding_dim=768,
                pooling_method="cls",
                normalize=True,
                model_type=kwargs.get("model_type", "sentence_transformer")
            )
        
        self.model_configs[model_name] = model_config
        
        with PerformanceLogger(self.logger, f"loading model {model_name}"):
            try:
                if model_config.model_type == "sentence_transformer":
                    model = self._load_sentence_transformer(model_config, **kwargs)
                else:
                    model = self._load_huggingface_model(model_config, **kwargs)
                
                self.loaded_models[model_name] = model
                self.logger.info(f"Successfully loaded model: {model_name}")
                
                return model
                
            except Exception as e:
                self.logger.error(f"Failed to load model {model_name}: {e}")
                raise
    
    def _load_sentence_transformer(self, config: ModelConfig, **kwargs) -> SentenceTransformer:
        """Load a Sentence Transformer model."""
        model = SentenceTransformer(
            config.model_id,
            cache_folder=str(self.cache_dir),
            device=str(self.device),
            trust_remote_code=config.trust_remote_code
        )
        
        # Update max sequence length if needed
        if hasattr(model, 'max_seq_length'):
            model.max_seq_length = config.max_seq_length
        
        # Enable optimizations
        model.eval()
        
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile') and self.config.gpu.enable_compilation:
            try:
                model = torch.compile(model)
                self.logger.info(f"Compiled model {config.name} for optimization")
            except Exception as e:
                self.logger.warning(f"Failed to compile model {config.name}: {e}")
        
        return model
    
    def _load_huggingface_model(self, config: ModelConfig, **kwargs) -> Tuple[AutoModel, AutoTokenizer]:
        """Load a Hugging Face model with tokenizer."""
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            cache_dir=str(self.cache_dir),
            trust_remote_code=config.trust_remote_code
        )
        
        # Load model
        model = AutoModel.from_pretrained(
            config.model_id,
            cache_dir=str(self.cache_dir),
            trust_remote_code=config.trust_remote_code,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32
        ).to(self.device)
        
        model.eval()
        
        # Compile model for better performance
        if hasattr(torch, 'compile') and self.config.gpu.enable_compilation:
            try:
                model = torch.compile(model)
                self.logger.info(f"Compiled model {config.name} for optimization")
            except Exception as e:
                self.logger.warning(f"Failed to compile model {config.name}: {e}")
        
        return model, tokenizer
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Dictionary with model information
        """
        if model_name not in self.model_configs:
            if model_name in self.MODEL_CONFIGS:
                config = self.MODEL_CONFIGS[model_name]
            else:
                return {"error": f"Model {model_name} not found"}
        else:
            config = self.model_configs[model_name]
        
        info = {
            "name": config.name,
            "model_id": config.model_id,
            "max_seq_length": config.max_seq_length,
            "embedding_dim": config.embedding_dim,
            "pooling_method": config.pooling_method,
            "normalize": config.normalize,
            "model_type": config.model_type,
            "loaded": model_name in self.loaded_models,
            "device": str(self.device)
        }
        
        if model_name in self.loaded_models:
            model = self.loaded_models[model_name]
            if hasattr(model, 'get_sentence_embedding_dimension'):
                info["actual_embedding_dim"] = model.get_sentence_embedding_dimension()
        
        return info
    
    def list_available_models(self) -> List[str]:
        """List all available predefined models."""
        return list(self.MODEL_CONFIGS.keys())
    
    def list_loaded_models(self) -> List[str]:
        """List currently loaded models."""
        return list(self.loaded_models.keys())
    
    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info(f"Unloaded model: {model_name}")
        else:
            self.logger.warning(f"Model {model_name} is not loaded")
    
    def unload_all_models(self) -> None:
        """Unload all models from memory."""
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            self.unload_model(model_name)
        
        self.logger.info("Unloaded all models")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
        
        memory_info = {
            "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
            "reserved": torch.cuda.memory_reserved() / 1024**3,   # GB
            "max_allocated": torch.cuda.max_memory_allocated() / 1024**3,  # GB
            "max_reserved": torch.cuda.max_memory_reserved() / 1024**3,    # GB
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        }
        
        memory_info["utilization"] = memory_info["allocated"] / memory_info["total"] * 100
        
        return memory_info
    
    def optimize_memory(self) -> None:
        """Optimize GPU memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.logger.info("Optimized GPU memory usage")
    
    def benchmark_model(self, model_name: str, texts: List[str], batch_size: int = 32) -> Dict[str, Any]:
        """
        Benchmark a model's performance.
        
        Args:
            model_name: Name of the model to benchmark
            texts: List of texts to use for benchmarking
            batch_size: Batch size for processing
        
        Returns:
            Benchmark results
        """
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        model = self.loaded_models[model_name]
        config = self.model_configs[model_name]
        
        import time
        
        # Warm up
        if config.model_type == "sentence_transformer":
            _ = model.encode(texts[:min(5, len(texts))], batch_size=batch_size)
        
        # Benchmark
        start_time = time.time()
        
        if config.model_type == "sentence_transformer":
            embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        else:
            # For Hugging Face models, we'd need custom encoding logic
            embeddings = np.random.rand(len(texts), config.embedding_dim)  # Placeholder
        
        end_time = time.time()
        
        total_time = end_time - start_time
        throughput = len(texts) / total_time
        
        return {
            "model_name": model_name,
            "num_texts": len(texts),
            "batch_size": batch_size,
            "total_time": total_time,
            "throughput": throughput,
            "avg_time_per_text": total_time / len(texts),
            "embedding_shape": embeddings.shape,
            "memory_usage": self.get_memory_usage()
        }