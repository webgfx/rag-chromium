"""
Core configuration management for the RAG system.
Handles environment variables, YAML configs, and runtime parameters.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv


class GPUConfig(BaseModel):
    """GPU and CUDA configuration."""
    device: str = Field(default="cuda", description="Primary compute device")
    visible_devices: str = Field(default="0", description="CUDA visible devices")
    memory_fraction: float = Field(default=0.8, description="GPU memory fraction to use")
    enable_mixed_precision: bool = Field(default=True, description="Enable automatic mixed precision")
    enable_compilation: bool = Field(default=True, description="Enable model compilation")


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = Field(default="BAAI/bge-m3", description="Embedding model name")
    device: str = Field(default="cuda", description="Device for embedding computation")
    batch_size: int = Field(default=64, description="Batch size for embedding generation")
    max_length: int = Field(default=8192, description="Maximum sequence length")
    pooling_method: str = Field(default="cls", description="Pooling method for embeddings")
    normalize: bool = Field(default=True, description="Normalize embeddings")


class LLMConfig(BaseModel):
    """Language model configuration."""
    model_name: str = Field(default="microsoft/DialoGPT-large", description="LLM model name")
    device: str = Field(default="cuda", description="Device for LLM inference")
    max_length: int = Field(default=4096, description="Maximum generation length")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling parameter")
    do_sample: bool = Field(default=True, description="Enable sampling")
    quantization: Optional[str] = Field(default=None, description="Quantization method")


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    backend: str = Field(default="chromadb", description="Vector database backend")
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=8000, description="Database port")
    collection_name: str = Field(default="chromium_rag", description="Collection name")
    embedding_dimension: int = Field(default=1024, description="Embedding dimension")
    similarity_metric: str = Field(default="cosine", description="Similarity metric")


class DataConfig(BaseModel):
    """Data processing configuration."""
    chromium_repo_path: str = Field(default=r"d:\r\cr\src", description="Chromium repository path")
    cache_dir: str = Field(default="./data/cache", description="Data cache directory")
    embeddings_dir: str = Field(default="./data/embeddings", description="Embeddings storage directory")
    max_chunk_size: int = Field(default=2048, description="Maximum chunk size for processing")
    chunk_overlap: int = Field(default=256, description="Overlap between chunks")
    supported_extensions: list = Field(
        default=[".py", ".cpp", ".cc", ".h", ".js", ".ts", ".java", ".md", ".txt"],
        description="Supported file extensions"
    )


class RetrievalConfig(BaseModel):
    """Retrieval system configuration."""
    top_k: int = Field(default=20, description="Number of documents to retrieve")
    rerank_k: int = Field(default=5, description="Number of documents to rerank")
    similarity_threshold: float = Field(default=0.75, description="Minimum similarity threshold")
    enable_hybrid_search: bool = Field(default=True, description="Enable hybrid search")
    bm25_weight: float = Field(default=0.3, description="BM25 weight in hybrid search")
    semantic_weight: float = Field(default=0.7, description="Semantic weight in hybrid search")


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of worker processes")
    reload: bool = Field(default=False, description="Enable auto-reload")


class UIConfig(BaseModel):
    """User interface configuration."""
    host: str = Field(default="0.0.0.0", description="UI host")
    port: int = Field(default=8501, description="UI port")
    theme: str = Field(default="dark", description="UI theme")


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""
    wandb_project: str = Field(default="chromium-rag", description="Weights & Biases project")
    wandb_entity: Optional[str] = Field(default=None, description="Weights & Biases entity")
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: str = Field(default="./logs", description="Log directory")


class Config(BaseModel):
    """Main configuration class."""
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

    def __init__(self, config_path: Optional[Union[str, Path]] = None, **kwargs):
        """Initialize configuration from environment variables, file, and kwargs."""
        # Load environment variables
        load_dotenv()
        
        # Start with environment variables
        config_data = self._load_from_env()
        
        # Override with config file if provided
        if config_path:
            file_config = self._load_from_file(config_path)
            config_data = self._merge_configs(config_data, file_config)
        
        # Override with kwargs
        if kwargs:
            config_data = self._merge_configs(config_data, kwargs)
        
        super().__init__(**config_data)

    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # GPU configuration
        if os.getenv("CUDA_VISIBLE_DEVICES"):
            config.setdefault("gpu", {})["visible_devices"] = os.getenv("CUDA_VISIBLE_DEVICES")
        
        # Embedding configuration
        if os.getenv("EMBEDDING_MODEL"):
            config.setdefault("embedding", {})["model_name"] = os.getenv("EMBEDDING_MODEL")
        if os.getenv("EMBEDDING_BATCH_SIZE"):
            config.setdefault("embedding", {})["batch_size"] = int(os.getenv("EMBEDDING_BATCH_SIZE"))
        
        # LLM configuration
        if os.getenv("LLM_MODEL"):
            config.setdefault("llm", {})["model_name"] = os.getenv("LLM_MODEL")
        if os.getenv("LLM_TEMPERATURE"):
            config.setdefault("llm", {})["temperature"] = float(os.getenv("LLM_TEMPERATURE"))
        
        # Vector DB configuration
        if os.getenv("VECTOR_DB"):
            config.setdefault("vector_db", {})["backend"] = os.getenv("VECTOR_DB")
        if os.getenv("COLLECTION_NAME"):
            config.setdefault("vector_db", {})["collection_name"] = os.getenv("COLLECTION_NAME")
        
        # Data configuration
        if os.getenv("CHROMIUM_REPO_PATH"):
            config.setdefault("data", {})["chromium_repo_path"] = os.getenv("CHROMIUM_REPO_PATH")
        
        # API configuration
        if os.getenv("API_HOST"):
            config.setdefault("api", {})["host"] = os.getenv("API_HOST")
        if os.getenv("API_PORT"):
            config.setdefault("api", {})["port"] = int(os.getenv("API_PORT"))
        
        return config

    def _load_from_file(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def save(self, config_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)

    def get_device(self) -> str:
        """Get the primary compute device."""
        return self.gpu.device if self.gpu.device != "auto" else "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu"

    def setup_gpu(self) -> None:
        """Set up GPU environment variables."""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu.visible_devices
        
        # Set memory fraction for PyTorch
        import torch
        if torch.cuda.is_available() and self.get_device() == "cuda":
            torch.cuda.set_per_process_memory_fraction(self.gpu.memory_fraction)


# Global configuration instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config