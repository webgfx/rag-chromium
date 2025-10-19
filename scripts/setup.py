"""
Setup and initialization script for the Chromium RAG system.
Performs initial configuration, model downloads, and system validation.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any
import torch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag_system.core.config import Config, get_config
from rag_system.core.logger import setup_logger
from rag_system.embeddings.models import EmbeddingModelManager
from rag_system.embeddings.generator import EmbeddingGenerator
from rag_system.data.chromium import ChromiumDataExtractor


def check_gpu_availability():
    """Check GPU availability and configuration."""
    logger = setup_logger("setup.gpu_check")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1024**3
            logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f} GB)")
        
        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            logger.info("GPU memory allocation test: PASSED")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"GPU memory allocation test: FAILED - {e}")
            return False
        
        return True
    else:
        logger.warning("CUDA not available - will use CPU (much slower)")
        return False


def check_chromium_repository(repo_path: str):
    """Check if Chromium repository exists and is accessible."""
    logger = setup_logger("setup.repo_check")
    
    try:
        extractor = ChromiumDataExtractor(repo_path)
        stats = extractor.get_repository_stats()
        
        logger.info(f"Chromium repository check: PASSED")
        logger.info(f"Repository stats: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Chromium repository check: FAILED - {e}")
        logger.error(f"Please ensure the Chromium repository exists at: {repo_path}")
        return False


def download_embedding_models():
    """Download and test embedding models."""
    logger = setup_logger("setup.models")
    
    model_manager = EmbeddingModelManager()
    
    # Test models to download
    models_to_test = ["bge-m3", "bge-large-en"]
    
    for model_name in models_to_test:
        try:
            logger.info(f"Loading model: {model_name}")
            model = model_manager.load_model(model_name)
            
            # Test encoding with sample text
            if model_name in model_manager.MODEL_CONFIGS:
                config = model_manager.MODEL_CONFIGS[model_name]
                if config.model_type == "sentence_transformer":
                    test_embeddings = model.encode(["Test text for embedding"])
                    logger.info(f"Model {model_name}: LOADED (embedding dim: {test_embeddings.shape[1]})")
                else:
                    logger.info(f"Model {model_name}: LOADED (Hugging Face model)")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    return True


def test_embedding_generation():
    """Test the embedding generation pipeline."""
    logger = setup_logger("setup.embedding_test")
    
    try:
        # Create embedding generator
        generator = EmbeddingGenerator(model_name="bge-m3", batch_size=4)
        
        # Test with sample texts
        sample_texts = [
            "This is a commit message for fixing a bug in the renderer",
            "Added new feature for WebGL context handling",
            "def process_chromium_data(data): return data.process()",
            "Fixed memory leak in GPU process initialization"
        ]
        
        logger.info("Testing embedding generation...")
        embeddings = generator.encode_texts(sample_texts, show_progress=True)
        
        logger.info(f"Embedding generation test: PASSED")
        logger.info(f"Generated {len(embeddings)} embeddings with shape {embeddings.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"Embedding generation test: FAILED - {e}")
        return False


def setup_directories():
    """Create necessary directories."""
    logger = setup_logger("setup.directories")
    
    config = get_config()
    
    directories = [
        config.data.cache_dir,
        config.data.embeddings_dir,
        config.monitoring.log_dir,
        "data/processed",
        "data/vectorstore",
        "models",
        "logs"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return True


def create_sample_config():
    """Create a sample configuration file."""
    logger = setup_logger("setup.config")
    
    config_file = Path("config.yaml")
    
    if config_file.exists():
        logger.info("Configuration file already exists")
        return True
    
    sample_config = """
# Chromium RAG System Configuration

# GPU Configuration
gpu:
  device: "cuda"
  visible_devices: "0"
  memory_fraction: 0.8
  enable_mixed_precision: true
  enable_compilation: true

# Embedding Configuration  
embedding:
  model_name: "BAAI/bge-m3"
  device: "cuda"
  batch_size: 64
  max_length: 8192
  pooling_method: "cls"
  normalize: true

# Data Configuration
data:
  chromium_repo_path: "d:\\r\\cr\\src"
  cache_dir: "./data/cache"
  embeddings_dir: "./data/embeddings"
  max_chunk_size: 2048
  chunk_overlap: 256

# Vector Database Configuration
vector_db:
  backend: "chromadb"
  host: "localhost"
  port: 8000
  collection_name: "chromium_rag"
  embedding_dimension: 1024
  similarity_metric: "cosine"

# API Configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

# Monitoring Configuration
monitoring:
  wandb_project: "chromium-rag"
  enable_monitoring: true
  log_level: "INFO"
  log_dir: "./logs"
"""
    
    with open(config_file, 'w') as f:
        f.write(sample_config)
    
    logger.info(f"Created sample configuration file: {config_file}")
    return True


def run_system_validation():
    """Run comprehensive system validation."""
    logger = setup_logger("setup.validation")
    
    logger.info("Starting Chromium RAG system validation...")
    
    checks = [
        ("GPU Availability", check_gpu_availability),
        ("Directory Setup", setup_directories),
        ("Configuration", create_sample_config),
        ("Embedding Models", download_embedding_models),
        ("Embedding Generation", test_embedding_generation),
    ]
    
    # Only check Chromium repo if the path exists
    config = get_config()
    repo_path = Path(config.data.chromium_repo_path)
    if repo_path.exists():
        checks.insert(-2, ("Chromium Repository", lambda: check_chromium_repository(str(repo_path))))
    else:
        logger.warning(f"Chromium repository not found at {repo_path} - skipping repository check")
    
    results = {}
    
    for check_name, check_func in checks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running check: {check_name}")
        logger.info(f"{'='*50}")
        
        try:
            result = check_func()
            results[check_name] = result
            
            if result:
                logger.info(f"✅ {check_name}: PASSED")
            else:
                logger.error(f"❌ {check_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {check_name}: ERROR - {e}")
            results[check_name] = False
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("VALIDATION SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for check_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{check_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 System validation completed successfully!")
        logger.info("The Chromium RAG system is ready to use.")
        
        logger.info("\nNext steps:")
        logger.info("1. Adjust configuration in config.yaml if needed")
        logger.info("2. Run data ingestion: python -m rag_system.data.ingest --mode recent --days 7")
        logger.info("3. Start the API server: python -m rag_system.api.server")
        logger.info("4. Launch the web interface: streamlit run rag_system/ui/app.py")
        
    else:
        logger.error("❌ System validation failed. Please fix the issues above.")
        return False
    
    return True


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Chromium RAG System")
    parser.add_argument("--skip-gpu-check", action="store_true", help="Skip GPU availability check")
    parser.add_argument("--skip-models", action="store_true", help="Skip model downloads")
    parser.add_argument("--repo-path", type=str, help="Override Chromium repository path")
    
    args = parser.parse_args()
    
    # Override config if repo path provided
    if args.repo_path:
        os.environ["CHROMIUM_REPO_PATH"] = args.repo_path
    
    # Initialize logging
    logger = setup_logger("setup")
    
    logger.info("🚀 Starting Chromium RAG System Setup")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    try:
        success = run_system_validation()
        if success:
            logger.info("Setup completed successfully! 🎉")
            return 0
        else:
            logger.error("Setup failed! Please check the errors above.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())