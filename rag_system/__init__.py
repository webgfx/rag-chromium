"""
Chromium RAG System - Advanced Retrieval Augmented Generation for Chromium Development

This package provides a comprehensive RAG system specifically designed for processing
and querying Chromium repository data, with full GPU acceleration and state-of-the-art
ML models.
"""

__version__ = "1.0.0"
__author__ = "Chromium RAG Team"
__email__ = "contact@chromium-rag.com"

# Imports will be available after core modules are fully set up
try:
    from .core.config import Config, get_config
    from .core.logger import setup_logger
    
    # Initialize default configuration
    config = get_config()
    logger = setup_logger(__name__)
except ImportError:
    # Fallback during initial setup
    config = None
    logger = None

__all__ = ["config", "logger", "Config", "setup_logger"]