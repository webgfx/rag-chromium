#!/usr/bin/env python3
"""
Comprehensive validation of the RAG system components.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any

from rag_system.core.config import Config
from rag_system.core.logger import setup_logger
from rag_system.data.chromium import ChromiumDataExtractor
from rag_system.data.preprocessor import DataPreprocessor
from rag_system.data.chunker import TextChunker, CodeChunker, DiffChunker
from rag_system.embeddings.generator import EmbeddingGenerator


def validate_data_ingestion() -> bool:
    """Validate data ingestion pipeline."""
    logger = setup_logger("validation.data_ingestion")
    
    try:
        # Test Chromium data extraction
        extractor = ChromiumDataExtractor()
        commits = extractor.get_recent_commits(max_count=1)
        
        if not commits:
            logger.error("No commits found")
            return False
        
        logger.info(f"✓ Successfully extracted {len(commits)} commits")
        
        # Test preprocessing
        preprocessor = DataPreprocessor()
        commit = commits[0]
        
        # Test diff processing
        if commit.diff:
            processed_diff = preprocessor.process_diff(commit.diff)
            logger.info(f"✓ Successfully processed diff ({len(processed_diff)} files)")
        
        # Test message processing
        if commit.message:
            processed_message = preprocessor.clean_text(commit.message)
            logger.info(f"✓ Successfully processed commit message")
        
        return True
        
    except Exception as e:
        logger.error(f"Data ingestion validation failed: {e}")
        return False


def validate_chunking() -> bool:
    """Validate chunking system."""
    logger = setup_logger("validation.chunking")
    
    try:
        # Test text chunking
        text_chunker = TextChunker()
        sample_text = "This is a sample text for testing chunking functionality. " * 50
        text_chunks = text_chunker.chunk_text(sample_text)
        logger.info(f"✓ Text chunking: {len(text_chunks)} chunks")
        
        # Test code chunking
        code_chunker = CodeChunker()
        sample_code = '''
def sample_function():
    """This is a sample function."""
    for i in range(10):
        print(f"Line {i}")
    return True

class SampleClass:
    def __init__(self):
        self.value = 42
    
    def method(self):
        return self.value * 2
'''
        code_chunks = code_chunker.chunk_code(sample_code, language="python")
        logger.info(f"✓ Code chunking: {len(code_chunks)} chunks")
        
        # Test diff chunking
        diff_chunker = DiffChunker()
        sample_diff = '''
--- a/test.py
+++ b/test.py
@@ -1,5 +1,7 @@
 def old_function():
-    return "old"
+    return "new"
+    # Added comment
 
 def another_function():
     pass
'''
        diff_chunks = diff_chunker.chunk_diff(sample_diff)
        logger.info(f"✓ Diff chunking: {len(diff_chunks)} chunks")
        
        return True
        
    except Exception as e:
        logger.error(f"Chunking validation failed: {e}")
        return False


def validate_embeddings() -> bool:
    """Validate embedding generation."""
    logger = setup_logger("validation.embeddings")
    
    try:
        # Test embedding generation
        generator = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        
        sample_texts = [
            "This is a test document about machine learning.",
            "Python function for data processing.",
            "Bug fix in the rendering pipeline.",
            "Added new feature for user authentication."
        ]
        
        embeddings = generator.encode_texts(sample_texts)
        
        if len(embeddings) != len(sample_texts):
            logger.error(f"Embedding count mismatch: {len(embeddings)} != {len(sample_texts)}")
            return False
        
        embedding_dim = len(embeddings[0])
        logger.info(f"✓ Generated {len(embeddings)} embeddings (dim: {embedding_dim})")
        
        # Test similarity
        similarity = np.dot(embeddings[0], embeddings[1])
        logger.info(f"✓ Embedding similarity computation: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Embedding validation failed: {e}")
        return False


def validate_processed_data() -> bool:
    """Validate processed data files."""
    logger = setup_logger("validation.processed_data")
    
    try:
        # Check for processed data files
        cache_dir = Path("data/cache/processed")
        processed_files = list(cache_dir.glob("*.json"))
        
        if not processed_files:
            logger.warning("No processed data files found")
            return True  # Not an error, just no data yet
        
        # Validate most recent file
        latest_file = max(processed_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'chunks' not in data:
            logger.error("Processed data missing 'chunks' field")
            return False
        
        chunks = data['chunks']
        logger.info(f"✓ Found {len(chunks)} processed chunks in {latest_file.name}")
        
        # Validate chunk structure
        if chunks:
            sample_chunk = chunks[0]
            required_fields = ['id', 'content', 'type', 'commit_hash']
            missing_fields = [field for field in required_fields if field not in sample_chunk]
            
            if missing_fields:
                logger.error(f"Missing required fields in chunks: {missing_fields}")
                return False
        
        logger.info("✓ Chunk structure validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Processed data validation failed: {e}")
        return False


def validate_embeddings_data() -> bool:
    """Validate embedding data files."""
    logger = setup_logger("validation.embeddings_data")
    
    try:
        # Check for embedding files
        embeddings_dir = Path("data/embeddings")
        if not embeddings_dir.exists():
            logger.warning("No embeddings directory found")
            return True  # Not an error, just no embeddings yet
        
        embedding_files = list(embeddings_dir.glob("*.json"))
        
        if not embedding_files:
            logger.warning("No embedding files found")
            return True  # Not an error, just no embeddings yet
        
        # Validate most recent file
        latest_file = max(embedding_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        required_fields = ['model', 'embedding_dimension', 'total_embeddings', 'embeddings']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            logger.error(f"Missing required fields in embedding data: {missing_fields}")
            return False
        
        embeddings = data['embeddings']
        expected_count = data['total_embeddings']
        embedding_dim = data['embedding_dimension']
        
        if len(embeddings) != expected_count:
            logger.error(f"Embedding count mismatch: {len(embeddings)} != {expected_count}")
            return False
        
        if embeddings and len(embeddings[0]) != embedding_dim:
            logger.error(f"Embedding dimension mismatch: {len(embeddings[0])} != {embedding_dim}")
            return False
        
        logger.info(f"✓ Validated {len(embeddings)} embeddings (dim: {embedding_dim}) in {latest_file.name}")
        return True
        
    except Exception as e:
        logger.error(f"Embedding data validation failed: {e}")
        return False


def main():
    """Run comprehensive RAG system validation."""
    logger = setup_logger("validation.main")
    
    logger.info("Starting comprehensive RAG system validation...")
    
    validations = [
        ("Data Ingestion", validate_data_ingestion),
        ("Chunking System", validate_chunking),
        ("Embedding Generation", validate_embeddings),
        ("Processed Data Files", validate_processed_data),
        ("Embedding Data Files", validate_embeddings_data),
    ]
    
    results = {}
    
    for name, validation_func in validations:
        logger.info(f"\\n{'='*50}")
        logger.info(f"Validating: {name}")
        logger.info('='*50)
        
        try:
            result = validation_func()
            results[name] = result
            
            if result:
                logger.info(f"✅ {name}: PASSED")
            else:
                logger.error(f"❌ {name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {name}: ERROR - {e}")
            results[name] = False
    
    # Summary
    logger.info(f"\\n{'='*50}")
    logger.info("VALIDATION SUMMARY")
    logger.info('='*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{name:<25} {status}")
    
    logger.info(f"\\nOverall: {passed}/{total} validations passed")
    
    if passed == total:
        logger.info("🎉 All validations PASSED! RAG system is fully operational.")
    else:
        logger.warning(f"⚠️  {total - passed} validations FAILED. Review errors above.")
    
    return passed == total


if __name__ == "__main__":
    main()