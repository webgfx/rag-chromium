#!/usr/bin/env python3
"""Test model loading with detailed error output."""

from sentence_transformers import SentenceTransformer
import traceback

try:
    print("Attempting to load model BAAI/bge-large-en-v1.5...")
    model = SentenceTransformer(
        'BAAI/bge-large-en-v1.5', 
        device='cuda',
        cache_folder='data/cache/models'
    )
    print("✅ Model loaded successfully!")
    print(f"Model device: {model.device}")
    print(f"Max sequence length: {model.max_seq_length}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
