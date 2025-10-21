#!/usr/bin/env python3
"""
Pre-warm the RAG system to make subsequent queries instant.
Run this once after startup, then use copilot_rag_interface.py normally.
"""

import sys
from pathlib import Path
import time

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from copilot_rag_interface import CopilotRAGInterface

def warm_up_system():
    """Pre-load all models and connections."""
    
    print("\n🔥 Warming up RAG system...")
    print("="*60)
    
    start = time.time()
    
    # Create interface
    interface = CopilotRAGInterface()
    
    # Force load all lazy-loaded components
    print("📦 Loading embedding model...")
    _ = interface.embedding_generator
    
    print("💾 Connecting to vector database...")
    _ = interface.vector_db
    
    print("🔧 Initializing retriever...")
    _ = interface.retriever
    
    # Do a test query to fully warm up
    print("🧪 Running test query to warm up system...")
    _ = interface.query("test", top_k=1)
    
    elapsed = time.time() - start
    
    print("="*60)
    print(f"✅ System warmed up in {elapsed:.2f}s")
    print("\nNow you can run queries instantly:")
    print('  python copilot_rag_interface.py "your question"')
    print("\nSubsequent queries will be <1s!")
    print("="*60)


if __name__ == "__main__":
    warm_up_system()
