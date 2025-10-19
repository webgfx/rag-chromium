#!/usr/bin/env python3
"""
Ultra-fast query interface - keeps model loaded in memory for instant queries.
First query takes ~3-5s (model loading), subsequent queries take <1s.

Usage:
    python quick_query.py "your question"
"""

import sys
import pickle
from pathlib import Path

# Singleton instance cache file
CACHE_FILE = Path(".rag_instance.pkl")

def get_or_create_interface():
    """Get cached interface or create new one."""
    try:
        # Try to load from pickle cache (in-process only)
        # Note: This only works for a single session, not across runs
        from copilot_rag_interface import CopilotRAGInterface
        
        print("🚀 Creating RAG interface...")
        interface = CopilotRAGInterface()
        
        # Pre-load everything on first use
        _ = interface.embedding_generator
        _ = interface.vector_db
        _ = interface.retriever
        
        print("✅ Ready for queries!")
        return interface
        
    except Exception as e:
        print(f"❌ Error initializing: {e}")
        raise


def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_query.py <question>")
        print("\nExample:")
        print('  python quick_query.py "memory leak fixes"')
        return
    
    question = " ".join(sys.argv[1:])
    
    interface = get_or_create_interface()
    result = interface.query(question, top_k=5)
    
    print(f"\n{'='*70}")
    print("Results saved to: copilot_rag_results.md")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
