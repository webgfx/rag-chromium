#!/usr/bin/env python3
"""
Simple interface for GitHub Copilot to query the Chromium RAG system.
Usage: Run queries and results will be saved to a file that Copilot can read.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_system.vector.database import VectorDatabase
from rag_system.retrieval.retriever import AdvancedRetriever
from rag_system.embeddings.generator import EmbeddingGenerator


class CopilotRAGInterface:
    """Simple interface for Copilot to query Chromium knowledge."""
    
    def __init__(self):
        # Lazy initialization - only load when needed
        self._embedding_generator = None
        self._vector_db = None
        self._retriever = None
        self.results_file = Path("copilot_rag_results.md")
    
    @property
    def embedding_generator(self):
        """Lazy load embedding generator."""
        if self._embedding_generator is None:
            print("🔄 Loading embedding model (one-time initialization)...")
            self._embedding_generator = EmbeddingGenerator(
                model_name='BAAI/bge-large-en-v1.5',
                batch_size=64  # Larger batch for efficiency
            )
        return self._embedding_generator
    
    @property
    def vector_db(self):
        """Lazy load vector database."""
        if self._vector_db is None:
            print("🔄 Connecting to vector database...")
            self._vector_db = VectorDatabase(collection_name="chromium_complete")
        return self._vector_db
    
    @property
    def retriever(self):
        """Lazy load retriever."""
        if self._retriever is None:
            self._retriever = AdvancedRetriever(
                vector_db=self.vector_db,
                embedding_generator=self.embedding_generator
            )
        return self._retriever
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Query the RAG system and return formatted results."""
        
        print(f"\n🔍 Querying Chromium RAG: {question}")
        print("⏱️  Generating query embedding...")
        
        # Fast retrieval without expensive reranking or multi-stage
        import time
        start = time.time()
        
        results = self.retriever.retrieve(
            query=question,
            n_results=top_k,
            retrieval_strategy="semantic",  # Fastest strategy
            use_reranking=False  # Skip expensive reranking
        )
        
        elapsed = time.time() - start
        print(f"✅ Retrieved {len(results)} results in {elapsed:.2f}s")
        
        if not results:
            return "No relevant results found."
        
        # Format results efficiently using list comprehension
        print("📝 Formatting results...")
        
        # Build header
        parts = [
            f"# Chromium RAG Query Results\n\n",
            f"**Query:** {question}\n\n",
            f"**Found:** {len(results)} relevant commits in {elapsed:.2f}s\n\n",
            "---\n\n"
        ]
        
        # Build results
        for idx, result in enumerate(results, 1):
            doc = result.search_result.document
            metadata = doc.metadata or {}
            
            parts.append(f"## Result {idx}\n\n")
            parts.append(f"**Commit:** `{metadata.get('commit_sha', 'unknown')[:8]}`\n")
            parts.append(f"**Author:** {metadata.get('author_name', 'unknown')}\n")
            parts.append(f"**Date:** {metadata.get('commit_date', 'unknown')}\n")
            parts.append(f"**Relevance Score:** {result.retrieval_score:.3f}\n\n")
            
            # Only include message if available and non-empty
            msg = metadata.get('commit_message', '')
            if msg:
                parts.append(f"**Message:**\n```\n{msg[:200]}\n```\n\n")
            
            # Content preview
            parts.append(f"**Content:**\n```\n{doc.content[:500]}...\n```\n\n")
            
            # Skip explanations for faster formatting
            parts.append("---\n\n")
        
        formatted_output = ''.join(parts)
        
        # Save to file for Copilot to read
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        print(f"✅ Results saved to: {self.results_file}")
        print(f"📋 You can now ask Copilot to read this file for context!")
        
        return formatted_output
    
    def generate_answer(self, question: str, top_k: int = 5) -> str:
        """Generate a complete answer using RAG."""
        
        print(f"\n🤖 Generating answer for: {question}")
        print("⏱️  Searching database...")
        
        import time
        start = time.time()
        
        # Fast retrieval
        results = self.retriever.retrieve(
            query=question,
            n_results=top_k,
            retrieval_strategy="semantic",
            use_reranking=False
        )
        
        elapsed = time.time() - start
        print(f"✅ Found {len(results)} results in {elapsed:.2f}s")
        
        if not results:
            return "No relevant information found."
        
        # Format efficiently
        parts = [
            f"# Chromium RAG Answer\n\n",
            f"**Question:** {question}\n\n",
            f"**Summary:** Based on {len(results)} relevant commits from Chromium (found in {elapsed:.2f}s):\n\n"
        ]
        
        # Top 3 content previews
        for idx, result in enumerate(results[:3], 1):
            doc = result.search_result.document
            parts.append(f"{idx}. {doc.content[:200]}...\n\n")
        
        parts.append("---\n\n")
        parts.append(f"## Sources ({len(results)} commits)\n\n")
        
        # Compact source list
        for idx, result in enumerate(results, 1):
            doc = result.search_result.document
            metadata = doc.metadata or {}
            sha = metadata.get('commit_sha', 'unknown')[:8]
            author = metadata.get('author_name', 'unknown')
            date = metadata.get('commit_date', 'unknown')[:10]
            parts.append(f"{idx}. Commit `{sha}` by {author} ({date})\n")
        
        formatted_output = ''.join(parts)
        
        # Save for Copilot
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        print(f"✅ Answer saved to: {self.results_file}")
        
        return formatted_output


def main():
    """CLI interface for quick queries."""
    
    if len(sys.argv) < 2:
        print("Usage: python copilot_rag_interface.py <your question>")
        print("\nExample:")
        print('  python copilot_rag_interface.py "How does Chrome handle memory leaks?"')
        print("\nOr use interactively:")
        print('  rag = CopilotRAGInterface()')
        print('  rag.query("your question")')
        return
    
    question = " ".join(sys.argv[1:])
    
    print("\n📚 Chromium RAG Interface for GitHub Copilot\n")
    
    # Create interface (lazy loaded, so fast)
    interface = CopilotRAGInterface()
    
    # Default to quick search (mode 1) - no interactive prompt for speed
    # Use mode=2 explicitly if needed: add --answer flag
    if "--answer" in sys.argv:
        result = interface.generate_answer(question.replace("--answer", "").strip())
    else:
        result = interface.query(question)
    
    print("\n" + "="*70)
    print("Next steps:")
    print("="*70)
    print("1. Open copilot_rag_results.md in VS Code")
    print("2. Ask Copilot: '@workspace read copilot_rag_results.md and help me understand this'")
    print("3. Or paste this into Copilot Chat for more context")
    print("="*70)


if __name__ == "__main__":
    main()
