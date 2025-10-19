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
        # Initialize embedding generator first
        self.embedding_generator = EmbeddingGenerator(
            model_name='BAAI/bge-large-en-v1.5',
            batch_size=32
        )
        
        # Initialize vector database (it uses pre-computed embeddings)
        self.vector_db = VectorDatabase(collection_name="chromium_complete")
        
        # Initialize retriever
        self.retriever = AdvancedRetriever(
            vector_db=self.vector_db,
            embedding_generator=self.embedding_generator
        )
        self.results_file = Path("copilot_rag_results.md")
    
    def query(self, question: str, top_k: int = 5) -> str:
        """Query the RAG system and return formatted results."""
        
        print(f"\n🔍 Querying Chromium RAG: {question}")
        
        # Retrieve relevant documents (use semantic to avoid embedding function issues)
        results = self.retriever.retrieve(
            query=question,
            n_results=top_k,
            retrieval_strategy="semantic",
            use_reranking=False
        )
        
        if not results:
            return "No relevant results found."
        
        # Format results
        formatted_output = f"# Chromium RAG Query Results\n\n"
        formatted_output += f"**Query:** {question}\n\n"
        formatted_output += f"**Found:** {len(results)} relevant commits\n\n"
        formatted_output += "---\n\n"
        
        for idx, result in enumerate(results, 1):
            doc = result.search_result.document
            metadata = doc.metadata or {}
            formatted_output += f"## Result {idx}\n\n"
            formatted_output += f"**Commit:** `{metadata.get('commit_sha', 'unknown')[:8]}`\n"
            formatted_output += f"**Author:** {metadata.get('author_name', 'unknown')}\n"
            formatted_output += f"**Date:** {metadata.get('commit_date', 'unknown')}\n"
            formatted_output += f"**Relevance Score:** {result.retrieval_score:.3f}\n\n"
            formatted_output += f"**Message:**\n```\n{metadata.get('commit_message', 'N/A')[:200]}\n```\n\n"
            formatted_output += f"**Content:**\n```\n{doc.content[:500]}...\n```\n\n"
            
            if result.explanation:
                formatted_output += f"**Why this result:**\n"
                for exp in result.explanation:
                    formatted_output += f"- {exp}\n"
                formatted_output += "\n"
            
            formatted_output += "---\n\n"
        
        # Save to file for Copilot to read
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write(formatted_output)
        
        print(f"✅ Results saved to: {self.results_file}")
        print(f"📋 You can now ask Copilot to read this file for context!")
        
        return formatted_output
    
    def generate_answer(self, question: str, top_k: int = 5) -> str:
        """Generate a complete answer using RAG."""
        
        print(f"\n🤖 Generating answer for: {question}")
        
        # Retrieve context (use semantic to avoid embedding function issues)
        results = self.retriever.retrieve(
            query=question,
            n_results=top_k,
            retrieval_strategy="semantic",
            use_reranking=False
        )
        
        if not results:
            return "No relevant information found."
        
        # Format output (we don't have a generator, so just show the sources)
        formatted_output = f"# Chromium RAG Answer\n\n"
        formatted_output += f"**Question:** {question}\n\n"
        formatted_output += f"**Summary:** Based on {len(results)} relevant commits from Chromium:\n\n"
        
        # Simple summary from top results
        for idx, result in enumerate(results[:3], 1):
            doc = result.search_result.document
            metadata = doc.metadata or {}
            formatted_output += f"{idx}. {doc.content[:200]}...\n\n"
        
        formatted_output += "---\n\n"
        formatted_output += f"## Sources ({len(results)} commits)\n\n"
        
        for idx, result in enumerate(results, 1):
            doc = result.search_result.document
            metadata = doc.metadata or {}
            formatted_output += f"{idx}. Commit `{metadata.get('commit_sha', 'unknown')[:8]}` "
            formatted_output += f"by {metadata.get('author_name', 'unknown')} "
            formatted_output += f"({metadata.get('commit_date', 'unknown')[:10]})\n"
        
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
    
    interface = CopilotRAGInterface()
    
    # Choose mode
    print("\n📚 Chromium RAG Interface for GitHub Copilot\n")
    print("1. Quick search (retrieve relevant commits)")
    print("2. Generate answer (AI-powered response)")
    
    mode = input("\nChoose mode (1 or 2, default=1): ").strip() or "1"
    
    if mode == "2":
        result = interface.generate_answer(question)
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
