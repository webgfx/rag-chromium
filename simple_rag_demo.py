#!/usr/bin/env python3
"""
Quick test script to demonstrate RAG functionality with fallback generation.
"""

import sys
from pathlib import Path
import time

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from rag_system.vector.database import VectorDatabase
from rag_system.embeddings.generator import EmbeddingGenerator
from rag_system.retrieval.retriever import AdvancedRetriever
from rag_system.core.logger import setup_logger

logger = setup_logger(__name__)


class SimpleRAGDemo:
    """Simple RAG demonstration focusing on retrieval with basic response formatting."""
    
    def __init__(self):
        """Initialize the RAG components."""
        print("🤖 Initializing Simple RAG Demo")
        print("=" * 50)
        
        # Initialize vector database
        print("1. Loading vector database...")
        self.vector_db = VectorDatabase(collection_name="chromium_embeddings")
        
        # Initialize embedding generator with the same model used for indexing
        print("2. Loading embedding model...")
        self.embedding_generator = EmbeddingGenerator(model_name="BAAI/bge-m3")  # Use BGE-M3
        
        # Initialize retriever
        print("3. Setting up retriever...")
        self.retriever = AdvancedRetriever(self.vector_db, self.embedding_generator)
        
        print("✅ RAG Demo initialized successfully!")
    
    def query(self, question: str, n_results: int = 3) -> dict:
        """Process a RAG query with retrieval and simple response formatting."""
        print(f"\n🔍 Processing: '{question}'")
        print("-" * 60)
        
        start_time = time.time()
        
        # Step 1: Retrieval
        try:
            retrieval_results = self.retriever.retrieve(
                query=question,
                n_results=n_results,
                retrieval_strategy='semantic',  # Use semantic only to avoid dimension issues
                use_reranking=False  # Disable reranking to avoid TF-IDF issues
            )
            
            print(f"✅ Retrieved {len(retrieval_results)} relevant contexts")
            
            # Step 2: Format response based on retrieved context
            response = self._format_response(question, retrieval_results)
            
            end_time = time.time()
            
            result = {
                'question': question,
                'answer': response,
                'retrieval_time': end_time - start_time,
                'contexts': [
                    {
                        'content': result.search_result.document.content[:300] + '...',
                        'file_path': result.search_result.document.metadata.get('file_path', 'Unknown'),
                        'score': result.search_result.score,
                        'chunk_id': result.search_result.document.metadata.get('chunk_id', 'N/A')
                    }
                    for result in retrieval_results
                ]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return {
                'question': question,
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'retrieval_time': time.time() - start_time,
                'contexts': [],
                'error': str(e)
            }
    
    def _format_response(self, question: str, retrieval_results) -> str:
        """Format a response based on retrieved contexts."""
        if not retrieval_results:
            return "I couldn't find relevant information in the Chromium codebase for your question."
        
        # Extract key information from contexts
        contexts = []
        for result in retrieval_results:
            doc = result.search_result.document
            file_path = doc.metadata.get('file_path', 'Unknown file')
            content = doc.content[:200]  # First 200 chars
            contexts.append(f"From {file_path}: {content}...")
        
        # Create formatted response
        response = f"""Based on the Chromium codebase, here's what I found regarding your question: "{question}"

**Relevant Information:**

"""
        
        for i, context in enumerate(contexts, 1):
            response += f"{i}. {context}\n\n"
        
        response += """**Summary:**
The retrieved contexts show relevant information from the Chromium codebase. For detailed implementation, please refer to the specific files mentioned above.

**Note:** This is a retrieval-based response. The information comes directly from the Chromium repository commits and code changes."""
        
        return response


def main():
    """Run the simple RAG demo."""
    print("🚀 Simple RAG Demo for Chromium Development")
    print("=" * 60)
    
    # Initialize demo
    try:
        demo = SimpleRAGDemo()
    except Exception as e:
        print(f"❌ Failed to initialize demo: {e}")
        return
    
    # Test queries
    test_queries = [
        "How does Chromium handle memory management?",
        "What are the main components of the rendering engine?",
        "How is JavaScript execution handled in Chromium?",
        "What security measures are implemented in Chromium?",
        "How does Chromium optimize page loading performance?"
    ]
    
    print(f"\nTesting {len(test_queries)} queries...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📝 Query {i}/{len(test_queries)}")
        
        try:
            result = demo.query(query)
            
            # Display result
            print(f"❓ Question: {result['question']}")
            print(f"⏱️  Retrieval Time: {result['retrieval_time']:.2f}s")
            print(f"📊 Contexts Found: {len(result['contexts'])}")
            
            if result['contexts']:
                print("\n🔍 Top Context Sources:")
                for j, ctx in enumerate(result['contexts'][:2], 1):  # Show top 2
                    print(f"   {j}. {ctx['file_path']} (score: {ctx['score']:.3f})")
            
            print(f"\n💡 Response:\n{result['answer']}")
            print("\n" + "="*60)
            
        except Exception as e:
            print(f"❌ Query {i} failed: {e}")
            continue
    
    print("\n🎉 RAG Demo completed!")
    
    # Show some statistics
    print("\n📊 System Status:")
    stats = demo.vector_db.get_collection_stats()
    print(f"  Vector Database: {stats.get('total_documents', 'N/A')} documents")
    print(f"  Embedding Model: BAAI/bge-m3 (1024 dimensions)")
    print(f"  Retrieval Strategy: Semantic search with BGE embeddings")


if __name__ == "__main__":
    main()