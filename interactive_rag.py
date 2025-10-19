#!/usr/bin/env python3
"""
Interactive RAG system for Chromium development queries.
"""

import sys
from pathlib import Path
import time

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simple_rag_demo import SimpleRAGDemo


def print_banner():
    """Print the welcome banner."""
    print("🤖 Interactive Chromium RAG System")
    print("=" * 50)
    print("Ask questions about Chromium development!")
    print("The system will search through Chromium commit data")
    print("and provide relevant information from the codebase.")
    print()
    print("Commands:")
    print("  /help        - Show this help")
    print("  /stats       - Show system statistics") 
    print("  /exit, /quit - Exit the program")
    print("  <question>   - Ask any question about Chromium")
    print("=" * 50)


def print_help():
    """Print help information."""
    print("\n📖 Help - How to use the Chromium RAG System")
    print("-" * 50)
    print("This system can help you with:")
    print("• Understanding Chromium architecture")
    print("• Finding information about specific components")
    print("• Learning about security implementations")
    print("• Performance optimization techniques")
    print("• Bug fixing approaches")
    print("• Code patterns and best practices")
    print()
    print("Example questions:")
    print("• How does Chromium handle memory management?")
    print("• What are the main components of the V8 engine?")
    print("• How is WebGL implemented in Chromium?")
    print("• What security measures prevent XSS attacks?")
    print("• How does Chromium optimize network requests?")
    print("-" * 50)


def print_stats(demo):
    """Print system statistics."""
    print("\n📊 System Statistics")
    print("-" * 30)
    try:
        stats = demo.vector_db.get_collection_stats()
        print(f"Vector Database:")
        print(f"  • Documents indexed: {stats.get('total_documents', 'N/A')}")
        print(f"  • Collection name: chromium_embeddings")
        print(f"  • Storage path: data/cache/vector_db")
        print()
        print(f"Embedding Model:")
        print(f"  • Model: BAAI/bge-m3")
        print(f"  • Dimensions: 1024")
        print(f"  • Device: CPU")
        print()
        print(f"Retrieval:")
        print(f"  • Strategy: Semantic search")
        print(f"  • Results per query: 3")
        print(f"  • Re-ranking: Disabled")
    except Exception as e:
        print(f"Error getting stats: {e}")
    print("-" * 30)


def format_result_compact(result):
    """Format result in a compact way for interactive use."""
    print(f"\n💡 Answer ({result['retrieval_time']:.2f}s):")
    print("-" * 40)
    
    if not result['contexts']:
        print("❌ No relevant information found.")
        return
    
    # Show a shorter, more focused response
    contexts = result['contexts']
    print(f"📚 Found {len(contexts)} relevant sources:")
    
    for i, ctx in enumerate(contexts, 1):
        file_path = ctx['file_path'] if ctx['file_path'] else 'Unknown file'
        score = ctx['score']
        content = ctx['content'][:150]  # First 150 chars
        
        print(f"\n{i}. {file_path} (relevance: {score:.3f})")
        print(f"   {content}...")
    
    print(f"\n🔍 Based on these Chromium sources, the information shows:")
    print(f"   Recent changes and configurations related to your query.")
    print(f"   For detailed implementation, check the specific files above.")


def main():
    """Run the interactive RAG system."""
    print_banner()
    
    # Initialize the RAG system
    print("🔄 Initializing RAG system...")
    try:
        demo = SimpleRAGDemo()
        print("✅ System ready!")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Interactive loop
    while True:
        try:
            # Get user input
            user_input = input("\n🤖 Ask me about Chromium > ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['/exit', '/quit', '/q']:
                print("👋 Goodbye! Happy Chromium development!")
                break
            elif user_input.lower() == '/help':
                print_help()
                continue
            elif user_input.lower() == '/stats':
                print_stats(demo)
                continue
            elif user_input.startswith('/'):
                print("❓ Unknown command. Type /help for available commands.")
                continue
            
            # Process the query
            print(f"🔍 Searching for: '{user_input}'")
            
            start_time = time.time()
            result = demo.query(user_input, n_results=3)
            
            # Display result
            format_result_compact(result)
            
            # Show query time
            total_time = time.time() - start_time
            print(f"\n⏱️  Total processing time: {total_time:.2f}s")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy Chromium development!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try rephrasing your question or type /help for guidance.")


if __name__ == "__main__":
    main()