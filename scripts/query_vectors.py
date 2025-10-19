#!/usr/bin/env python3
"""
Vector database query interface for testing and exploration.
"""

import argparse
import json
from typing import List, Dict, Any

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.vector.database import VectorDatabase, SearchResult


def print_search_results(results: List[SearchResult], show_content: bool = True):
    """Print search results in a readable format."""
    if not results:
        print("No results found.")
        return
    
    print(f"\nFound {len(results)} results:")
    print("=" * 80)
    
    for result in results:
        print(f"Rank: {result.rank} | Score: {result.score:.4f}")
        print(f"ID: {result.document.id}")
        
        metadata = result.document.metadata or {}
        print(f"File: {metadata.get('file_path', 'N/A')}")
        print(f"Type: {metadata.get('chunk_type', 'N/A')}")
        print(f"Language: {metadata.get('language', 'N/A')}")
        print(f"Commit: {metadata.get('commit_hash', 'N/A')[:8]}...")
        print(f"Author: {metadata.get('author', 'N/A')}")
        
        if show_content:
            content = result.document.content
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"Content: {content}")
        
        print("-" * 80)


def interactive_search(vector_db: VectorDatabase):
    """Interactive search interface."""
    print("\n🔍 Interactive Vector Database Search")
    print("Commands:")
    print("  search <query>         - Semantic search")
    print("  hybrid <query>         - Hybrid search (semantic + keyword)")
    print("  filter <query> <key=value> - Search with metadata filter")
    print("  stats                  - Show collection statistics") 
    print("  quit                   - Exit")
    print("-" * 60)
    
    while True:
        try:
            command = input("\n> ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            parts = command.split(None, 1)
            cmd = parts[0].lower()
            
            if cmd == 'search' and len(parts) > 1:
                query = parts[1]
                print(f"Searching for: '{query}'")
                results = vector_db.search(query, n_results=5)
                print_search_results(results)
                
            elif cmd == 'hybrid' and len(parts) > 1:
                query = parts[1]
                print(f"Hybrid search for: '{query}'")
                results = vector_db.hybrid_search(query, n_results=5)
                print_search_results(results)
                
            elif cmd == 'filter' and len(parts) > 1:
                # Parse "query key=value" format
                args = parts[1].split()
                if len(args) >= 2:
                    query = args[0]
                    filter_parts = '='.join(args[1:]).split('=', 1)
                    if len(filter_parts) == 2:
                        filter_key, filter_value = filter_parts
                        where = {filter_key: filter_value}
                        print(f"Searching for: '{query}' with filter {filter_key}={filter_value}")
                        results = vector_db.search(query, n_results=5, where=where)
                        print_search_results(results)
                    else:
                        print("Invalid filter format. Use: filter <query> <key=value>")
                else:
                    print("Usage: filter <query> <key=value>")
                    
            elif cmd == 'stats':
                stats = vector_db.get_collection_stats()
                print("\n📊 Collection Statistics:")
                print(f"Collection: {stats.get('collection_name')}")
                print(f"Total documents: {stats.get('total_documents')}")
                print(f"Persist directory: {stats.get('persist_directory')}")
                
                sample_metadata = stats.get('sample_metadata', [])
                if sample_metadata:
                    print(f"Sample metadata fields:")
                    fields = set()
                    for meta in sample_metadata:
                        if meta:
                            fields.update(meta.keys())
                    for field in sorted(fields):
                        print(f"  - {field}")
                        
            else:
                print("Unknown command. Type 'quit' to exit.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Query vector database")
    parser.add_argument("--collection", default="chromium_embeddings",
                       help="Collection name to query")
    parser.add_argument("--query", help="Search query (non-interactive mode)")
    parser.add_argument("--n-results", type=int, default=10,
                       help="Number of results to return")
    parser.add_argument("--hybrid", action="store_true",
                       help="Use hybrid search")
    parser.add_argument("--filter", help="Metadata filter (key=value format)")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive mode")
    parser.add_argument("--stats", action="store_true",
                       help="Show collection statistics")
    parser.add_argument("--no-content", action="store_true",
                       help="Don't show document content in results")
    
    args = parser.parse_args()
    
    # Setup
    config = get_config()
    logger = setup_logger(__name__)
    
    # Initialize vector database
    logger.info(f"Connecting to vector database collection: {args.collection}")
    vector_db = VectorDatabase(collection_name=args.collection)
    
    # Show stats if requested
    if args.stats:
        stats = vector_db.get_collection_stats()
        print("\n📊 Collection Statistics:")
        print(f"Collection: {stats.get('collection_name')}")
        print(f"Total documents: {stats.get('total_documents')}")
        print(f"Persist directory: {stats.get('persist_directory')}")
        
        sample_metadata = stats.get('sample_metadata', [])
        if sample_metadata:
            print(f"Sample metadata fields:")
            fields = set()
            for meta in sample_metadata:
                if meta:
                    fields.update(meta.keys())
            for field in sorted(fields):
                print(f"  - {field}")
        
        if not args.query and not args.interactive:
            return
    
    # Interactive mode
    if args.interactive:
        interactive_search(vector_db)
        return
    
    # Single query mode
    if args.query:
        # Parse filter if provided
        where = None
        if args.filter:
            filter_parts = args.filter.split('=', 1)
            if len(filter_parts) == 2:
                where = {filter_parts[0]: filter_parts[1]}
            else:
                print(f"Invalid filter format: {args.filter}")
                return
        
        # Perform search
        if args.hybrid:
            print(f"Hybrid search for: '{args.query}'")
            results = vector_db.hybrid_search(
                args.query, 
                n_results=args.n_results,
                where=where
            )
        else:
            print(f"Semantic search for: '{args.query}'")
            results = vector_db.search(
                args.query,
                n_results=args.n_results,
                where=where
            )
        
        print_search_results(results, show_content=not args.no_content)
        
        # Save results to JSON if we have results
        if results:
            output_file = f"search_results_{args.collection}.json"
            results_data = [result.to_dict() for result in results]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'query': args.query,
                    'search_type': 'hybrid' if args.hybrid else 'semantic',
                    'filter': where,
                    'total_results': len(results),
                    'results': results_data
                }, f, indent=2, default=str)
            
            print(f"\nResults saved to: {output_file}")
    
    else:
        print("No query provided. Use --query for single search or --interactive for interactive mode.")


if __name__ == "__main__":
    main()