#!/usr/bin/env python3
"""
Test script for the advanced retrieval system.
"""

import argparse
import json
from typing import List

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.vector.database import VectorDatabase
from rag_system.embeddings.generator import EmbeddingGenerator
from rag_system.retrieval.retriever import AdvancedRetriever, RetrievalResult


def print_retrieval_results(results: List[RetrievalResult], show_details: bool = True):
    """Print retrieval results in a formatted way."""
    if not results:
        print("No results found.")
        return
    
    print(f"\n🔍 Found {len(results)} results:")
    print("=" * 100)
    
    for i, result in enumerate(results):
        document = result.search_result.document
        metadata = document.metadata or {}
        
        print(f"\n📄 Result #{i+1}")
        print(f"   Score: {result.retrieval_score:.4f} (original: {result.search_result.score:.4f})")
        print(f"   File: {metadata.get('file_path', 'N/A')}")
        print(f"   Type: {metadata.get('chunk_type', 'N/A')}")
        print(f"   Author: {metadata.get('author', 'N/A')}")
        print(f"   Date: {metadata.get('commit_date', 'N/A')}")
        
        # Show content preview
        content = document.content
        if len(content) > 300:
            content = content[:300] + "..."
        print(f"   Content: {content}")
        
        if show_details:
            # Show relevance signals
            if result.relevance_signals:
                print(f"   Signals: {result.relevance_signals}")
            
            # Show explanation
            if result.explanation:
                print(f"   Why: {'; '.join(result.explanation)}")
        
        print("-" * 100)


def test_retrieval_strategies():
    """Test different retrieval strategies."""
    logger = setup_logger(__name__)
    
    # Initialize components
    logger.info("Initializing retrieval system...")
    vector_db = VectorDatabase(collection_name="chromium_embeddings")
    embedding_generator = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
    retriever = AdvancedRetriever(vector_db, embedding_generator)
    
    # Test queries with different characteristics
    test_queries = [
        {
            'query': 'memory leak bug fix',
            'description': 'Bug fix query with technical terms'
        },
        {
            'query': 'GPU performance optimization in rendering pipeline',
            'description': 'Performance query with specific components'
        },
        {
            'query': 'crash in blink renderer',
            'description': 'Component-specific crash query'
        },
        {
            'query': 'security vulnerability CVE patch',
            'description': 'Security-focused query'
        },
        {
            'query': 'JavaScript V8 engine compilation',
            'description': 'Technology-specific query'
        }
    ]
    
    strategies = ['semantic', 'hybrid', 'multi_stage']
    
    for query_info in test_queries:
        query = query_info['query']
        description = query_info['description']
        
        print(f"\n{'='*80}")
        print(f"🔍 Testing Query: '{query}'")
        print(f"📝 Description: {description}")
        print('='*80)
        
        for strategy in strategies:
            print(f"\n🔧 Strategy: {strategy.upper()}")
            print("-" * 50)
            
            try:
                results = retriever.retrieve(
                    query=query,
                    n_results=3,
                    retrieval_strategy=strategy,
                    use_reranking=True
                )
                
                print_retrieval_results(results, show_details=False)
                
            except Exception as e:
                logger.error(f"Error with strategy {strategy}: {e}")
    
    # Show retrieval system stats
    print(f"\n{'='*80}")
    print("📊 Retrieval System Statistics")
    print('='*80)
    
    stats = retriever.get_retrieval_stats()
    print(json.dumps(stats, indent=2, default=str))


def interactive_retrieval():
    """Interactive retrieval interface."""
    logger = setup_logger(__name__)
    
    # Initialize components
    logger.info("Initializing retrieval system...")
    vector_db = VectorDatabase(collection_name="chromium_embeddings")
    embedding_generator = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
    retriever = AdvancedRetriever(vector_db, embedding_generator)
    
    print("\n🔍 Interactive Advanced Retrieval System")
    print("Commands:")
    print("  <query>                    - Search with default settings")
    print("  semantic <query>           - Use semantic retrieval")
    print("  hybrid <query>             - Use hybrid retrieval") 
    print("  multistage <query>         - Use multi-stage retrieval")
    print("  stats                      - Show system statistics")
    print("  quit                       - Exit")
    print("-" * 80)
    
    while True:
        try:
            command = input("\n🔍 > ").strip()
            
            if not command:
                continue
            
            if command.lower() in ['quit', 'exit', 'q']:
                break
            
            parts = command.split(None, 1)
            
            if parts[0].lower() == 'stats':
                stats = retriever.get_retrieval_stats()
                print("\n📊 System Statistics:")
                print(json.dumps(stats, indent=2, default=str))
                continue
            
            # Determine strategy and query
            if parts[0].lower() in ['semantic', 'hybrid', 'multistage']:
                if len(parts) < 2:
                    print("Usage: <strategy> <query>")
                    continue
                strategy = parts[0].lower()
                if strategy == 'multistage':
                    strategy = 'multi_stage'
                query = parts[1]
            else:
                strategy = 'hybrid'  # Default strategy
                query = command
            
            print(f"\n🔍 Searching: '{query}' (strategy: {strategy})")
            
            # Perform retrieval
            results = retriever.retrieve(
                query=query,
                n_results=5,
                retrieval_strategy=strategy,
                use_reranking=True
            )
            
            print_retrieval_results(results, show_details=True)
            
        except KeyboardInterrupt:
            print("\nBye! 👋")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test advanced retrieval system")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive mode")
    parser.add_argument("--test-strategies", action="store_true",
                       help="Test all retrieval strategies")
    parser.add_argument("--query", help="Single query to test")
    parser.add_argument("--strategy", default="hybrid",
                       choices=['semantic', 'hybrid', 'multi_stage'],
                       help="Retrieval strategy to use")
    parser.add_argument("--n-results", type=int, default=5,
                       help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_retrieval()
    elif args.test_strategies:
        test_retrieval_strategies()
    elif args.query:
        # Single query test
        logger = setup_logger(__name__)
        
        # Initialize components
        vector_db = VectorDatabase(collection_name="chromium_embeddings")
        embedding_generator = EmbeddingGenerator(model_name="BAAI/bge-large-en-v1.5")
        retriever = AdvancedRetriever(vector_db, embedding_generator)
        
        print(f"🔍 Query: '{args.query}'")
        print(f"🔧 Strategy: {args.strategy}")
        
        results = retriever.retrieve(
            query=args.query,
            n_results=args.n_results,
            retrieval_strategy=args.strategy,
            use_reranking=True
        )
        
        print_retrieval_results(results, show_details=True)
        
    else:
        print("Use --interactive, --test-strategies, or --query to test the system")


if __name__ == "__main__":
    main()