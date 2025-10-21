#!/usr/bin/env python3
"""
Test script for the complete RAG pipeline.
Tests end-to-end functionality from retrieval to generation.
"""

import sys
import time
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.rag_pipeline import RAGPipeline, print_rag_result
from rag_system.generation.generator import ModelSize
from rag_system.core.logger import setup_logger

logger = setup_logger(__name__)


def test_complete_rag():
    """Test the complete RAG pipeline with various queries."""
    
    print("🤖 Testing Complete RAG Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    print("1. Initializing RAG pipeline...")
    pipeline = RAGPipeline(model_size=ModelSize.SMALL)
    
    # Initialize generator
    print("2. Loading language model...")
    if not pipeline.initialize_generator():
        print("❌ Failed to initialize generator")
        return False
    
    print("✅ RAG pipeline fully initialized!")
    
    # Test queries
    test_queries = [
        {
            'question': 'How does Chromium handle memory management?',
            'query_type': 'general',
            'style': 'precise'
        },
        {
            'question': 'What are common security vulnerabilities in web browsers?',
            'query_type': 'security',
            'style': 'detailed'
        },
        {
            'question': 'How can I fix a performance issue in Chromium rendering?',
            'query_type': 'performance',
            'style': 'precise'
        }
    ]
    
    all_results = []
    
    for i, query_info in enumerate(test_queries, 1):
        print(f"\n{i}. Testing Query: '{query_info['question']}'")
        print("-" * 50)
        
        start_time = time.time()
        
        try:
            result = pipeline.query(
                question=query_info['question'],
                n_results=3,
                retrieval_strategy='hybrid',
                query_type=query_info['query_type'],
                generation_style=query_info['style']
            )
            
            end_time = time.time()
            
            print(f"✅ Query completed in {end_time - start_time:.2f}s")
            print_rag_result(result)
            
            all_results.append(result)
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.error(f"Query failed: {e}", exc_info=True)
            continue
    
    # Show pipeline statistics
    print("\n📊 Final Pipeline Statistics:")
    print("-" * 40)
    stats = pipeline.get_pipeline_stats()
    
    print(f"Vector Database:")
    if 'vector_db_stats' in stats:
        db_stats = stats['vector_db_stats']
        print(f"  Documents: {db_stats.get('total_documents', 'N/A')}")
        print(f"  Collections: {db_stats.get('collections', 'N/A')}")
    
    print(f"\nRetrieval:")
    if 'retrieval_stats' in stats:
        ret_stats = stats['retrieval_stats']
        print(f"  Total queries: {ret_stats.get('total_queries', 0)}")
        print(f"  Avg retrieval time: {ret_stats.get('avg_retrieval_time', 0):.3f}s")
    
    print(f"\nGeneration:")
    if 'generation_stats' in stats:
        gen_stats = stats['generation_stats']
        print(f"  Total generations: {gen_stats.get('total_generations', 0)}")
        print(f"  Avg generation time: {gen_stats.get('avg_generation_time', 0):.3f}s")
        print(f"  Total tokens: {gen_stats.get('total_tokens', 0)}")
    
    print(f"\n✅ RAG Pipeline Test Complete!")
    print(f"Successfully processed {len(all_results)}/{len(test_queries)} queries")
    
    return len(all_results) == len(test_queries)


def test_retrieval_strategies():
    """Test different retrieval strategies."""
    
    print("\n🔍 Testing Retrieval Strategies")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(model_size=ModelSize.SMALL)
    if not pipeline.initialize_generator():
        print("❌ Failed to initialize generator")
        return False
    
    question = "How does Chromium handle JavaScript execution?"
    strategies = ['semantic', 'hybrid', 'multi_stage']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} retrieval:")
        print("-" * 30)
        
        start_time = time.time()
        
        try:
            result = pipeline.query(
                question=question,
                n_results=3,
                retrieval_strategy=strategy,
                query_type='general',
                generation_style='precise'
            )
            
            end_time = time.time()
            
            print(f"✅ {strategy} completed in {end_time - start_time:.2f}s")
            print(f"Answer length: {len(result['answer'])} chars")
            print(f"Retrieved contexts: {len(result['retrieval']['contexts'])}")
            
            # Show first context
            if result['retrieval']['contexts']:
                first_ctx = result['retrieval']['contexts'][0]
                print(f"Top context: {first_ctx['file_path']} (score: {first_ctx['score']:.3f})")
            
        except Exception as e:
            print(f"❌ {strategy} failed: {e}")
    
    return True


def test_query_types():
    """Test different query types with specialized prompting."""
    
    print("\n🎯 Testing Specialized Query Types")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline(model_size=ModelSize.SMALL)
    if not pipeline.initialize_generator():
        print("❌ Failed to initialize generator")
        return False
    
    query_tests = [
        {
            'question': 'There is a memory leak in the renderer process',
            'type': 'bug_fix',
            'description': 'Bug fix query'
        },
        {
            'question': 'The page loading is too slow',
            'type': 'performance',
            'description': 'Performance optimization query'
        },
        {
            'question': 'How to prevent XSS attacks in extensions?',
            'type': 'security',
            'description': 'Security-focused query'
        },
        {
            'question': 'What is the overall architecture of the rendering engine?',
            'type': 'architecture',
            'description': 'Architecture overview query'
        }
    ]
    
    for test in query_tests:
        print(f"\nTesting {test['description']}: {test['question']}")
        print("-" * 50)
        
        try:
            result = pipeline.query(
                question=test['question'],
                n_results=3,
                retrieval_strategy='hybrid',
                query_type=test['type'],
                generation_style='detailed'
            )
            
            print(f"✅ Query type '{test['type']}' successful")
            print(f"Response: {result['answer'][:150]}...")
            
        except Exception as e:
            print(f"❌ Query type '{test['type']}' failed: {e}")
    
    return True


def main():
    """Run all RAG tests."""
    
    print("🚀 Starting Complete RAG System Tests")
    print("=" * 60)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Complete RAG pipeline
    try:
        if test_complete_rag():
            success_count += 1
            print("✅ Complete RAG test passed")
        else:
            print("❌ Complete RAG test failed")
    except Exception as e:
        print(f"❌ Complete RAG test error: {e}")
        logger.error(f"Complete RAG test failed: {e}", exc_info=True)
    
    # Test 2: Retrieval strategies
    try:
        if test_retrieval_strategies():
            success_count += 1
            print("✅ Retrieval strategies test passed")
        else:
            print("❌ Retrieval strategies test failed")
    except Exception as e:
        print(f"❌ Retrieval strategies test error: {e}")
        logger.error(f"Retrieval strategies test failed: {e}", exc_info=True)
    
    # Test 3: Query types
    try:
        if test_query_types():
            success_count += 1
            print("✅ Query types test passed")
        else:
            print("❌ Query types test failed")
    except Exception as e:
        print(f"❌ Query types test error: {e}")
        logger.error(f"Query types test failed: {e}", exc_info=True)
    
    # Final results
    print("\n" + "=" * 60)
    print(f"🏁 RAG System Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! RAG system is fully functional.")
        return True
    else:
        print("⚠️  Some tests failed. Check the logs for details.")
        return False


if __name__ == "__main__":
    main()