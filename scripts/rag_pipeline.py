#!/usr/bin/env python3
"""
Complete RAG pipeline combining retrieval and generation.
"""

import argparse
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.vector.database import VectorDatabase
from rag_system.embeddings.generator import EmbeddingGenerator
from rag_system.retrieval.retriever import AdvancedRetriever
from rag_system.generation.generator import AdvancedGenerator, ModelSize


class RAGPipeline:
    """Complete RAG pipeline orchestrating retrieval and generation."""
    
    def __init__(
        self,
        collection_name: str = "chromium_embeddings",
        model_size: ModelSize = ModelSize.SMALL,
        embedding_model: str = "BAAI/bge-large-en-v1.5"
    ):
        """
        Initialize the complete RAG pipeline.
        
        Args:
            collection_name: Vector database collection name
            model_size: Size of the LLM to use
            embedding_model: Embedding model for retrieval
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.RAGPipeline")
        
        # Initialize components
        self.logger.info("Initializing RAG pipeline components...")
        
        # Vector database and retrieval
        self.vector_db = VectorDatabase(collection_name=collection_name)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.retriever = AdvancedRetriever(self.vector_db, self.embedding_generator)
        
        # Generation
        self.generator = AdvancedGenerator(model_size=model_size)
        
        self.logger.info("RAG pipeline initialized")
    
    def initialize_generator(self, custom_model_name: Optional[str] = None) -> bool:
        """Initialize the generation component."""
        return self.generator.initialize(custom_model_name)
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        retrieval_strategy: str = "hybrid",
        query_type: str = "general",
        generation_style: str = "precise",
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Process a complete RAG query from question to response.
        
        Args:
            question: User question
            n_results: Number of retrieval results to use
            retrieval_strategy: Retrieval strategy to use
            query_type: Type of query for specialized prompting
            generation_style: Generation style for response
            stream: Whether to stream the response
            
        Returns:
            Complete RAG result including retrieval and generation
        """
        self.logger.info(f"Processing RAG query: '{question[:50]}...'")
        
        # Step 1: Retrieval
        retrieval_results = self.retriever.retrieve(
            query=question,
            n_results=n_results,
            retrieval_strategy=retrieval_strategy,
            use_reranking=True
        )
        
        self.logger.info(f"Retrieved {len(retrieval_results)} relevant contexts")
        
        # Step 2: Generation
        generation_result = self.generator.generate_response(
            query=question,
            retrieval_results=retrieval_results,
            query_type=query_type,
            generation_style=generation_style,
            stream=stream
        )
        
        # Combine results
        rag_result = {
            'question': question,
            'answer': generation_result.response,
            'retrieval': {
                'strategy': retrieval_strategy,
                'num_results': len(retrieval_results),
                'contexts': [
                    {
                        'content': result.search_result.document.content[:200] + '...',
                        'file_path': result.search_result.document.metadata.get('file_path', ''),
                        'score': result.search_result.score,
                        'explanation': result.explanation
                    }
                    for result in retrieval_results
                ]
            },
            'generation': {
                'model': generation_result.model_name,
                'query_type': query_type,
                'style': generation_style,
                'generation_time': generation_result.generation_time,
                'token_count': generation_result.token_count,
                'metadata': generation_result.metadata
            },
            'timestamp': generation_result.metadata.get('timestamp')
        }
        
        self.logger.info(f"RAG query completed in {generation_result.generation_time:.2f}s")
        
        return rag_result
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            'retrieval_stats': self.retriever.get_retrieval_stats(),
            'generation_stats': self.generator.get_generation_stats(),
            'vector_db_stats': self.vector_db.get_collection_stats()
        }


def print_rag_result(result: Dict[str, Any]):
    """Print RAG result in a formatted way."""
    print("\n" + "="*80)
    print(f"🤖 RAG RESPONSE")
    print("="*80)
    
    print(f"\n❓ Question: {result['question']}")
    print(f"\n💡 Answer:\n{result['answer']}")
    
    print(f"\n📊 Generation Stats:")
    gen_info = result['generation']
    print(f"   Model: {gen_info['model']}")
    print(f"   Time: {gen_info['generation_time']:.2f}s")
    print(f"   Tokens: {gen_info['token_count']}")
    print(f"   Style: {gen_info['style']} ({gen_info['query_type']})")
    
    print(f"\n🔍 Retrieved Contexts ({result['retrieval']['num_results']}):")
    for i, ctx in enumerate(result['retrieval']['contexts'], 1):
        print(f"   {i}. {ctx['file_path']} (score: {ctx['score']:.3f})")
        if ctx['explanation']:
            print(f"      Why: {'; '.join(ctx['explanation'][:2])}")
    
    print("\n" + "="*80)


def interactive_rag():
    """Interactive RAG interface."""
    print("\n🤖 Interactive Chromium RAG System")
    print("="*50)
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = RAGPipeline(model_size=ModelSize.SMALL)
    
    # Initialize generator
    print("Loading language model...")
    if not pipeline.initialize_generator():
        print("❌ Failed to initialize generator")
        return
    
    print("✅ RAG pipeline ready!")
    print("\nCommands:")
    print("  <question>                     - Ask a question")
    print("  :debug <question>              - Debug mode with detailed info")
    print("  :style <style> <question>      - Use specific generation style")
    print("  :type <type> <question>        - Use specific query type")
    print("  :stats                         - Show pipeline statistics")
    print("  :quit                          - Exit")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n🤖 > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in [':quit', ':exit', ':q']:
                break
            
            if user_input == ':stats':
                stats = pipeline.get_pipeline_stats()
                print("\n📊 Pipeline Statistics:")
                print(json.dumps(stats, indent=2, default=str))
                continue
            
            # Parse commands
            parts = user_input.split(' ', 2)
            
            if parts[0].startswith(':'):
                command = parts[0][1:]  # Remove ':'
                
                if command == 'debug' and len(parts) > 1:
                    question = ' '.join(parts[1:])
                    generation_style = 'detailed'
                elif command == 'style' and len(parts) > 2:
                    generation_style = parts[1]
                    question = ' '.join(parts[2:])
                elif command == 'type' and len(parts) > 2:
                    query_type = parts[1]
                    question = ' '.join(parts[2:])
                    generation_style = 'precise'
                else:
                    print("Invalid command format")
                    continue
            else:
                question = user_input
                generation_style = 'precise'
                query_type = 'general'
            
            # Set defaults if not specified
            if 'query_type' not in locals():
                query_type = 'general'
            if 'generation_style' not in locals():
                generation_style = 'precise'
            
            print(f"\n🔍 Processing: '{question}'")
            print(f"   Style: {generation_style}, Type: {query_type}")
            
            # Process query
            result = pipeline.query(
                question=question,
                n_results=3,
                retrieval_strategy='hybrid',
                query_type=query_type,
                generation_style=generation_style
            )
            
            print_rag_result(result)
            
            # Reset variables
            if 'query_type' in locals():
                del query_type
            if 'generation_style' in locals():
                del generation_style
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Complete RAG pipeline for Chromium development")
    parser.add_argument("--interactive", action="store_true",
                       help="Start interactive mode")
    parser.add_argument("--question", help="Single question to process")
    parser.add_argument("--model-size", choices=['small', 'medium', 'large'],
                       default='small', help="LLM model size")
    parser.add_argument("--query-type", 
                       choices=['general', 'bug_fix', 'performance', 'security', 'architecture'],
                       default='general', help="Query type for specialized prompting")
    parser.add_argument("--generation-style",
                       choices=['precise', 'creative', 'detailed'],
                       default='precise', help="Generation style")
    parser.add_argument("--retrieval-strategy",
                       choices=['semantic', 'hybrid', 'multi_stage'],
                       default='hybrid', help="Retrieval strategy")
    parser.add_argument("--n-results", type=int, default=5,
                       help="Number of retrieval results to use")
    parser.add_argument("--stats", action="store_true",
                       help="Show pipeline statistics")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_rag()
    elif args.question:
        # Single question mode
        model_size = ModelSize(args.model_size)
        pipeline = RAGPipeline(model_size=model_size)
        
        print("Initializing pipeline...")
        if not pipeline.initialize_generator():
            print("❌ Failed to initialize generator")
            return
        
        print("✅ Processing question...")
        result = pipeline.query(
            question=args.question,
            n_results=args.n_results,
            retrieval_strategy=args.retrieval_strategy,
            query_type=args.query_type,
            generation_style=args.generation_style
        )
        
        print_rag_result(result)
        
        # Save result
        output_file = f"rag_result_{int(time.time())}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Result saved to: {output_file}")
        
    elif args.stats:
        # Stats mode
        pipeline = RAGPipeline()
        stats = pipeline.get_pipeline_stats()
        print("\n📊 RAG Pipeline Statistics:")
        print(json.dumps(stats, indent=2, default=str))
        
    else:
        print("Use --interactive, --question, or --stats to use the system")


if __name__ == "__main__":
    main()