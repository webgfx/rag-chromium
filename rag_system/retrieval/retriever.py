#!/usr/bin/env python3
"""
Advanced retrieval system for RAG with query expansion, re-ranking, and context-aware filtering.
Provides sophisticated retrieval capabilities specifically designed for Chromium development queries.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.vector.database import VectorDatabase, SearchResult
from rag_system.embeddings.generator import EmbeddingGenerator


@dataclass
class RetrievalQuery:
    """Represents a structured retrieval query."""
    original_query: str
    expanded_queries: List[str] = field(default_factory=list)
    filters: Dict[str, Any] = field(default_factory=dict)
    boost_terms: List[str] = field(default_factory=list)
    context: Optional[str] = None
    query_type: str = "general"  # general, bug_fix, feature, performance, security
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'original_query': self.original_query,
            'expanded_queries': self.expanded_queries,
            'filters': self.filters,
            'boost_terms': self.boost_terms,
            'context': self.context,
            'query_type': self.query_type
        }


@dataclass 
class RetrievalResult:
    """Enhanced search result with retrieval metadata."""
    search_result: SearchResult
    retrieval_score: float
    relevance_signals: Dict[str, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'search_result': self.search_result.to_dict(),
            'retrieval_score': self.retrieval_score,
            'relevance_signals': self.relevance_signals,
            'explanation': self.explanation
        }


class ChromiumQueryProcessor:
    """Processes and expands queries specifically for Chromium development context."""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.ChromiumQueryProcessor")
        
        # Chromium-specific terminology and expansions
        self.chromium_terms = {
            'bug': ['crash', 'error', 'exception', 'failure', 'issue', 'problem'],
            'performance': ['optimization', 'speed', 'latency', 'memory', 'cpu', 'gpu', 'efficiency'],
            'security': ['vulnerability', 'exploit', 'cve', 'patch', 'fix', 'sanitizer'],
            'rendering': ['paint', 'composite', 'draw', 'graphics', 'display', 'viewport'],
            'ui': ['interface', 'ux', 'design', 'layout', 'style', 'css', 'html'],
            'network': ['http', 'https', 'fetch', 'request', 'response', 'connection'],
            'storage': ['database', 'cache', 'indexeddb', 'localstorage', 'file'],
            'javascript': ['js', 'v8', 'engine', 'runtime', 'execution', 'compilation'],
            'memory': ['leak', 'allocation', 'deallocation', 'heap', 'stack', 'oom'],
            'threading': ['thread', 'async', 'sync', 'parallel', 'concurrent', 'lock']
        }
        
        # Common Chromium file patterns and components
        self.chromium_components = {
            'blink': ['renderer', 'webkit', 'dom', 'css', 'html'],
            'content': ['browser', 'renderer', 'process', 'sandbox'],
            'chrome': ['browser', 'ui', 'app', 'service'],
            'net': ['network', 'http', 'socket', 'ssl', 'quic'],
            'gpu': ['graphics', 'opengl', 'vulkan', 'directx', 'metal'],
            'media': ['audio', 'video', 'codec', 'streaming', 'webrtc'],
            'extensions': ['addon', 'plugin', 'api', 'manifest'],
            'devtools': ['inspector', 'debug', 'profiler', 'console']
        }
        
        # Query pattern recognition
        self.query_patterns = {
            'bug_fix': [
                r'\b(bug|crash|error|exception|failure|fix|patch)\b',
                r'\b(cve-\d+|security|vulnerability)\b',
                r'\b(memory leak|null pointer|segfault)\b'
            ],
            'feature': [
                r'\b(add|implement|feature|functionality|support)\b',
                r'\b(new|create|introduce|enable)\b'
            ],
            'performance': [
                r'\b(optimize|performance|speed|fast|slow|latency)\b',
                r'\b(memory|cpu|gpu|efficiency|throughput)\b'
            ],
            'refactor': [
                r'\b(refactor|cleanup|reorganize|restructure)\b',
                r'\b(rename|move|extract|split)\b'
            ]
        }
    
    def classify_query(self, query: str) -> str:
        """Classify query type based on content."""
        query_lower = query.lower()
        
        for query_type, patterns in self.query_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return query_type
        
        return "general"
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with Chromium-specific terms and synonyms."""
        expanded = [query]  # Include original query
        query_lower = query.lower()
        
        # Add expanded terms based on Chromium terminology
        for category, synonyms in self.chromium_terms.items():
            if category in query_lower:
                for synonym in synonyms[:3]:  # Limit to top 3 synonyms
                    expanded_query = re.sub(
                        r'\b' + re.escape(category) + r'\b',
                        synonym,
                        query_lower
                    )
                    if expanded_query != query_lower:
                        expanded.append(expanded_query)
        
        # Add component-specific expansions
        for component, related_terms in self.chromium_components.items():
            if component in query_lower:
                for term in related_terms[:2]:  # Limit to top 2 terms
                    expanded.append(f"{query} {term}")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expanded = []
        for item in expanded:
            if item not in seen:
                seen.add(item)
                unique_expanded.append(item)
        
        self.logger.debug(f"Expanded '{query}' to {len(unique_expanded)} variants")
        return unique_expanded
    
    def extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract metadata filters from query."""
        filters = {}
        query_lower = query.lower()
        
        # File type filters - ChromaDB doesn't support $contains, so we'll skip for now
        # In a full implementation, you'd pre-process this or use where_document
        
        # Component filters - skip for now due to ChromaDB limitations
        # We'll handle this in the where_document parameter instead
        
        # Author filters
        author_match = re.search(r'by\s+([a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+)', query_lower)
        if author_match:
            filters['author_email'] = author_match.group(1)
        
        # Time filters
        time_patterns = {
            'last week': 7,
            'last month': 30,
            'last year': 365
        }
        
        for pattern, days in time_patterns.items():
            if pattern in query_lower:
                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
                filters['commit_date'] = {'$gte': cutoff_date}
                break
        
        return filters
    
    def extract_boost_terms(self, query: str) -> List[str]:
        """Extract terms that should be boosted in retrieval."""
        boost_terms = []
        query_lower = query.lower()
        
        # Technical terms that should be boosted
        technical_terms = [
            'memory leak', 'null pointer', 'segfault', 'crash',
            'performance', 'optimization', 'security', 'vulnerability',
            'regression', 'fix', 'patch', 'update'
        ]
        
        for term in technical_terms:
            if term in query_lower:
                boost_terms.append(term)
        
        return boost_terms


class AdvancedRetriever:
    """
    Advanced retrieval system with multiple retrieval strategies and re-ranking.
    """
    
    def __init__(
        self,
        vector_db: VectorDatabase,
        embedding_generator: EmbeddingGenerator,
        collection_name: str = "chromium_embeddings"
    ):
        """
        Initialize the advanced retriever.
        
        Args:
            vector_db: Vector database instance
            embedding_generator: Embedding generator for query encoding
            collection_name: Name of the collection to search
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.AdvancedRetriever")
        
        self.vector_db = vector_db
        self.embedding_generator = embedding_generator
        self.collection_name = collection_name
        
        # Initialize query processor
        self.query_processor = ChromiumQueryProcessor()
        
        # Initialize TF-IDF for keyword matching
        self.tfidf_vectorizer = None
        self.document_vectors = None
        self._initialize_tfidf()
        
        self.logger.info("Initialized advanced retriever")
    
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer with document corpus."""
        try:
            # Get sample documents for TF-IDF initialization
            stats = self.vector_db.get_collection_stats()
            total_docs = stats.get('total_documents', 0)
            
            if total_docs > 0:
                # Sample documents for TF-IDF (limit to 1000 for performance)
                sample_size = min(1000, total_docs)
                
                # This is a simplified approach - in production, you'd want to
                # build TF-IDF from all documents or a representative sample
                self.logger.info(f"TF-IDF initialization skipped - would need {sample_size} documents")
                
        except Exception as e:
            self.logger.warning(f"Could not initialize TF-IDF: {e}")
    
    def process_query(self, query: str, context: Optional[str] = None) -> RetrievalQuery:
        """Process and enhance a raw query."""
        # Classify query type
        query_type = self.query_processor.classify_query(query)
        
        # Expand query
        expanded_queries = self.query_processor.expand_query(query)
        
        # Extract filters
        filters = self.query_processor.extract_filters(query)
        
        # Extract boost terms
        boost_terms = self.query_processor.extract_boost_terms(query)
        
        retrieval_query = RetrievalQuery(
            original_query=query,
            expanded_queries=expanded_queries,
            filters=filters,
            boost_terms=boost_terms,
            context=context,
            query_type=query_type
        )
        
        self.logger.info(f"Processed query: type={query_type}, "
                        f"expansions={len(expanded_queries)}, "
                        f"filters={len(filters)}")
        
        return retrieval_query
    
    def retrieve(
        self,
        query: str,
        n_results: int = 10,
        context: Optional[str] = None,
        use_reranking: bool = True,
        retrieval_strategy: str = "hybrid"  # semantic, hybrid, multi_stage
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents using advanced strategies.
        
        Args:
            query: Search query
            n_results: Number of results to return
            context: Additional context for the query  
            use_reranking: Whether to apply re-ranking
            retrieval_strategy: Strategy to use for retrieval
            
        Returns:
            List of enhanced retrieval results
        """
        self.logger.info(f"Retrieving with strategy: {retrieval_strategy}")
        
        # Process query
        retrieval_query = self.process_query(query, context)
        
        # Apply retrieval strategy
        if retrieval_strategy == "semantic":
            results = self._semantic_retrieval(retrieval_query, n_results * 2)
        elif retrieval_strategy == "hybrid":
            results = self._hybrid_retrieval(retrieval_query, n_results * 2)
        elif retrieval_strategy == "multi_stage":
            results = self._multi_stage_retrieval(retrieval_query, n_results * 2)
        else:
            raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")
        
        # Convert to RetrievalResults
        retrieval_results = []
        for i, result in enumerate(results):
            retrieval_result = RetrievalResult(
                search_result=result,
                retrieval_score=result.score,
                relevance_signals={'original_rank': i + 1}
            )
            retrieval_results.append(retrieval_result)
        
        # Apply re-ranking if requested
        if use_reranking and len(retrieval_results) > 1:
            retrieval_results = self._rerank_results(
                retrieval_query, retrieval_results
            )
        
        # Limit to requested number of results
        final_results = retrieval_results[:n_results]
        
        # Add explanations
        for i, result in enumerate(final_results):
            result.explanation = self._generate_explanation(
                retrieval_query, result, i + 1
            )
        
        self.logger.info(f"Retrieved {len(final_results)} results")
        return final_results
    
    def _semantic_retrieval(
        self, 
        retrieval_query: RetrievalQuery, 
        n_results: int
    ) -> List[SearchResult]:
        """Perform semantic retrieval using embeddings."""
        # Generate embedding for the original query
        query_embeddings = self.embedding_generator.encode_texts([retrieval_query.original_query])
        query_embedding = query_embeddings[0].tolist()
        
        # Search using embedding
        results = self.vector_db.search(
            query=query_embedding,
            n_results=n_results,
            where=retrieval_query.filters or None
        )
        
        return results
    
    def _hybrid_retrieval(
        self,
        retrieval_query: RetrievalQuery,
        n_results: int
    ) -> List[SearchResult]:
        """Perform hybrid retrieval combining semantic and keyword matching."""
        # Get semantic results using our embedding generator
        semantic_results = self._semantic_retrieval(retrieval_query, n_results)
        
        # Get keyword results using document content filtering  
        keyword_results = self.vector_db.search(
            query="",  # Empty query to avoid embedding issues
            n_results=n_results,
            where=retrieval_query.filters or None,
            where_document={"$contains": retrieval_query.original_query.lower()}
        )
        
        # Simple combination - in practice you'd want more sophisticated merging
        all_results = semantic_results + keyword_results
        
        # Remove duplicates by ID
        seen_ids = set()
        unique_results = []
        for result in all_results:
            if result.document.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.document.id)
        
        # Sort by score and limit
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:n_results]
    
    def _multi_stage_retrieval(
        self,
        retrieval_query: RetrievalQuery,
        n_results: int
    ) -> List[SearchResult]:
        """Perform multi-stage retrieval with query expansion."""
        all_results = []
        seen_ids = set()
        
        # Stage 1: Original query
        original_results = self._semantic_retrieval(retrieval_query, n_results // 2)
        for result in original_results:
            if result.document.id not in seen_ids:
                all_results.append(result)
                seen_ids.add(result.document.id)
        
        # Stage 2: Expanded queries
        for expanded_query in retrieval_query.expanded_queries[1:3]:  # Limit to 2 expansions
            temp_query = RetrievalQuery(
                original_query=expanded_query,
                filters=retrieval_query.filters
            )
            expanded_results = self._semantic_retrieval(temp_query, n_results // 4)
            
            for result in expanded_results:
                if result.document.id not in seen_ids and len(all_results) < n_results:
                    # Adjust score for expanded queries
                    result.score *= 0.9  # Slight penalty for expanded queries
                    all_results.append(result)
                    seen_ids.add(result.document.id)
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        
        return all_results[:n_results]
    
    def _rerank_results(
        self,
        retrieval_query: RetrievalQuery,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Re-rank results using multiple signals."""
        for result in results:
            # Calculate relevance signals
            signals = self._calculate_relevance_signals(retrieval_query, result)
            result.relevance_signals.update(signals)
            
            # Calculate combined score
            result.retrieval_score = self._calculate_combined_score(result)
        
        # Sort by retrieval score
        results.sort(key=lambda x: x.retrieval_score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(results):
            result.relevance_signals['final_rank'] = i + 1
        
        return results
    
    def _calculate_relevance_signals(
        self,
        retrieval_query: RetrievalQuery,
        result: RetrievalResult
    ) -> Dict[str, float]:
        """Calculate various relevance signals."""
        signals = {}
        
        document = result.search_result.document
        content = document.content.lower()
        query_lower = retrieval_query.original_query.lower()
        metadata = document.metadata or {}
        
        # Text matching signals
        signals['exact_match'] = 1.0 if query_lower in content else 0.0
        signals['term_frequency'] = len(re.findall(re.escape(query_lower), content))
        
        # Boost term matching
        boost_score = 0.0
        for term in retrieval_query.boost_terms:
            if term.lower() in content:
                boost_score += 1.0
        signals['boost_terms'] = boost_score / max(len(retrieval_query.boost_terms), 1)
        
        # Query type specific signals
        if retrieval_query.query_type == "bug_fix":
            bug_terms = ['fix', 'bug', 'crash', 'error', 'patch']
            bug_score = sum(1 for term in bug_terms if term in content)
            signals['bug_relevance'] = bug_score / len(bug_terms)
        
        # Metadata signals
        signals['content_length'] = min(len(content) / 1000, 1.0)  # Normalize to [0,1]
        signals['has_file_path'] = 1.0 if metadata.get('file_path') else 0.0
        
        # Recent commit bonus
        commit_date = metadata.get('commit_date', '')
        if commit_date:
            try:
                commit_dt = datetime.fromisoformat(commit_date.replace('Z', '+00:00'))
                days_old = (datetime.now() - commit_dt.replace(tzinfo=None)).days
                signals['recency'] = max(0, 1.0 - (days_old / 365))  # Decay over a year
            except:
                signals['recency'] = 0.0
        else:
            signals['recency'] = 0.0
        
        return signals
    
    def _calculate_combined_score(self, result: RetrievalResult) -> float:
        """Calculate combined relevance score."""
        signals = result.relevance_signals
        original_score = result.search_result.score
        
        # Weight configuration
        weights = {
            'semantic': 0.4,
            'exact_match': 0.2,
            'boost_terms': 0.15,
            'bug_relevance': 0.1,
            'recency': 0.1,
            'content_length': 0.05
        }
        
        combined_score = (
            original_score * weights['semantic'] +
            signals.get('exact_match', 0) * weights['exact_match'] +
            signals.get('boost_terms', 0) * weights['boost_terms'] +
            signals.get('bug_relevance', 0) * weights['bug_relevance'] +
            signals.get('recency', 0) * weights['recency'] +
            signals.get('content_length', 0) * weights['content_length']
        )
        
        return combined_score
    
    def _generate_explanation(
        self,
        retrieval_query: RetrievalQuery,
        result: RetrievalResult,
        rank: int
    ) -> List[str]:
        """Generate explanation for why this result was retrieved."""
        explanations = []
        
        signals = result.relevance_signals
        
        # Primary relevance
        explanations.append(f"Ranked #{rank} with semantic similarity score: {result.search_result.score:.3f}")
        
        # Specific matching signals
        if signals.get('exact_match', 0) > 0:
            explanations.append("Contains exact query match")
        
        if signals.get('boost_terms', 0) > 0:
            explanations.append(f"Matches {len(retrieval_query.boost_terms)} important terms")
        
        if signals.get('bug_relevance', 0) > 0.5:
            explanations.append("High relevance for bug-related query")
        
        if signals.get('recency', 0) > 0.7:
            explanations.append("Recent commit (recency bonus)")
        
        # Metadata info
        metadata = result.search_result.document.metadata or {}
        if metadata.get('file_path'):
            explanations.append(f"File: {metadata['file_path']}")
        
        if metadata.get('chunk_type'):
            explanations.append(f"Content type: {metadata['chunk_type']}")
        
        return explanations
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics."""
        vector_stats = self.vector_db.get_collection_stats()
        
        stats = {
            'collection_stats': vector_stats,
            'query_processor': {
                'chromium_terms': len(self.query_processor.chromium_terms),
                'components': len(self.query_processor.chromium_components),
                'patterns': len(self.query_processor.query_patterns)
            },
            'retrieval_strategies': ['semantic', 'hybrid', 'multi_stage'],
            'reranking_signals': [
                'exact_match', 'term_frequency', 'boost_terms',
                'bug_relevance', 'content_length', 'recency'
            ]
        }
        
        return stats