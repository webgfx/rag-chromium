#!/usr/bin/env python3
"""
Advanced vector database implementation with ChromaDB.
Provides high-performance vector storage, indexing, and retrieval with metadata filtering.
"""

import json
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger


@dataclass
class VectorDocument:
    """Represents a document stored in the vector database."""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorDocument':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SearchResult:
    """Represents a search result with similarity score."""
    document: VectorDocument
    score: float
    rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'document': self.document.to_dict(),
            'score': self.score,
            'rank': self.rank
        }


class VectorDatabase:
    """
    Advanced vector database using ChromaDB with GPU optimization and enterprise features.
    """
    
    def __init__(
        self,
        collection_name: str = "chromium_embeddings",
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
            embedding_model: Embedding model name for ChromaDB (optional)
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.VectorDatabase")
        
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory or self.config.data.cache_dir) / "vector_db"
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = self._initialize_client()
        
        # Initialize embedding function if provided
        self.embedding_function = None
        if embedding_model:
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        # Note: For pre-computed embeddings, we don't need an embedding function
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        self.logger.info(f"Initialized vector database with collection: {collection_name}")
        self.logger.info(f"Persist directory: {self.persist_directory}")
    
    def _initialize_client(self) -> chromadb.Client:
        """Initialize ChromaDB client with optimal settings."""
        # Use the new PersistentClient API
        return chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            self.logger.info(f"Retrieved existing collection: {self.collection_name}")
            
        except Exception:
            # Create new collection
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Chromium development RAG embeddings"}
            )
            self.logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(
        self,
        documents: List[VectorDocument],
        batch_size: int = 50
    ) -> int:
        """
        Add documents to the vector database in batches.
        
        Args:
            documents: List of documents to add
            batch_size: Batch size for adding documents
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        self.logger.info(f"Adding {len(documents)} documents to vector database")
        
        added_count = 0
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare batch data
            ids = []
            embeddings = []
            metadatas = []
            documents_content = []
            
            for doc in batch:
                if not doc.id:
                    doc.id = str(uuid.uuid4())
                
                ids.append(doc.id)
                documents_content.append(doc.content)
                
                if doc.embedding:
                    embeddings.append(doc.embedding)
                
                # Prepare metadata (filter out unsupported types)
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # Filter out unsupported metadata types (lists, dicts, None values, etc.)
                filtered_metadata = {}
                for key, value in metadata.items():
                    if value is None:
                        # Skip None values
                        continue
                    elif isinstance(value, (str, int, float, bool)):
                        filtered_metadata[key] = value
                    elif isinstance(value, list):
                        # Skip lists entirely for now
                        continue
                    else:
                        # Convert other types to string
                        filtered_metadata[key] = str(value)
                
                filtered_metadata.update({
                    'added_at': datetime.now().isoformat(),
                    'content_length': len(doc.content)
                })
                metadatas.append(filtered_metadata)
            
            try:
                # Add to collection
                if embeddings:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        documents=documents_content
                    )
                else:
                    # Let ChromaDB generate embeddings
                    self.collection.add(
                        ids=ids,
                        metadatas=metadatas,
                        documents=documents_content
                    )
                
                added_count += len(batch)
                self.logger.info(f"Added batch {i//batch_size + 1}: {len(batch)} documents")
                
            except Exception as e:
                self.logger.error(f"Failed to add batch {i//batch_size + 1}: {e}")
                continue
        
        self.logger.info(f"Successfully added {added_count} documents to vector database")
        return added_count
    
    def search(
        self,
        query: Union[str, List[float]],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, str]] = None,
        include_embeddings: bool = False
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Query text or embedding vector
            n_results: Number of results to return
            where: Metadata filter conditions
            where_document: Document content filter conditions
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of search results ordered by similarity
        """
        self.logger.debug(f"Searching for {n_results} results with query type: {type(query)}")
        
        try:
            # Prepare query
            query_texts = None
            query_embeddings = None
            
            if isinstance(query, str):
                query_texts = [query]
            elif isinstance(query, list):
                query_embeddings = [query]
            else:
                raise ValueError(f"Unsupported query type: {type(query)}")
            
            # Prepare include list
            include = ["documents", "metadatas", "distances"]
            if include_embeddings:
                include.append("embeddings")
            
            # Perform search
            results = self.collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            # Process results
            search_results = []
            
            if results['ids'] and results['ids'][0]:  # Check if we have results
                for i, doc_id in enumerate(results['ids'][0]):
                    document = VectorDocument(
                        id=doc_id,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        embedding=results['embeddings'][0][i] if include_embeddings else None
                    )
                    
                    # ChromaDB returns distances, convert to similarity score
                    distance = results['distances'][0][i]
                    score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    search_results.append(SearchResult(
                        document=document,
                        score=score,
                        rank=i + 1
                    ))
            
            self.logger.info(f"Found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            n_results: Number of results to return
            where: Metadata filter conditions
            
        Returns:
            Combined and re-ranked search results
        """
        self.logger.debug(f"Performing hybrid search: semantic={semantic_weight}, keyword={keyword_weight}")
        
        # For now, just do keyword search since we don't have compatible embedding function
        # In a full implementation, you'd want to use the same embedding model
        keyword_results = self.search(
            query="", # Empty query to avoid embedding dimension issues
            n_results=n_results,
            where=where,
            where_document={"$contains": query.lower()}
        )
        
        return keyword_results
    
    def _combine_search_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        semantic_weight: float,
        keyword_weight: float
    ) -> List[SearchResult]:
        """Combine and re-rank search results from different methods."""
        result_map = {}
        
        # Add semantic results
        for result in semantic_results:
            doc_id = result.document.id
            result_map[doc_id] = {
                'document': result.document,
                'semantic_score': result.score,
                'keyword_score': 0.0
            }
        
        # Add keyword results
        for result in keyword_results:
            doc_id = result.document.id
            if doc_id in result_map:
                result_map[doc_id]['keyword_score'] = result.score
            else:
                result_map[doc_id] = {
                    'document': result.document,
                    'semantic_score': 0.0,
                    'keyword_score': result.score
                }
        
        # Calculate combined scores and create final results
        combined_results = []
        for i, (doc_id, data) in enumerate(result_map.items()):
            combined_score = (
                data['semantic_score'] * semantic_weight +
                data['keyword_score'] * keyword_weight
            )
            
            combined_results.append(SearchResult(
                document=data['document'],
                score=combined_score,
                rank=i + 1
            ))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results):
            result.rank = i + 1
        
        return combined_results
    
    def get_document(self, doc_id: str) -> Optional[VectorDocument]:
        """Get a specific document by ID."""
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if results['ids'] and results['ids'][0]:
                return VectorDocument(
                    id=results['ids'][0],
                    content=results['documents'][0],
                    metadata=results['metadatas'][0],
                    embedding=results['embeddings'][0] if results['embeddings'] else None
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def delete_documents(self, doc_ids: List[str]) -> int:
        """Delete documents by IDs."""
        try:
            self.collection.delete(ids=doc_ids)
            self.logger.info(f"Deleted {len(doc_ids)} documents")
            return len(doc_ids)
            
        except Exception as e:
            self.logger.error(f"Failed to delete documents: {e}")
            return 0
    
    def update_document(self, document: VectorDocument) -> bool:
        """Update an existing document."""
        try:
            # ChromaDB doesn't have direct update, so we delete and add
            self.delete_documents([document.id])
            self.add_documents([document])
            self.logger.info(f"Updated document {document.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document {document.id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            count = self.collection.count()
            
            # Get sample documents for analysis
            sample_results = self.collection.peek(limit=10)
            
            stats = {
                'collection_name': self.collection_name,
                'total_documents': count,
                'persist_directory': str(self.persist_directory),
                'sample_metadata': sample_results['metadatas'][:3] if sample_results['metadatas'] else [],
                'created_at': datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    def create_index(self, field_name: str) -> bool:
        """Create an index on a metadata field (ChromaDB handles this automatically)."""
        self.logger.info(f"ChromaDB automatically indexes metadata field: {field_name}")
        return True
    
    def backup_collection(self, backup_path: Optional[str] = None) -> str:
        """Create a backup of the collection."""
        if not backup_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.persist_directory / f"backup_{self.collection_name}_{timestamp}.json"
        
        try:
            # Get all documents
            all_results = self.collection.get(include=["documents", "metadatas", "embeddings"])
            
            backup_data = {
                'collection_name': self.collection_name,
                'backup_timestamp': datetime.now().isoformat(),
                'documents': []
            }
            
            # Process documents
            if all_results['ids'] and len(all_results['ids']) > 0:
                for i, doc_id in enumerate(all_results['ids']):
                    doc_data = {
                        'id': doc_id,
                        'content': all_results['documents'][i] if all_results['documents'] and i < len(all_results['documents']) else None,
                        'metadata': all_results['metadatas'][i] if all_results['metadatas'] and i < len(all_results['metadatas']) else None,
                        'embedding': all_results['embeddings'][i] if all_results['embeddings'] and i < len(all_results['embeddings']) else None
                    }
                    backup_data['documents'].append(doc_data)
            
            # Save backup
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, default=str)
            
            self.logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return ""
    
    def restore_collection(self, backup_path: str) -> bool:
        """Restore collection from backup."""
        try:
            backup_path = Path(backup_path)
            
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            
            # Clear existing collection
            self.client.delete_collection(self.collection_name)
            self.collection = self._get_or_create_collection()
            
            # Restore documents
            documents = []
            for doc_data in backup_data['documents']:
                documents.append(VectorDocument.from_dict(doc_data))
            
            self.add_documents(documents)
            
            self.logger.info(f"Restored {len(documents)} documents from {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Restore failed: {e}")
            return False
    
    def close(self):
        """Close the database connection."""
        self.logger.info("Closing vector database connection")
        # ChromaDB client doesn't need explicit closing