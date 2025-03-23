"""Retrieval service for Vanna AI Repository Q&A System using Qdrant."""

import os
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores.qdrant import Qdrant
import qdrant_client

logger = logging.getLogger(__name__)


class VectorStore:
    """Vector store for document retrieval using Qdrant."""
    
    def __init__(self, config):
        """Initialize the vector store with Qdrant."""
        self.config = config
        self.persist_directory = config.get("qdrant_persist_directory", "./data/vector_store")
        
        # Ensure the directory exists
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.collection_name = config.get("qdrant_collection_name", "vanna_repo")
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            deployment=config.get("azure_embedding_deployment"),
            openai_api_version=config.get("azure_api_version"),
            azure_endpoint=config.get("azure_endpoint"),
            api_key=config.get("azure_api_key"),
        )
        
        # Initialize Qdrant
        self._init_qdrant()
    
    def _init_qdrant(self):
        """Initialize the Qdrant client and collection."""
        try:
            # Initialize the Qdrant client with local persistence
            self.qdrant_client = qdrant_client.QdrantClient(
                path=self.persist_directory
            )
            
            # Check if collection exists and create it if it doesn't
            collections = self.qdrant_client.get_collections().collections
            collection_exists = any(collection.name == self.collection_name for collection in collections)
            
            if not collection_exists:
                # Get vector size from the embeddings model
                # The OpenAI embedding model uses 1536 dimensions
                vector_size = 1536
                
                # Create the collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qdrant_client.models.VectorParams(
                        size=vector_size,
                        distance=qdrant_client.models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection '{self.collection_name}'")
            else:
                logger.info(f"Loaded existing collection '{self.collection_name}'")
            
            # Initialize LangChain's Qdrant wrapper
            self.langchain_qdrant = Qdrant(
                client=self.qdrant_client, 
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
            
        except Exception as e:
            logger.error(f"Error initializing Qdrant: {e}")
            raise
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Add document chunks to the vector store."""
        if not chunks:
            logger.warning("No chunks to add to vector store")
            return
        
        start_time = time.time()
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        try:
            # Extract texts and metadata
            texts = []
            metadatas = []
            
            for chunk in chunks:
                # Skip chunks without content
                if not chunk.get('content'):
                    continue
                
                texts.append(chunk['content'])
                metadatas.append(chunk['metadata'])
            
            # Add documents to Qdrant using LangChain's wrapper
            self.langchain_qdrant.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(texts)} chunks to vector store in {time.time() - start_time:.2f} seconds")
        
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector store."""
        start_time = time.time()
        logger.info(f"Searching for documents similar to query: {query}")
        
        try:
            # Use LangChain's Qdrant wrapper for similarity search
            results = self.langchain_qdrant.similarity_search_with_score(
                query=query,
                k=k
            )
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'similarity_score': score
                })
            
            logger.info(f"Found {len(formatted_results)} similar documents in {time.time() - start_time:.2f} seconds")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            point_count = collection_info.points_count
            
            return {
                "document_count": point_count,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
            return {"error": str(e)}


class EnhancedVectorStore(VectorStore):
    """Enhanced vector store with improved retrieval techniques."""
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining different techniques."""
        start_time = time.time()
        logger.info(f"Performing hybrid search for query: {query}")
        
        try:
            # 1. Get regular search results
            regular_results = self.similarity_search(query, k=k)
            
            # 2. Get results with additional query context
            expanded_query = f"Question about Vanna AI: {query}"
            expanded_results = self.similarity_search(expanded_query, k=k)
            
            # 3. Combine and deduplicate results
            all_results = regular_results + expanded_results
            unique_results = {}
            
            for result in all_results:
                content = result.get('content', '')
                if content not in unique_results:
                    unique_results[content] = result
                else:
                    # Keep the one with better score
                    if result['similarity_score'] < unique_results[content]['similarity_score']:
                        unique_results[content] = result
            
            # 4. Sort by score and limit to k results
            combined_results = list(unique_results.values())
            combined_results.sort(key=lambda r: r['similarity_score'])
            top_results = combined_results[:k]
            
            logger.info(f"Hybrid search found {len(top_results)} documents in {time.time() - start_time:.2f} seconds")
            return top_results
            
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            # Fall back to regular search
            return self.similarity_search(query, k)
