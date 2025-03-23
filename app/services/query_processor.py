"""Query processing service for Vanna AI Repository Q&A System."""

import logging
import time
from typing import Dict, Any, List, Optional
from app.services.retrieval import VectorStore, EnhancedVectorStore
from app.services.llm_service import LLMService
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processor for handling user queries."""
    
    def __init__(self, config, vector_store: VectorStore, llm_service: LLMService):
        """Initialize the query processor."""
        self.config = config
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.retrieval_k = config.get("retrieval_k", 5)
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query end-to-end."""
        start_time = time.time()
        logger.info(f"Processing query: {query}")
        
        try:
            # 1. Check if query is in scope
            scope_check = self.llm_service.check_query_scope(query)
            
            if not scope_check.get('in_scope', True):
                # Generate out-of-scope response
                response = self.llm_service.generate_out_of_scope_response(query)
                
                return {
                    "query": query,
                    "answer": response.get('answer'),
                    "in_scope": False,
                    "sources": [],
                    "total_processing_time": time.time() - start_time,
                    "scope_check_time": scope_check.get('processing_time', 0),
                    "retrieval_time": 0,
                    "generation_time": response.get('processing_time', 0)
                }
            
            # 2. Retrieve relevant documents for in-scope queries
            retrieval_start = time.time()
            relevant_docs = self.vector_store.similarity_search(query, k=self.retrieval_k)
            retrieval_time = time.time() - retrieval_start
            
            # 3. Generate an answer
            response = self.llm_service.generate_answer(query, relevant_docs)
            
            # 4. Format and return the complete response
            return {
                "query": query,
                "answer": response.get('answer'),
                "in_scope": True,
                "sources": response.get('sources', []),
                "total_processing_time": time.time() - start_time,
                "scope_check_time": scope_check.get('processing_time', 0),
                "retrieval_time": retrieval_time,
                "generation_time": response.get('processing_time', 0)
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "in_scope": True,
                "sources": [],
                "total_processing_time": time.time() - start_time,
                "error": str(e)
            }


class EnhancedQueryProcessor(QueryProcessor):
    """Enhanced query processor with improved techniques."""
    
    def __init__(self, config, vector_store: VectorStore, llm_service: LLMService):
        super().__init__(config, vector_store, llm_service)
        self.use_enhanced_retrieval = True
    
    def expand_query(self, original_query: str) -> List[str]:
        """Expand a query into multiple related queries."""
        expansion_prompt = f"""Given the following user question about the Vanna AI repository, generate 3 different reformulations of the question that might help retrieve additional relevant information. Make the reformulations diverse in their approach and vocabulary, while maintaining the core intent of the original question.

Original question: {original_query}

Generate 3 reformulations separated by triple dashes (---):"""
        
        try:
            # Create messages directly
            messages = [
                SystemMessage(content="You are an AI assistant that helps reformulate questions to improve information retrieval."),
                HumanMessage(content=expansion_prompt)
            ]
            
            # Get response
            response = self.llm_service.llm.invoke(messages)
            response_text = response.content
            
            # Split by the triple dash separator
            reformulations = [q.strip() for q in response_text.split("---")]
            
            # Clean up the reformulations
            clean_reformulations = []
            for q in reformulations:
                # Remove numbering if present (e.g., "1. Question" -> "Question")
                if q and len(q.strip()) > 0:
                    # Check if it starts with a number followed by a period
                    if q.strip()[0].isdigit() and ". " in q[:5]:
                        q = q.split(". ", 1)[1]
                    clean_reformulations.append(q.strip())
            
            # Add the original query
            all_queries = [original_query] + clean_reformulations
            
            # Remove duplicates and empty strings
            all_queries = [q for q in all_queries if q]
            all_queries = list(dict.fromkeys(all_queries))
            
            logger.info(f"Expanded query '{original_query}' into {len(all_queries)} queries")
            return all_queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            # Return just the original query on error
            return [original_query]
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query with enhanced techniques."""
        start_time = time.time()
        logger.info(f"Processing query with enhancements: {query}")
        
        try:
            # 1. Check if query is in scope
            scope_check = self.llm_service.check_query_scope(query)
            
            if not scope_check.get('in_scope', True):
                # Generate out-of-scope response
                response = self.llm_service.generate_out_of_scope_response(query)
                
                return {
                    "query": query,
                    "answer": response.get('answer'),
                    "in_scope": False,
                    "sources": [],
                    "total_processing_time": time.time() - start_time,
                    "scope_check_time": scope_check.get('processing_time', 0),
                    "retrieval_time": 0,
                    "generation_time": response.get('processing_time', 0)
                }
            
            # 2. Expand the query into multiple variations
            retrieval_start = time.time()
            expanded_queries = self.expand_query(query)
            
            # 3. Retrieve documents for each query variation
            all_docs = []
            for expanded_query in expanded_queries:
                docs = self.vector_store.similarity_search(expanded_query, k=self.retrieval_k)
                all_docs.extend(docs)
            
            # 4. Remove duplicates and re-rank
            unique_docs = {}
            for doc in all_docs:
                content = doc.get('content', '')
                # Skip empty docs
                if not content:
                    continue
                
                # Use content as key for deduplication
                if content not in unique_docs:
                    unique_docs[content] = doc
                else:
                    # If we've seen this doc before, keep the one with the better score
                    existing_score = unique_docs[content].get('similarity_score', 0)
                    new_score = doc.get('similarity_score', 0)
                    if new_score > existing_score:
                        unique_docs[content] = doc
            
            # Convert back to list and sort by similarity score
            relevant_docs = list(unique_docs.values())
            relevant_docs.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            
            # Limit to top results
            relevant_docs = relevant_docs[:self.retrieval_k]
            retrieval_time = time.time() - retrieval_start
            
            # 5. Generate an answer with GitHub references
            response = self.llm_service.generate_answer_with_references(query, relevant_docs)
            
            # 6. Format and return the complete response
            return {
                "query": query,
                "answer": response.get('answer'),
                "in_scope": True,
                "sources": [r.get('github_url') for r in response.get('references', [])],
                "references": response.get('references', []),
                "total_processing_time": time.time() - start_time,
                "scope_check_time": scope_check.get('processing_time', 0),
                "retrieval_time": retrieval_time,
                "generation_time": response.get('processing_time', 0),
                "expanded_queries": expanded_queries
            }
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "in_scope": True,
                "sources": [],
                "total_processing_time": time.time() - start_time,
                "error": str(e)
            }
