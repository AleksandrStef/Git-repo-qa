"""LLM service for Vanna AI Repository Q&A System."""

import logging
import time
from typing import Dict, Any, List, Optional
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)


class LLMService:
    """LLM service for generating answers and checking query scope."""
    
    def __init__(self, config):
        """Initialize the LLM service with Azure OpenAI."""
        self.config = config
        
        # Initialize Azure OpenAI
        self.llm = AzureChatOpenAI(
            azure_deployment=config.get("azure_llm_deployment"),
            openai_api_version=config.get("azure_api_version"),
            azure_endpoint=config.get("azure_endpoint"),
            api_key=config.get("azure_api_key"),
            temperature=0.1  # Low temperature for more deterministic responses
        )
        
        # Initialize prompt templates
        self._init_prompt_templates()
    
    def _init_prompt_templates(self):
        """Initialize prompt templates for different tasks."""
        # System message for in-scope questions
        self.in_scope_system_template = """You are an AI assistant specialized in answering questions about the Vanna AI repository (https://github.com/vanna-ai/vanna). 
        Vanna is an open-source Python RAG (Retrieval-Augmented Generation) framework for SQL generation.
        
        Given context information about the repository, answer the user's question accurately and concisely.
        If the answer is in the provided context, cite the file path where the information comes from.
        If the answer cannot be determined from the context, say so clearly.
        
        Always include the GitHub links to the relevant files when answering questions about code or documentation.
        """
        
        # System message for out-of-scope detection
        self.scope_checker_system_template = """You are an AI assistant that determines if questions are related to the Vanna AI repository (https://github.com/vanna-ai/vanna).
        Vanna is an open-source Python RAG (Retrieval-Augmented Generation) framework for SQL generation.
        
        For each question, determine if it is:
        1. RELATED: The question is about Vanna's functionality, code structure, features, or usage.
        2. UNRELATED: The question is not about Vanna or is about a completely different topic.
        
        Respond with either "RELATED" or "UNRELATED" followed by a brief explanation.
        """
        
        # System message for out-of-scope responses
        self.out_of_scope_system_template = """You are an AI assistant specialized in answering questions about the Vanna AI repository.
        
        The user has asked a question that is not related to the Vanna AI repository. 
        Politely explain that the question is out of scope for this specific system, which is designed to answer questions about the Vanna AI repository.
        Suggest that they ask a question related to Vanna instead.
        
        Be concise and helpful in your response.
        """
    
    def check_query_scope(self, query: str) -> Dict[str, Any]:
        """Check if a query is in scope (related to the Vanna repository)."""
        start_time = time.time()
        logger.info(f"Checking if query is in scope: {query}")
        
        try:
            # Create messages directly
            messages = [
                SystemMessage(content=self.scope_checker_system_template),
                HumanMessage(content=f"Question: {query}")
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Parse response
            in_scope = "RELATED" in response_text.upper()
            
            result = {
                "in_scope": in_scope,
                "explanation": response_text,
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Query scope check result: in_scope={in_scope}")
            return result
        
        except Exception as e:
            logger.error(f"Error checking query scope: {e}")
            # Default to in-scope on error (to allow the full pipeline to run)
            return {
                "in_scope": True,
                "explanation": f"Error checking scope: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def generate_answer(self, query: str, context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer to an in-scope query using retrieved context."""
        start_time = time.time()
        logger.info(f"Generating answer for query: {query}")
        
        try:
            # Format context for the prompt
            formatted_context = ""
            github_base_url = "https://github.com/vanna-ai/vanna/blob/main/"
            
            for i, doc in enumerate(context_documents):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                file_path = metadata.get('file_path', '')
                
                # Create a GitHub URL
                if file_path:
                    relative_path = file_path.split('vanna/')[-1] if 'vanna/' in file_path else file_path
                    github_url = f"{github_base_url}{relative_path}"
                    formatted_context += f"\nDOCUMENT {i+1} (Source: {github_url}):\n{content}\n"
                else:
                    formatted_context += f"\nDOCUMENT {i+1}:\n{content}\n"
            
            # Create messages directly
            messages = [
                SystemMessage(content=self.in_scope_system_template),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{formatted_context}")
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            result = {
                "answer": response_text,
                "sources": [doc.get('metadata', {}).get('file_path', '') for doc in context_documents],
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Generated answer in {time.time() - start_time:.2f} seconds")
            return result
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": f"I encountered an error while generating an answer: {str(e)}",
                "sources": [],
                "processing_time": time.time() - start_time
            }
    
    def generate_out_of_scope_response(self, query: str) -> Dict[str, Any]:
        """Generate a response for an out-of-scope query."""
        start_time = time.time()
        logger.info(f"Generating out-of-scope response for query: {query}")
        
        try:
            # Create messages directly
            messages = [
                SystemMessage(content=self.out_of_scope_system_template),
                HumanMessage(content=f"Question: {query}")
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            result = {
                "answer": response_text,
                "sources": [],
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Generated out-of-scope response in {time.time() - start_time:.2f} seconds")
            return result
        
        except Exception as e:
            logger.error(f"Error generating out-of-scope response: {e}")
            return {
                "answer": "I'm sorry, but your question appears to be unrelated to the Vanna AI repository. This system is designed specifically to answer questions about Vanna AI. Please ask a question related to Vanna AI instead.",
                "sources": [],
                "processing_time": time.time() - start_time
            }
    
    def generate_answer_with_references(self, query: str, context_documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate an answer to an in-scope query with specific GitHub references."""
        start_time = time.time()
        logger.info(f"Generating answer with references for query: {query}")
        
        try:
            # Format context for the prompt
            formatted_context = ""
            github_base_url = "https://github.com/vanna-ai/vanna/blob/main/"
            references = []
            
            for i, doc in enumerate(context_documents):
                content = doc.get('content', '')
                metadata = doc.get('metadata', {})
                file_path = metadata.get('file_path', '')
                
                # Create a GitHub URL with line numbers if possible
                if file_path:
                    relative_path = file_path.split('vanna/')[-1] if 'vanna/' in file_path else file_path
                    github_url = f"{github_base_url}{relative_path}"
                    
                    # Add to references
                    references.append({
                        "file_path": file_path,
                        "github_url": github_url,
                        "relevance_score": doc.get('similarity_score', 0)
                    })
                    
                    formatted_context += f"\nDOCUMENT {i+1} (Source: {github_url}):\n{content}\n"
                else:
                    formatted_context += f"\nDOCUMENT {i+1}:\n{content}\n"
            
            # Enhanced system prompt that encourages specific file and line references
            enhanced_system_template = """You are an AI assistant specialized in answering questions about the Vanna AI repository (https://github.com/vanna-ai/vanna). 
            Vanna is an open-source Python RAG (Retrieval-Augmented Generation) framework for SQL generation.
            
            Given context information about the repository, answer the user's question accurately and concisely.
            
            IMPORTANT: When referring to code or information from the repository:
            1. Include specific GitHub links to the relevant files using the format: [file_name](github_url)
            2. Be specific about where in the file the information comes from (e.g., "In the `function_name` function" or "In the class definition")
            3. When quoting code or text, use exact quotes and be specific about their location
            
            If the answer cannot be determined from the context, say so clearly.
            """
            
            # Create messages directly
            messages = [
                SystemMessage(content=enhanced_system_template),
                HumanMessage(content=f"Question: {query}\n\nContext:\n{formatted_context}")
            ]
            
            # Get response
            response = self.llm.invoke(messages)
            response_text = response.content
            
            result = {
                "answer": response_text,
                "references": references,
                "processing_time": time.time() - start_time
            }
            
            logger.info(f"Generated answer with references in {time.time() - start_time:.2f} seconds")
            return result
        
        except Exception as e:
            logger.error(f"Error generating answer with references: {e}")
            return {
                "answer": f"I encountered an error while generating an answer: {str(e)}",
                "references": [],
                "processing_time": time.time() - start_time
            }
