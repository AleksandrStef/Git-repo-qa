"""Script to evaluate the Q&A system."""

import os
import logging
import time
import json
import sys
from dotenv import load_dotenv

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

from app.services.query_processor import QueryProcessor, EnhancedQueryProcessor
from app.services.retrieval import VectorStore, EnhancedVectorStore
from app.services.llm_service import LLMService
from app.core.config import get_settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Test questions
TEST_QUESTIONS = [
    # In-scope questions about Vanna
    "What is Vanna AI?",
    "How does the RAG framework in Vanna work?",
    "What vector databases are supported by Vanna?",
    "How do I train a Vanna model with DDL statements?",
    "What LLMs are supported by Vanna?",
    "How can I use Vanna with ChromaDB?",
    
    # Out-of-scope questions
    "What is the capital of France?",
    "How do I make chocolate chip cookies?",
    "What is the weather like today?",
    "Who won the last Super Bowl?",
    "Can you explain quantum physics?"
]

def main():
    """Script to evaluate the Q&A system."""
    start_time = time.time()
    logger.info("Starting evaluation")
    
    # Get config
    config = get_settings()
    
    try:
        # Initialize services
        vector_store = EnhancedVectorStore(config)
        llm_service = LLMService(config)
        query_processor = EnhancedQueryProcessor(config, vector_store, llm_service)
        
        # Process each test question
        results = []
        for i, question in enumerate(TEST_QUESTIONS):
            logger.info(f"[{i+1}/{len(TEST_QUESTIONS)}] Testing question: {question}")
            
            question_start_time = time.time()
            result = query_processor.process_query(question)
            question_time = time.time() - question_start_time
            
            # Format result for logging
            log_result = {
                "question": question,
                "in_scope": result.get("in_scope", True),
                "processing_time": f"{question_time:.2f}s",
                "answer_preview": result.get("answer", "")[:100] + "..." if result.get("answer") else ""
            }
            
            logger.info(f"Result: {json.dumps(log_result, indent=2)}")
            results.append(result)
        
        # Calculate statistics
        in_scope_count = sum(1 for r in results if r.get("in_scope", True))
        out_of_scope_count = len(results) - in_scope_count
        avg_processing_time = sum(r.get("total_processing_time", 0) for r in results) / len(results)
        
        logger.info(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Statistics:")
        logger.info(f"  - Total questions: {len(results)}")
        logger.info(f"  - In-scope questions: {in_scope_count}")
        logger.info(f"  - Out-of-scope questions: {out_of_scope_count}")
        logger.info(f"  - Average processing time: {avg_processing_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
