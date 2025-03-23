"""Script to index the Vanna AI repository."""

import os
import logging
import time
import sys
import traceback
from dotenv import load_dotenv

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

from app.utils.github import GitHubScraper
from app.services.document_processor import DocumentParser, TextChunker, EmbeddingService
from app.services.retrieval import VectorStore
from app.core.config import get_settings
from app.core.logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """Script to index the Vanna repository."""
    start_time = time.time()
    logger.info("Starting repository indexing")
    
    # Get config
    config = get_settings()
    
    # Check if Azure OpenAI settings are properly configured
    if not config.get("azure_api_key") or not config.get("azure_endpoint"):
        logger.error("Azure OpenAI API key or endpoint not set. Please check your .env file.")
        sys.exit(1)
    
    # Setup repository URL
    repo_url = os.getenv("REPO_URL", "https://github.com/vanna-ai/vanna")
    
    try:
        # Initialize components
        scraper = GitHubScraper(repo_url)
        parser = DocumentParser()
        chunker = TextChunker(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200)
        )
        
        # Initialize embedding service
        try:
            embedding_service = EmbeddingService(config)
            logger.info("Embedding service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
        
        # Initialize vector store
        try:
            vector_store = VectorStore(config)
            logger.info("Vector store initialized")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
        
        # Clone repository and get files
        try:
            repo_dir = scraper.clone_repository()
            logger.info(f"Repository cloned to {repo_dir}")
            
            files = scraper.get_file_paths(extensions=['.py', '.md', '.txt', '.ipynb', '.js', '.html', '.json', '.yaml', '.yml'])
            logger.info(f"Found {len(files)} files to process")
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise
        
        # Process files
        all_chunks = []
        file_count = 0
        
        for i, file_path in enumerate(files):
            try:
                # Parse document
                document = parser.parse_file(file_path)
                
                # Chunk document
                chunks = chunker.chunk_document(document)
                
                # Add to collection
                all_chunks.extend(chunks)
                file_count += 1
                
                logger.info(f"[{i+1}/{len(files)}] Processed {file_path} into {len(chunks)} chunks")
                
                # Process in smaller batches to avoid memory issues
                if file_count % 20 == 0 or i == len(files) - 1:
                    if all_chunks:
                        logger.info(f"Creating embeddings for batch of {len(all_chunks)} chunks")
                        chunks_with_embeddings = embedding_service.embed_documents(all_chunks)
                        
                        logger.info(f"Adding {len(chunks_with_embeddings)} chunks to vector store")
                        vector_store.add_documents(chunks_with_embeddings)
                        
                        all_chunks = []
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                logger.error(traceback.format_exc())
                # Continue with next file instead of stopping
                continue
        
        # Clean up
        scraper.cleanup()
        
        # Get vector store stats
        try:
            stats = vector_store.get_stats()
            logger.info(f"Vector store stats: {stats}")
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}")
        
        logger.info(f"Indexing completed in {time.time() - start_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
