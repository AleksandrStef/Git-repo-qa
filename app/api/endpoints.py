"""API endpoints for Vanna AI Repository Q&A System."""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List, Dict, Any, Optional
import logging
import time

from app.core.config import get_settings
from app.api.models import (
    QueryRequest, 
    QueryResponse, 
    IndexingRequest, 
    IndexingResponse,
    HealthResponse,
    StatsResponse
)
from app.services.document_processor import DocumentParser, TextChunker, EmbeddingService
from app.services.retrieval import VectorStore, EnhancedVectorStore
from app.services.llm_service import LLMService
from app.services.query_processor import QueryProcessor, EnhancedQueryProcessor
from app.utils.github import GitHubScraper

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Dependency for services
def get_services():
    """Get instances of all required services."""
    config = get_settings()
    
    # Initialize services
    try:
        vector_store = EnhancedVectorStore(config)
        llm_service = LLMService(config)
        query_processor = EnhancedQueryProcessor(config, vector_store, llm_service)
        
        return {
            "vector_store": vector_store,
            "llm_service": llm_service,
            "query_processor": query_processor,
            "config": config
        }
    except Exception as e:
        logger.error(f"Error initializing services: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing services: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    services: Dict[str, Any] = Depends(get_services)
):
    """Process a query about the Vanna repository."""
    query_processor = services["query_processor"]
    
    try:
        result = query_processor.process_query(request.query)
        return QueryResponse(
            query=result["query"],
            answer=result["answer"],
            in_scope=result["in_scope"],
            sources=result["sources"],
            processing_time_ms=int(result["total_processing_time"] * 1000)
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@router.post("/indexing", response_model=IndexingResponse)
async def start_indexing(
    request: IndexingRequest,
    background_tasks: BackgroundTasks,
    services: Dict[str, Any] = Depends(get_services)
):
    """Start the indexing process for the Vanna repository."""
    config = services["config"]
    vector_store = services["vector_store"]
    
    # Start indexing in the background
    background_tasks.add_task(
        index_repository,
        repo_url=request.repo_url,
        config=config,
        vector_store=vector_store,
        reset=request.reset
    )
    
    return IndexingResponse(
        message="Indexing started in the background",
        status="running",
        repo_url=request.repo_url
    )


async def index_repository(repo_url: str, config: Dict[str, Any], vector_store: VectorStore, reset: bool = False):
    """Index the repository in the background."""
    logger.info(f"Starting indexing process for {repo_url}")
    start_time = time.time()
    
    try:
        # Initialize components
        scraper = GitHubScraper(repo_url)
        parser = DocumentParser()
        chunker = TextChunker(
            chunk_size=config.get("chunk_size", 1000),
            chunk_overlap=config.get("chunk_overlap", 200)
        )
        embedding_service = EmbeddingService(config)
        
        # Clone repository and get files
        repo_dir = scraper.clone_repository()
        files = scraper.get_file_paths(extensions=['.py', '.md', '.txt', '.ipynb', '.js', '.html', '.json', '.yaml', '.yml'])
        
        logger.info(f"Found {len(files)} files to process")
        
        # Process files
        all_chunks = []
        for file_path in files:
            try:
                # Parse document
                document = parser.parse_file(file_path)
                
                # Chunk document
                chunks = chunker.chunk_document(document)
                
                # Add to collection
                all_chunks.extend(chunks)
                
                logger.info(f"Processed {file_path} into {len(chunks)} chunks")
            
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
        
        # Create embeddings for all chunks
        logger.info(f"Creating embeddings for {len(all_chunks)} chunks")
        chunks_with_embeddings = embedding_service.embed_documents(all_chunks)
        
        # Add to vector store
        vector_store.add_documents(chunks_with_embeddings)
        
        logger.info(f"Indexing completed in {time.time() - start_time:.2f} seconds")
        
        # Clean up
        scraper.cleanup()
    
    except Exception as e:
        logger.error(f"Error during indexing: {e}")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    services: Dict[str, Any] = Depends(get_services)
):
    """Check the health of the system."""
    vector_store = services["vector_store"]
    
    try:
        # Get vector store stats
        stats = vector_store.get_stats()
        
        return HealthResponse(
            status="healthy",
            vector_store_document_count=stats.get("document_count", 0),
            api_version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            vector_store_document_count=0,
            api_version="1.0.0",
            error=str(e)
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    services: Dict[str, Any] = Depends(get_services)
):
    """Get statistics about the system."""
    vector_store = services["vector_store"]
    
    try:
        # Get vector store stats
        stats = vector_store.get_stats()
        
        return StatsResponse(
            vector_store=stats,
            document_count=stats.get("document_count", 0)
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")
