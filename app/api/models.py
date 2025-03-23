"""API models for Vanna AI Repository Q&A System."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for the query endpoint."""
    query: str = Field(..., description="The question about the Vanna repository")


class QueryResponse(BaseModel):
    """Response model for the query endpoint."""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The answer to the query")
    in_scope: bool = Field(..., description="Whether the query is in scope")
    sources: List[str] = Field(default=[], description="Source files used for the answer")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class IndexingRequest(BaseModel):
    """Request model for the indexing endpoint."""
    repo_url: str = Field(
        default="https://github.com/vanna-ai/vanna",
        description="The URL of the repository to index"
    )
    reset: bool = Field(
        default=False,
        description="Whether to reset the vector store before indexing"
    )


class IndexingResponse(BaseModel):
    """Response model for the indexing endpoint."""
    message: str = Field(..., description="Status message")
    status: str = Field(..., description="Indexing status")
    repo_url: str = Field(..., description="The URL of the repository being indexed")


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""
    status: str = Field(..., description="Health status")
    vector_store_document_count: int = Field(..., description="Number of documents in the vector store")
    api_version: str = Field(..., description="API version")
    error: Optional[str] = Field(None, description="Error message if unhealthy")


class StatsResponse(BaseModel):
    """Response model for the stats endpoint."""
    vector_store: Dict[str, Any] = Field(..., description="Vector store statistics")
    document_count: int = Field(..., description="Total number of documents indexed")
