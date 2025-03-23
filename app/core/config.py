"""Configuration module for Vanna AI Repository Q&A System."""

import os
from typing import Dict, Any, Optional
from pydantic import BaseModel
from functools import lru_cache


class Settings(BaseModel):
    """Application settings."""
    # Azure OpenAI settings
    azure_api_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    azure_api_version: Optional[str] = "2023-05-15"
    azure_llm_deployment: Optional[str] = "gpt-4"
    azure_embedding_deployment: Optional[str] = "text-embedding-ada-002"
    
    # Vector store settings (Qdrant)
    qdrant_persist_directory: Optional[str] = "./data/vector_store"
    qdrant_collection_name: Optional[str] = "vanna_repo"
    
    # Document processing settings
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    
    # Retrieval settings
    retrieval_k: Optional[int] = 5
    
    # API settings
    api_prefix: str = "/api/v1"
    
    # Repository settings
    repo_url: Optional[str] = "https://github.com/vanna-ai/vanna"


def load_settings() -> Settings:
    """Load settings from environment variables."""
    return Settings(
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15"),
        azure_llm_deployment=os.getenv("AZURE_OPENAI_LLM_DEPLOYMENT", "gpt-4"),
        azure_embedding_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        qdrant_persist_directory=os.getenv("QDRANT_PERSIST_DIRECTORY", "./data/vector_store"),
        qdrant_collection_name=os.getenv("QDRANT_COLLECTION_NAME", "vanna_repo"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "5")),
        repo_url=os.getenv("REPO_URL", "https://github.com/vanna-ai/vanna")
    )


@lru_cache()
def get_settings() -> Dict[str, Any]:
    """Get application settings as a dictionary."""
    settings = load_settings()
    return settings.model_dump()
