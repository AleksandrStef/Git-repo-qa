"""Document processing service for Vanna AI Repository Q&A System."""

import os
import logging
import re
import time
import markdown
from typing import List, Dict, Any, Optional
from langchain_openai import AzureOpenAIEmbeddings

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parser for different document types."""
    
    def __init__(self):
        pass
    
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a single file and extract its content with metadata."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # Process based on file type
            if file_ext in ['.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.cs']:
                return self._parse_code_file(file_path, content)
            elif file_ext in ['.md', '.markdown']:
                return self._parse_markdown_file(file_path, content)
            elif file_ext in ['.txt', '.csv', '.json', '.yml', '.yaml']:
                return self._parse_text_file(file_path, content)
            else:
                return self._parse_generic_file(file_path, content)
        
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return {
                'file_path': file_path,
                'content': '',
                'metadata': {
                    'status': 'error',
                    'error': str(e)
                }
            }
    
    def _parse_code_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Parse a code file."""
        # Extract docstrings, function definitions, class definitions
        # This is a simplified version and could be enhanced with a proper AST parser
        
        # Get relative path from repo root
        rel_path = os.path.basename(file_path)
        
        return {
            'file_path': file_path,
            'content': content,
            'metadata': {
                'type': 'code',
                'language': os.path.splitext(file_path)[1][1:],
                'filename': os.path.basename(file_path),
                'relative_path': rel_path,
                'size': len(content),
                'functions': self._extract_function_names(content),
                'classes': self._extract_class_names(content),
            }
        }
    
    def _parse_markdown_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Parse a markdown file."""
        # Convert markdown to plain text
        plain_text = re.sub(r'<.*?>', '', markdown.markdown(content))
        
        return {
            'file_path': file_path,
            'content': content,
            'plain_text': plain_text,
            'metadata': {
                'type': 'markdown',
                'filename': os.path.basename(file_path),
                'relative_path': os.path.basename(file_path),
                'size': len(content),
                'headings': self._extract_markdown_headings(content),
            }
        }
    
    def _parse_text_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Parse a text file."""
        return {
            'file_path': file_path,
            'content': content,
            'metadata': {
                'type': 'text',
                'filename': os.path.basename(file_path),
                'relative_path': os.path.basename(file_path),
                'size': len(content),
            }
        }
    
    def _parse_generic_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Parse a generic file."""
        return {
            'file_path': file_path,
            'content': content,
            'metadata': {
                'type': 'generic',
                'filename': os.path.basename(file_path),
                'relative_path': os.path.basename(file_path),
                'size': len(content),
            }
        }
    
    def _extract_function_names(self, content: str) -> List[str]:
        """Extract function names from a code file."""
        # Simple regex for Python functions
        function_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        return re.findall(function_pattern, content)
    
    def _extract_class_names(self, content: str) -> List[str]:
        """Extract class names from a code file."""
        # Simple regex for Python classes
        class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\(|:)'
        return re.findall(class_pattern, content)
    
    def _extract_markdown_headings(self, content: str) -> List[str]:
        """Extract headings from a markdown file."""
        heading_pattern = r'^(#+)\s+(.+?)$'
        headings = []
        
        for line in content.split('\n'):
            match = re.match(heading_pattern, line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headings.append(f"h{level}:{text}")
        
        return headings


class TextChunker:
    """Chunker for splitting documents into smaller pieces."""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks with metadata."""
        content = document.get('content', '')
        
        # Skip empty documents
        if not content:
            return []
        
        # Determine chunking strategy based on document type
        doc_type = document.get('metadata', {}).get('type', 'generic')
        
        if doc_type == 'code':
            return self._chunk_code(document)
        elif doc_type == 'markdown':
            return self._chunk_markdown(document)
        else:
            return self._chunk_by_size(document)
    
    def _chunk_code(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk code files by logical sections."""
        content = document.get('content', '')
        file_path = document.get('file_path', '')
        metadata = document.get('metadata', {})
        
        chunks = []
        lines = content.split('\n')
        
        # Try to find logical blocks (functions, classes, etc.)
        current_chunk = []
        current_chunk_type = None
        
        for line in lines:
            # Detect start of a new logical block
            if line.startswith('def ') or line.startswith('class '):
                # Save the previous chunk if it exists
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append({
                        'content': chunk_content,
                        'metadata': {
                            **metadata,
                            'chunk_type': current_chunk_type or 'code',
                            'file_path': file_path,
                            'size': len(chunk_content)
                        }
                    })
                
                # Start a new chunk
                current_chunk = [line]
                current_chunk_type = 'function' if line.startswith('def ') else 'class'
            else:
                current_chunk.append(line)
        
        # Add the last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    **metadata,
                    'chunk_type': current_chunk_type or 'code',
                    'file_path': file_path,
                    'size': len(chunk_content)
                }
            })
        
        return chunks
    
    def _chunk_markdown(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk markdown files by headings."""
        content = document.get('content', '')
        file_path = document.get('file_path', '')
        metadata = document.get('metadata', {})
        
        chunks = []
        lines = content.split('\n')
        
        # Chunk by headings
        current_chunk = []
        current_heading = None
        
        for line in lines:
            # Check if this is a heading line
            if line.startswith('#'):
                # Save the previous chunk if it exists
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append({
                        'content': chunk_content,
                        'metadata': {
                            **metadata,
                            'chunk_type': 'markdown_section',
                            'heading': current_heading,
                            'file_path': file_path,
                            'size': len(chunk_content)
                        }
                    })
                
                # Start a new chunk with this heading
                current_chunk = [line]
                current_heading = line.lstrip('#').strip()
            else:
                current_chunk.append(line)
        
        # Add the last chunk
        if current_chunk:
            chunk_content = '\n'.join(current_chunk)
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    **metadata,
                    'chunk_type': 'markdown_section',
                    'heading': current_heading,
                    'file_path': file_path,
                    'size': len(chunk_content)
                }
            })
        
        return chunks
    
    def _chunk_by_size(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text by size with overlap."""
        content = document.get('content', '')
        file_path = document.get('file_path', '')
        metadata = document.get('metadata', {})
        
        chunks = []
        
        # Simple chunking by size with overlap
        start = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # Try to break at a natural point (newline)
            if end < len(content):
                # Look for a newline within the last 20% of the chunk
                last_newline = content.rfind('\n', start + int(self.chunk_size * 0.8), end)
                if last_newline != -1:
                    end = last_newline + 1
            
            chunk_content = content[start:end]
            chunks.append({
                'content': chunk_content,
                'metadata': {
                    **metadata,
                    'chunk_type': 'text_chunk',
                    'chunk_index': len(chunks),
                    'file_path': file_path,
                    'size': len(chunk_content)
                }
            })
            
            # Move to next chunk, accounting for overlap
            start = end - self.chunk_overlap
            if start >= end:  # Ensure we make progress
                start = end
        
        return chunks


class EmbeddingService:
    """Service for generating embeddings."""
    
    def __init__(self, config):
        """Initialize the embedding service with Azure OpenAI."""
        self.config = config
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                deployment=config.get("azure_embedding_deployment"),
                openai_api_version=config.get("azure_api_version"),
                azure_endpoint=config.get("azure_endpoint"),
                api_key=config.get("azure_api_key"),
            )
        except Exception as e:
            logger.error(f"Error initializing AzureOpenAIEmbeddings: {e}")
            # Try fallback to regular "azure" API type
            self.embeddings = AzureOpenAIEmbeddings(
                azure_deployment=config.get("azure_embedding_deployment"),
                api_version=config.get("azure_api_version"),
                azure_endpoint=config.get("azure_endpoint"),
                api_key=config.get("azure_api_key"),
            )
    
    def embed_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create embeddings for a list of document chunks."""
        if not chunks:
            return []
        
        start_time = time.time()
        logger.info(f"Embedding {len(chunks)} document chunks")
        
        # Extract just the content for embedding
        texts = [chunk.get('content', '') for chunk in chunks]
        
        try:
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add embeddings back to the chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i]
            
            logger.info(f"Embedding completed in {time.time() - start_time:.2f} seconds")
            return chunks
        
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            # Return chunks without embeddings on error
            for chunk in chunks:
                chunk['embedding'] = None
            return chunks
