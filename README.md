# Vanna AI Repository Q&A System

This project implements an agentic AI solution using LangChain to answer questions about the Vanna AI repository. The system can recognize and respond to both in-scope and out-of-scope questions.

## Features

- **Document Indexing**: Indexes all relevant documents within the Vanna AI repository
- **Question Answering**: Uses Azure OpenAI to answer user questions based on the indexed content
- **Out-of-Scope Detection**: Detects and responds appropriately to questions not related to the repository
- **REST API**: Provides a REST API for interaction with the system
- **Dockerized**: Easy deployment using Docker

## Architecture

The system is built with the following components:

1. **Document Processing Pipeline**: Clones the repository, parses files, chunks text, and creates embeddings
2. **Vector Database**: Uses Qdrant to store document chunks and embeddings for efficient retrieval
3. **LLM Service**: Interfaces with Azure OpenAI for question answering and scope detection
4. **Query Processing**: Coordinates the end-to-end process of answering questions
5. **REST API**: Provides HTTP endpoints for the system

## Prerequisites

- Python 3.9+
- Azure OpenAI API access
- Docker (optional for containerized deployment)

## Environment Variables

The following environment variables need to be set:

```
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=your_endpoint
AZURE_OPENAI_API_VERSION=2023-05-15
AZURE_OPENAI_LLM_DEPLOYMENT=your_deployment_name
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=your_embedding_deployment_name

# Qdrant settings
QDRANT_PERSIST_DIRECTORY=./data/vector_store
QDRANT_COLLECTION_NAME=vanna_repo
```

## Local Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/AleksandrStef/Git-repo-qa.git
   cd vanna-ai-qa-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your Azure OpenAI credentials
   ```

5. Index the repository:
   ```bash
   python scripts/index_repository.py
   ```

6. Run the API:
   ```bash
   python main.py
   ```

7. Access the API at http://localhost:8000/docs

## Docker Setup

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. Access the API at http://localhost:8000/docs

## API Endpoints

- `POST /api/v1/query`: Submit a question to the system
- `POST /api/v1/indexing`: Start the indexing process
- `GET /api/v1/health`: Check the health of the system
- `GET /api/v1/stats`: Get statistics about the system

### Query Example

```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/query' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What vector databases are supported by Vanna?"
}'
```

## Enhanced Accuracy Features

This implementation includes several techniques to improve the accuracy of responses:

1. **GitHub Link Generation**: Generates precise references to files and code sections
2. **Query Expansion**: Reformulates the original query to improve retrieval
3. **Enhanced Document Processing**: Uses intelligent chunking to preserve context
4. **Hybrid Search**: Combines different search techniques for better results

## Evaluation

Run the evaluation script to test the system with a set of predefined questions:

```bash
python scripts/evaluate.py
```

## Vector Database: Qdrant

This system uses Qdrant as the vector database for storing and retrieving document embeddings. Qdrant offers several advantages:

1. **High Performance**: Efficient similarity search with optimized algorithms
2. **Scalability**: Works well for both small and large datasets
3. **Persistence**: Local file-based storage for easy deployment
4. **Rich Filtering**: Supports complex filtering based on metadata
5. **Active Development**: Well-maintained with regular updates

## Performance Metrics

- **Indexing Performance**: The system logs detailed timing information during indexing
- **Query Performance**: Each query response includes timing metrics for the different processing stages
- **API Response Time**: Typically under 3 seconds for most queries

## Limitations and Future Work

- The system currently indexes only the main branch of the repository

