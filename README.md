# Cognee Memory - Knowledge Graph System

A knowledge graph-based memory system powered by [Cognee](https://github.com/topoteretes/cognee) that extracts entities and relationships from documents for intelligent Q&A with full observability.

## Features

- **Document Upload** - Upload PDF, TXT, MD, JSON, CSV, DOCX files
- **Knowledge Graph** - Automatically extracts entities and relationships from documents
- **Graph-Based Q&A** - Ask questions and get answers with visible graph context (nodes & relationships)
- **Graph Visualization** - Interactive visualization of the knowledge graph
- **LangFuse Observability** - Full tracing of queries, retrievals, and LLM generations
- **Multiple Search Modes** - GRAPH_COMPLETION, CHUNKS, SUMMARIES, and more

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend     │────▶│     Neo4j       │
│   (Streamlit)   │     │    (FastAPI)    │     │  (Graph Store)  │
│   Port: 8501    │     │   Port: 8000    │     │   Port: 7688    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ├───────────────────┐
                               ▼                   ▼
                        ┌─────────────────┐ ┌─────────────────┐
                        │    LanceDB      │ │    LangFuse     │
                        │ (Vector Store)  │ │ (Observability) │
                        └─────────────────┘ │   Port: 3000    │
                                            └─────────────────┘
```

## Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API Key

### 1. Setup

```bash
# Clone or navigate to the project
cd knowledge-cognee

# Create .env file
make setup

# Edit .env and add your OpenAI API key
nano .env  # or use your preferred editor
```

Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=sk-your-api-key-here
```

### 2. Start

```bash
make start
```

This will:
- Build all Docker images
- Start Neo4j, LangFuse, Backend, and Frontend
- Display access URLs

### 3. Access

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **Neo4j Browser**: http://localhost:7475 (neo4j/cogneepassword)
- **LangFuse Dashboard**: http://localhost:3000

## Usage

### Upload Documents

1. Open http://localhost:8501
2. Use the sidebar to upload one or more documents
3. Wait for processing to complete (watch the status indicator)

### Ask Questions

1. Go to the "Chat" tab
2. Select search type (GRAPH_COMPLETION recommended for Q&A)
3. Type your question and press Enter
4. View the answer along with the Knowledge Graph Context showing:
   - **Retrieved Nodes** - Entities from the graph used to answer
   - **Relationships** - Connections between entities (e.g., `person → works_at → company`)

### View Knowledge Graph

1. Go to the "Knowledge Graph" tab
2. Click "Refresh Graph" to load the latest data
3. Interact with the visualization

### Monitor with LangFuse

1. Open http://localhost:3000
2. Create an account and project (first time only)
3. View traces showing:
   - Query processing pipeline
   - Context retrieval with node/connection counts
   - LLM generation details

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make start` | Build and start everything |
| `make wipe` | Delete ALL data and stop services |
| `make up` | Start services |
| `make down` | Stop services |
| `make restart` | Restart services |
| `make logs` | View all logs |
| `make status` | Show container status |
| `make health` | Check service health |

## Project Structure

```
knowledge-cognee/
├── docker-compose.yml    # Container orchestration
├── Makefile              # Build and run commands
├── .env.example          # Environment template
├── README.md             # This file
├── uploads/              # Uploaded documents
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py           # FastAPI + Cognee + LangFuse integration
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py            # Streamlit UI
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Processing status |
| `/upload` | POST | Upload documents |
| `/add-text` | POST | Add text directly |
| `/query` | POST | Query the knowledge graph |
| `/graph` | GET | Get graph data for visualization |
| `/documents` | GET | List uploaded documents |
| `/reset` | DELETE | Reset knowledge base |

### Query Response Format

The `/query` endpoint returns structured graph context:

```json
{
  "query": "who works at company X?",
  "answer": "Alice and Bob work at company X.",
  "search_type": "GRAPH_COMPLETION",
  "graph_context": {
    "nodes": [
      {"name": "Alice", "type": "Entity", "content": "..."},
      {"name": "Company X", "type": "Entity", "content": "..."}
    ],
    "connections": [
      {"source": "Alice", "relationship": "works_at", "target": "Company X"}
    ]
  }
}
```

## Search Types

- **GRAPH_COMPLETION** - Uses knowledge graph relationships for Q&A (recommended)
- **GRAPH_COMPLETION_COT** - Chain-of-thought reasoning over the graph
- **CHUNKS** - Returns raw text chunks
- **CHUNKS_LEXICAL** - Token-based lexical search
- **SUMMARIES** - Returns document summaries
- **CODE** - Code-specific search

## Configuration

Environment variables (`.env`):

```bash
# Required
OPENAI_API_KEY=sk-...

# Neo4j (pre-configured for Docker)
GRAPH_DATABASE_PROVIDER=neo4j
GRAPH_DATABASE_URL=bolt://localhost:7688
GRAPH_DATABASE_USERNAME=neo4j
GRAPH_DATABASE_PASSWORD=cogneepassword

# Vector DB (LanceDB - file-based)
VECTOR_DB_PROVIDER=lancedb

# LangFuse Observability
MONITORING_TOOL=langfuse
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_HOST=http://localhost:3000
```

## Troubleshooting

### Services won't start
```bash
make wipe
make start
```

### Port conflicts
Edit `docker-compose.yml` to change port mappings:
- Neo4j HTTP: 7475
- Neo4j Bolt: 7688
- Backend: 8000
- Frontend: 8501
- LangFuse: 3000

### LangFuse not showing traces
1. Ensure LangFuse is running: `docker compose ps langfuse`
2. Check API keys in `.env` match those in LangFuse dashboard
3. Restart backend after updating keys: `docker compose restart backend`

## Tech Stack

- **Cognee** - Knowledge graph extraction and search
- **Neo4j** - Graph database for storing entities and relationships
- **LanceDB** - Vector database for embeddings
- **FastAPI** - Backend API
- **Streamlit** - Frontend UI
- **LangFuse** - Observability and tracing
- **OpenAI** - LLM for entity extraction and answer generation

## License

MIT
