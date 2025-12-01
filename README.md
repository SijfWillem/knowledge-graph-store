# Cognee Memory - Hybrid GraphRAG System

A hybrid RAG (Retrieval-Augmented Generation) system powered by [Cognee](https://github.com/topoteretes/cognee) that combines knowledge graphs with vector search for intelligent document Q&A.

## Features

- **Document Upload** - Upload PDF, TXT, MD, JSON, CSV, DOCX files
- **Knowledge Graph** - Automatically extracts entities and relationships from documents
- **GraphRAG Chat** - Ask questions and get answers based on the knowledge graph
- **Graph Visualization** - Interactive visualization of the knowledge graph
- **Multiple Search Modes** - GRAPH_COMPLETION, RAG_COMPLETION, CHUNKS, SUMMARIES

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend     │────▶│     Neo4j       │
│   (Streamlit)   │     │    (FastAPI)    │     │  (Graph Store)  │
│   Port: 8501    │     │   Port: 8000    │     │   Port: 7688    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │    LanceDB      │
                        │ (Vector Store)  │
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
- Start Neo4j, Backend, and Frontend
- Display access URLs

### 3. Access

- **Frontend UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **Neo4j Browser**: http://localhost:7475 (neo4j/cogneepassword)

## Usage

### Upload Documents

1. Open http://localhost:8501
2. Use the sidebar to upload one or more documents
3. Wait for processing to complete (watch the status indicator)

### Ask Questions

1. Go to the "Chat" tab
2. Select search type (GRAPH_COMPLETION recommended for Q&A)
3. Type your question and press Enter
4. Get answers with knowledge graph sources

### View Knowledge Graph

1. Go to the "Knowledge Graph" tab
2. Click "Refresh Graph" to load the latest data
3. Interact with the visualization

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
│   └── main.py           # FastAPI + Cognee integration
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

## Search Types

- **GRAPH_COMPLETION** - Best for Q&A, uses knowledge graph relationships
- **RAG_COMPLETION** - Standard RAG with vector search
- **CHUNKS** - Returns raw text chunks
- **SUMMARIES** - Returns document summaries

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
```

## Troubleshooting

### Services won't start
```bash
make wipe
make start
```

### Database not initialized error
The Cognee database resets on container restart. Re-upload your documents after restarting.

### Port conflicts
Edit `docker-compose.yml` to change port mappings:
- Neo4j HTTP: 7475
- Neo4j Bolt: 7688
- Backend: 8000
- Frontend: 8501

## Tech Stack

- **Cognee** - Knowledge graph + RAG framework
- **Neo4j** - Graph database
- **LanceDB** - Vector database
- **FastAPI** - Backend API
- **Streamlit** - Frontend UI
- **OpenAI** - LLM for GraphRAG

## License

MIT
