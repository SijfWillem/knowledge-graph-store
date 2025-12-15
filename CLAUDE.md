# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge graph-based memory system powered by Cognee that extracts entities and relationships from documents for intelligent Q&A with LangFuse observability.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│     Backend     │────▶│     Neo4j       │
│   (Streamlit)   │     │    (FastAPI)    │     │  (Graph Store)  │
│   Port: 8501    │     │   Port: 8000    │     │   Port: 7688    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│    LanceDB      │  │    LangFuse     │  │ Knowledge Gap   │
│ (Vector Store)  │  │ (Observability) │  │   Dashboard     │
│    (local)      │  │   Port: 3000    │  │   Port: 3001    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

**Main Components:**
- `backend/main.py`: FastAPI server with Cognee integration, LangChain agent, LangFuse tracing
- `frontend/app.py`: Streamlit UI for document upload, chat, graph visualization
- `dashboard/`: Separate Next.js + FastAPI app for knowledge gap analysis

## Common Commands

### Docker (Primary Method)
```bash
make start          # Build and start all services
make stop           # Stop all services
make restart        # Restart all services
make logs           # View all logs
make wipe           # Delete ALL data and restart fresh
make health         # Check service health
```

### Local Development
```bash
make dev-db         # Start only databases (Neo4j, Weaviate, LangFuse)
make dev-backend    # Run backend locally
make dev-frontend   # Run frontend locally
```

### Dashboard Development
```bash
# Backend (FastAPI + Celery)
cd dashboard/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080

# Frontend (Next.js)
cd dashboard/frontend
npm install
npm run dev         # Development server with Turbopack
npm run build       # Production build
npm run lint        # ESLint
npm run test        # Jest tests
```

### Testing
```bash
# Dashboard backend tests
cd dashboard/backend
pytest tests/ -v
pytest tests/ --cov=app   # With coverage

# Dashboard frontend tests
cd dashboard/frontend
npm run test
npm run test:coverage
```

### Data Management
```bash
make clear-knowledge       # Clear knowledge graph (keeps services running)
make langfuse-clear-traces # Clear LangFuse traces (keeps project/keys)
```

## Key Technical Details

### LangFuse SDK Compatibility
The codebase uses LangFuse SDK v3 directly while Cognee 0.4.1 expects SDK v2. A compatibility shim is implemented in `backend/main.py` that creates a fake `langfuse.decorators` module for Cognee.

### Cross-Thread Async Pattern
LangChain tools run in thread pool executors but need to call async Cognee/Neo4j functions. The solution uses `nest_asyncio` and `run_coroutine_threadsafe` to schedule coroutines on the main event loop.

### Entity Classification
Custom entity extraction prompt in `backend/main.py` classifies entities into: Person, Organization, Location, Concept, Technology, Product, Event, Document, Date, Metric.

### Environment Variables
Required:
- `OPENAI_API_KEY`: OpenAI API key

Optional (configured in docker-compose for local setup):
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`: After LangFuse first start, create project and API keys at http://localhost:3000
- `GRAPH_DATABASE_*`: Neo4j connection (pre-configured for Docker)

## Services and Ports

| Service | Port | Description |
|---------|------|-------------|
| Frontend (Streamlit) | 8501 | Main UI |
| Backend (FastAPI) | 8000 | API server |
| Neo4j HTTP | 7475 | Browser (neo4j/cogneepassword) |
| Neo4j Bolt | 7688 | Database connection |
| LangFuse | 3000 | Observability dashboard |
| Dashboard Frontend | 3001 | Knowledge gap analysis UI |
| Dashboard API | 8080 | Knowledge gap analysis API |
| Weaviate | 8081 | Vector database |

## API Endpoints (Backend)

- `GET /health` - Health check
- `POST /upload` - Upload documents
- `POST /add-text` - Add text directly
- `POST /query` - Query knowledge graph
- `POST /agent/query` - Query via LangChain agent
- `GET /graph` - Get graph data for visualization
- `DELETE /reset` - Reset knowledge base
