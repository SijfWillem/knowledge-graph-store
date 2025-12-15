# Cognee Memory - Knowledge Graph System

A knowledge graph-based memory system powered by [Cognee](https://github.com/topoteretes/cognee) that extracts entities and relationships from documents for intelligent Q&A with full observability and **Knowledge Gap Intelligence Dashboard**.

## Features

### Core RAG System
- **Document Upload** - Upload PDF, TXT, MD, JSON, CSV, DOCX files
- **Knowledge Graph** - Automatically extracts entities and relationships from documents
- **Graph-Based Q&A** - Ask questions and get answers with visible graph context (nodes & relationships)
- **Graph Visualization** - Interactive visualization of the knowledge graph
- **LangFuse Observability** - Full tracing of queries, retrievals, and LLM generations
- **Multiple Search Modes** - GRAPH_COMPLETION, CHUNKS, SUMMARIES, and more

### Knowledge Gap Intelligence Dashboard
- **Dynamic Topic Discovery** - Automatically discovers topics from questions using BERTopic ML algorithm
- **Missing Knowledge Detection** - Identifies when the AI admits it doesn't know something
- **Knowledge Gap Tracking** - Tracks and prioritizes gaps in your knowledge base
- **Topic Analytics** - Success rates, confidence scores, and trends per topic
- **Quality Metrics** - Measures empathy, understanding, relevancy, clarity, and more
- **Satisfaction Tracking** - Thumbs up/down and CSAT distribution
- **Real-time Sync** - Syncs conversation data from LangFuse for analysis

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
                                                   │
                        ┌──────────────────────────┘
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              Knowledge Gap Dashboard                             │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Dashboard     │   Dashboard     │      Dashboard              │
│   Frontend      │   Backend       │      PostgreSQL             │
│   (Next.js)     │   (FastAPI)     │      (Analytics DB)         │
│   Port: 3001    │   Port: 8080    │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
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
- Start Neo4j, LangFuse, Backend, Frontend, and Knowledge Gap Dashboard
- Display access URLs

### 3. Access

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend UI** | http://localhost:8501 | Main Cognee chat interface |
| **Knowledge Gap Dashboard** | http://localhost:3001 | Analytics and gap intelligence |
| **Backend API** | http://localhost:8000 | Cognee RAG API |
| **Dashboard API** | http://localhost:8080 | Dashboard analytics API |
| **Neo4j Browser** | http://localhost:7475 | Graph database (neo4j/cogneepassword) |
| **LangFuse** | http://localhost:3000 | Observability dashboard |

## Usage

### Upload Documents & Ask Questions

1. Open http://localhost:8501
2. Use the sidebar to upload documents
3. Go to "Chat" tab and ask questions
4. View answers with Knowledge Graph Context

### Knowledge Gap Dashboard

1. Open http://localhost:3001
2. Click **"Refresh Data"** to sync conversations from LangFuse
3. Click **"Discover Topics"** to auto-discover topics from questions
4. View:
   - **Success Rate** - Percentage of successfully answered questions
   - **Topics Overview** - Auto-discovered topics with analytics
   - **Knowledge Gaps** - Questions the AI couldn't answer
   - **Quality Metrics** - Response quality scores
   - **Satisfaction** - User feedback analytics

### Dynamic Topic Discovery

The dashboard uses **BERTopic** to automatically discover topics from user questions:

- Questions are normalized to focus on **themes** rather than specific entities
- Example: "how old is John?" and "how old is Mary?" → grouped as "Age Questions"
- Topics are labeled based on meaningful keywords (filtering out names)
- Works with as few as 5 questions, scales to thousands

### Missing Knowledge Detection

The system automatically detects when the AI admits it doesn't know:

- Pattern matching for phrases like "I don't know", "no information available"
- Missing knowledge is reflected in success rate metrics
- Flagged conversations appear in Knowledge Gaps

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
| `make dashboard` | Start only the dashboard services |

## Project Structure

```
knowledge-cognee/
├── docker-compose.yml      # Container orchestration
├── Makefile                # Build and run commands
├── .env.example            # Environment template
├── README.md               # This file
├── CLAUDE.md               # AI assistant context
├── uploads/                # Uploaded documents
├── backend/                # Cognee RAG Backend
│   ├── Dockerfile
│   ├── requirements.txt
│   └── main.py             # FastAPI + Cognee + LangFuse integration
├── frontend/               # Streamlit UI
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
└── dashboard/              # Knowledge Gap Dashboard
    ├── backend/
    │   ├── Dockerfile
    │   ├── requirements.txt
    │   └── app/
    │       ├── main.py                 # FastAPI dashboard API
    │       ├── models/database.py      # SQLAlchemy models
    │       └── services/
    │           ├── analytics_service.py      # Metrics calculation
    │           ├── dynamic_topic_modeler.py  # BERTopic integration
    │           ├── langfuse_service.py       # LangFuse sync
    │           ├── gap_analyzer.py           # Knowledge gap detection
    │           └── quality_scorer.py         # Response quality scoring
    └── frontend/
        ├── Dockerfile
        └── src/
            ├── app/page.tsx            # Main dashboard page
            ├── components/             # React components
            └── lib/api.ts              # API client
```

## API Endpoints

### Cognee Backend (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload documents |
| `/query` | POST | Query the knowledge graph |
| `/graph` | GET | Get graph visualization data |

### Dashboard Backend (Port 8080)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analytics/overview` | GET | Dashboard metrics overview |
| `/api/analytics/conversations` | GET | List conversations with Q&A |
| `/api/analytics/trends` | GET | Conversation trends over time |
| `/api/topics/` | GET | List discovered topics with analytics |
| `/api/topics/{name}` | GET | Topic detail with sample questions |
| `/api/knowledge-gaps/` | GET | List knowledge gaps |
| `/api/quality/scores` | GET | Quality metric scores |
| `/api/sync/now` | POST | Sync data from LangFuse |
| `/api/sync/discover-topics` | POST | Run BERTopic discovery |
| `/api/sync/detect-gaps` | POST | Detect knowledge gaps |
| `/api/sync/update-topics` | POST | Recalculate topic analytics |

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
LANGFUSE_BASE_URL=http://localhost:3000
```

### Setting up LangFuse API Keys

1. Start services: `make start`
2. Open http://localhost:3000
3. Create an account and project
4. Go to **Settings > API Keys > Create new API key**
5. Copy keys to `.env` file (no spaces around `=`)
6. Restart: `make restart`

## Tech Stack

### Core RAG
- **Cognee** - Knowledge graph extraction and search
- **Neo4j** - Graph database for entities and relationships
- **LanceDB** - Vector database for embeddings
- **FastAPI** - Backend API
- **Streamlit** - Frontend UI
- **OpenAI** - LLM for extraction and generation

### Knowledge Gap Dashboard
- **Next.js 14** - React framework with App Router
- **TailwindCSS** - Styling
- **React Query** - Data fetching
- **Recharts** - Charts and visualizations
- **BERTopic** - ML-based topic discovery
- **Sentence Transformers** - Text embeddings
- **UMAP + HDBSCAN** - Dimensionality reduction and clustering
- **PostgreSQL** - Analytics database
- **LangFuse** - Observability and tracing

## Troubleshooting

### Services won't start
```bash
make wipe
make start
```

### LangFuse 401 Unauthorized
1. Check API keys in `.env` match those in LangFuse dashboard
2. Ensure no spaces around `=` in `.env` file
3. Unset shell environment variables: `unset LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY`
4. Restart: `docker-compose up -d dashboard-backend`

### Topics not discovering
- Need at least 5 questions for BERTopic to work
- Click "Discover Topics" button after syncing data
- Check logs: `docker logs dashboard-backend`

### Neo4j unhealthy
The health check has a 60-second start period. Wait for initialization to complete.

## License

MIT
