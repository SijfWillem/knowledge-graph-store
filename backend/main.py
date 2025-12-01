"""
Cognee RAG Backend API
Provides endpoints for document upload, knowledge graph building, and RAG queries.
"""
import os
import asyncio
import json
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import cognee
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import aiofiles

# Configuration
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global state for tracking processing
processing_status = {"status": "idle", "message": ""}


class QueryRequest(BaseModel):
    query: str
    search_type: str = "INSIGHTS"


class AddTextRequest(BaseModel):
    text: str
    dataset_name: str = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Cognee on startup."""
    # Reset any previous state
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)
    except Exception:
        pass  # Ignore errors on fresh start
    yield


app = FastAPI(
    title="Cognee RAG API",
    description="Hybrid RAG with Knowledge Graphs powered by Cognee",
    version="1.0.0",
    lifespan=lifespan
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/status")
async def get_status():
    """Get current processing status."""
    return processing_status


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    dataset_name: str = "default"
):
    """Upload one or more documents for processing."""
    global processing_status

    allowed_extensions = {".txt", ".pdf", ".md", ".json", ".csv", ".docx"}
    uploaded_files = []

    for file in files:
        file_ext = Path(file.filename).suffix.lower()

        if file_ext not in allowed_extensions:
            continue  # Skip unsupported files

        # Save file
        file_path = UPLOAD_DIR / file.filename
        async with aiofiles.open(file_path, "wb") as f:
            content = await file.read()
            await f.write(content)

        uploaded_files.append(str(file_path))

    if not uploaded_files:
        raise HTTPException(
            status_code=400,
            detail=f"No supported files uploaded. Allowed: {allowed_extensions}"
        )

    # Process in background
    processing_status = {"status": "processing", "message": f"Processing {len(uploaded_files)} file(s)..."}
    background_tasks.add_task(process_documents, uploaded_files, dataset_name)

    return {
        "message": f"{len(uploaded_files)} file(s) uploaded successfully",
        "files": [Path(f).name for f in uploaded_files],
        "status": "processing"
    }


async def process_documents(file_paths: list[str], dataset_name: str):
    """Process multiple documents with Cognee."""
    global processing_status

    try:
        total = len(file_paths)
        for i, file_path in enumerate(file_paths, 1):
            processing_status = {"status": "processing", "message": f"Adding document {i}/{total}: {Path(file_path).name}"}
            await cognee.add(file_path, dataset_name=dataset_name)

        processing_status = {"status": "processing", "message": "Building knowledge graph..."}
        await cognee.cognify()

        processing_status = {"status": "completed", "message": f"Processed {total} document(s) successfully"}

    except Exception as e:
        processing_status = {"status": "error", "message": str(e)}


async def process_document(file_path: str, dataset_name: str):
    """Process a single document with Cognee."""
    await process_documents([file_path], dataset_name)


@app.post("/add-text")
async def add_text(request: AddTextRequest, background_tasks: BackgroundTasks):
    """Add text directly to the knowledge base."""
    global processing_status

    processing_status = {"status": "processing", "message": "Adding text..."}
    background_tasks.add_task(process_text, request.text, request.dataset_name)

    return {"message": "Text added for processing", "status": "processing"}


async def process_text(text: str, dataset_name: str):
    """Process text with Cognee."""
    global processing_status

    try:
        await cognee.add(text, dataset_name=dataset_name)
        processing_status = {"status": "processing", "message": "Building knowledge graph..."}
        await cognee.cognify()
        processing_status = {"status": "completed", "message": "Text processed successfully"}
    except Exception as e:
        processing_status = {"status": "error", "message": str(e)}


@app.post("/query")
async def query_knowledge(request: QueryRequest):
    """Query the knowledge graph using RAG."""
    try:
        from cognee.modules.search.types import SearchType
        from neo4j import AsyncGraphDatabase

        # Map search type string to enum
        search_type_map = {
            "GRAPH_COMPLETION": SearchType.GRAPH_COMPLETION,
            "RAG_COMPLETION": SearchType.RAG_COMPLETION,
            "CHUNKS": SearchType.CHUNKS,
            "SUMMARIES": SearchType.SUMMARIES,
        }
        query_type = search_type_map.get(request.search_type, SearchType.GRAPH_COMPLETION)

        # Search using Cognee - GRAPH_COMPLETION returns the answer directly
        results = await cognee.search(request.query, query_type=query_type)

        # Extract the answer from results
        if results:
            if isinstance(results[0], str):
                # GRAPH_COMPLETION returns answer as string
                answer = results[0]
            elif hasattr(results[0], 'dict'):
                r = results[0].dict()
                answer = r.get('text') or r.get('content') or str(r)
            else:
                answer = str(results[0])
        else:
            answer = "No relevant information found in the knowledge base."

        # Get graph relationships as sources
        sources = []
        try:
            uri = os.environ.get("GRAPH_DATABASE_URL", "bolt://neo4j:7687")
            user = os.environ.get("GRAPH_DATABASE_USERNAME", "neo4j")
            password = os.environ.get("GRAPH_DATABASE_PASSWORD", "cogneepassword")

            driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            async with driver.session() as session:
                # Find relevant entities and their relationships
                result = await session.run("""
                    MATCH (n:Entity)-[r]->(m:Entity)
                    RETURN n.name as source, type(r) as relationship, m.name as target
                    LIMIT 10
                """)
                records = await result.data()
                for record in records:
                    sources.append(f"{record['source']} → {record['relationship']} → {record['target']}")
            await driver.close()
        except Exception:
            pass  # Sources are optional

        return {
            "query": request.query,
            "answer": answer,
            "sources": sources,
            "search_type": request.search_type
        }

    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


@app.get("/graph")
async def get_graph(show_all: bool = False):
    """Get the knowledge graph for visualization.

    By default, only shows Entity nodes and their relationships.
    Set show_all=true to see all node types (documents, chunks, summaries).
    """
    try:
        from neo4j import AsyncGraphDatabase

        uri = os.environ.get("GRAPH_DATABASE_URL", "bolt://neo4j:7687")
        user = os.environ.get("GRAPH_DATABASE_USERNAME", "neo4j")
        password = os.environ.get("GRAPH_DATABASE_PASSWORD", "cogneepassword")

        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))

        nodes = []
        edges = []
        seen_nodes = set()

        async with driver.session() as session:
            if show_all:
                # Show all nodes
                query = """
                    MATCH (n)
                    OPTIONAL MATCH (n)-[r]->(m)
                    RETURN n, r, m, labels(n) as n_labels, labels(m) as m_labels, type(r) as rel_type
                    LIMIT 500
                """
            else:
                # Only show Entity nodes and their relationships (the actual knowledge graph)
                query = """
                    MATCH (n:Entity)-[r]->(m:Entity)
                    RETURN n, r, m, labels(n) as n_labels, labels(m) as m_labels, type(r) as rel_type
                    LIMIT 500
                """

            result = await session.run(query)
            records = await result.data()

            for record in records:
                n = record.get('n')
                m = record.get('m')
                r = record.get('r')
                n_labels = record.get('n_labels', [])
                m_labels = record.get('m_labels', [])
                rel_type = record.get('rel_type')

                if n:
                    node_id = str(n.get('id', id(n)))
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        node_label = n.get('name') or n.get('text') or n.get('title') or node_id[:30]
                        # Get the most specific label (not __Node__)
                        node_type = next((l for l in n_labels if l != "__Node__"), "Entity")
                        nodes.append({
                            "id": node_id,
                            "label": str(node_label)[:50],
                            "type": node_type,
                            "data": {k: str(v)[:100] for k, v in n.items()}
                        })

                if m:
                    node_id = str(m.get('id', id(m)))
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        node_label = m.get('name') or m.get('text') or m.get('title') or node_id[:30]
                        node_type = next((l for l in m_labels if l != "__Node__"), "Entity")
                        nodes.append({
                            "id": node_id,
                            "label": str(node_label)[:50],
                            "type": node_type,
                            "data": {k: str(v)[:100] for k, v in m.items()}
                        })

                if r and n and m:
                    edges.append({
                        "source": str(n.get('id', id(n))),
                        "target": str(m.get('id', id(m))),
                        "label": rel_type or "related_to"
                    })

        await driver.close()

        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }

    except Exception as e:
        import traceback
        return {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.delete("/reset")
async def reset_knowledge_base():
    """Reset the entire knowledge base."""
    global processing_status

    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        # Clear uploads
        for file in UPLOAD_DIR.iterdir():
            if file.is_file():
                file.unlink()

        processing_status = {"status": "idle", "message": ""}

        return {"message": "Knowledge base reset successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/datasets")
async def list_datasets():
    """List all datasets."""
    try:
        from cognee.api.v1.datasets import datasets
        all_datasets = await datasets.list_datasets()
        return {"datasets": [d.name for d in all_datasets] if all_datasets else []}
    except Exception as e:
        return {"datasets": [], "error": str(e)}


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    try:
        documents = []
        for file in UPLOAD_DIR.iterdir():
            if file.is_file() and not file.name.startswith('.'):
                stat = file.stat()
                documents.append({
                    "name": file.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        # Sort by modification time, newest first
        documents.sort(key=lambda x: x["modified"], reverse=True)
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        return {"documents": [], "count": 0, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
