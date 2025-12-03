"""
Cognee RAG Backend API
Provides endpoints for document upload, knowledge graph building, and RAG queries.
Includes LangFuse observability for tracing LLM calls.
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

# LangFuse Observability - SDK v2 (compatible with Cognee 0.4.1)
# No-op decorator by default - works without any dependencies
def _noop_observe(*args, **kwargs):
    """No-op decorator when observability is disabled or unavailable."""
    def decorator(func):
        return func
    return decorator if not args else args[0]

def _noop_flush():
    """No-op flush when observability is disabled."""
    pass

class _NoopContext:
    """No-op context manager when observability is disabled."""
    @staticmethod
    def update_current_observation(**kwargs):
        pass
    @staticmethod
    def update_current_trace(**kwargs):
        pass
    @staticmethod
    def flush():
        pass

observe = _noop_observe
langfuse_flush = _noop_flush
langfuse_ctx = _NoopContext()
OBSERVABILITY_ENABLED = False

# Only attempt to import langfuse if explicitly enabled AND keys are set
if os.environ.get("MONITORING_TOOL") == "langfuse":
    _secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    _public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    _host = os.environ.get("LANGFUSE_HOST", "")

    if _secret_key and _public_key and not _secret_key.startswith("sk-lf-your"):
        try:
            # LangFuse SDK v2 import path (required for Cognee 0.4.1)
            from langfuse.decorators import observe as langfuse_observe, langfuse_context

            observe = langfuse_observe
            # Use langfuse_context.flush() for decorator-based tracing
            langfuse_flush = langfuse_context.flush
            langfuse_ctx = langfuse_context
            OBSERVABILITY_ENABLED = True
            print(f"✓ LangFuse observability enabled (SDK v2)")
            print(f"  Host: {_host}")
            print(f"  Public Key: {_public_key[:20]}...")
        except ImportError as e:
            print(f"⚠ LangFuse import failed: {e} - running without observability")
    else:
        print("⚠ LangFuse keys not configured - running without observability")

# Configuration
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global state for tracking processing
processing_status = {"status": "idle", "message": ""}


class QueryRequest(BaseModel):
    query: str
    search_type: str = "GRAPH_COMPLETION"
    top_k: int = 10


class AddTextRequest(BaseModel):
    text: str
    dataset_name: str = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Cognee on startup."""
    # NOTE: Prune calls commented out to preserve data across restarts
    # Uncomment if you want to reset the knowledge base on each startup
    # try:
    #     await cognee.prune.prune_data()
    #     await cognee.prune.prune_system(metadata=True)
    # except Exception:
    #     pass  # Ignore errors on fresh start
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
    return {
        "status": "healthy",
        "observability": {
            "enabled": OBSERVABILITY_ENABLED,
            "tool": os.environ.get("MONITORING_TOOL", "none")
        }
    }


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


@observe(name="process_documents", as_type="workflow")
async def process_documents(file_paths: list[str], dataset_name: str):
    """Process multiple documents with Cognee. Traced by LangFuse."""
    global processing_status

    try:
        total = len(file_paths)
        for i, file_path in enumerate(file_paths, 1):
            processing_status = {"status": "processing", "message": f"Adding document {i}/{total}: {Path(file_path).name}"}
            await cognee.add(file_path, dataset_name=dataset_name)

        processing_status = {"status": "processing", "message": "Building knowledge graph..."}
        await cognee.cognify()

        processing_status = {"status": "processing", "message": "Adding memory algorithms..."}
        await cognee.memify()

        processing_status = {"status": "completed", "message": f"Processed {total} document(s) successfully"}

    except Exception as e:
        processing_status = {"status": "error", "message": str(e)}
    finally:
        # Flush traces to ensure they're sent to LangFuse
        langfuse_flush()


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


@observe(name="process_text", as_type="workflow")
async def process_text(text: str, dataset_name: str):
    """Process text with Cognee. Traced by LangFuse."""
    global processing_status

    try:
        await cognee.add(text, dataset_name=dataset_name)
        processing_status = {"status": "processing", "message": "Building knowledge graph..."}
        await cognee.cognify()
        processing_status = {"status": "processing", "message": "Adding memory algorithms..."}
        await cognee.memify()
        processing_status = {"status": "completed", "message": "Text processed successfully"}
    except Exception as e:
        processing_status = {"status": "error", "message": str(e)}
    finally:
        # Flush traces to ensure they're sent to LangFuse
        langfuse_flush()


# Helper function to extract text from various object types
def extract_text(obj):
    """Extract readable text from various object types."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Try common text field names
        for key in ['text', 'content', 'chunk_text', 'summary', 'description', 'name', 'value']:
            if key in obj and obj[key] and isinstance(obj[key], str):
                return obj[key]
        return None
    # Handle Pydantic models (v1 and v2)
    if hasattr(obj, 'model_dump'):
        return extract_text(obj.model_dump())
    if hasattr(obj, 'dict'):
        return extract_text(obj.dict())
    # Handle objects with text attributes
    for attr in ['text', 'content', 'chunk_text', 'summary', 'description', 'name']:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val and isinstance(val, str):
                return val
    return None


def parse_graph_context(raw_context):
    """
    Parse the raw context from Cognee to extract nodes and relationships.
    Returns a structured dict with nodes, connections, and raw text.

    Cognee's only_context=True returns a list with a dict containing:
    {'all available datasets': 'Nodes:\nNode: name\n__node_content_start__\ncontent\n__node_content_end__\n\nConnections:\nsource --[rel]--> target\n...'}
    """
    import re

    nodes = []
    connections = []
    raw_text = ""

    if not raw_context:
        return {"nodes": [], "connections": [], "raw_context": ""}

    # Extract the formatted string from Cognee's response
    for item in raw_context:
        if isinstance(item, dict):
            # Check for the 'all available datasets' key (Cognee's format)
            for key, value in item.items():
                if isinstance(value, str) and ('Node:' in value or 'Nodes:' in value):
                    raw_text = value
                    break

    if not raw_text:
        # Fallback: try to get string content directly
        raw_text = str(raw_context)

    # Parse the Nodes section
    # Format: Node: name [tags]\n__node_content_start__\ncontent\n__node_content_end__
    node_pattern = r'Node:\s*([^\n]+)\n__node_content_start__\n(.*?)__node_content_end__'
    for match in re.finditer(node_pattern, raw_text, re.DOTALL):
        node_header = match.group(1).strip()
        node_content = match.group(2).strip()

        # Extract just the name (before any brackets)
        name_match = re.match(r'^([^\[]+)', node_header)
        node_name = name_match.group(1).strip() if name_match else node_header

        # Skip nodes with None or empty content
        if node_content and node_content.lower() != 'none':
            nodes.append({
                "name": node_name[:100],
                "type": "Entity",
                "content": node_content[:500]
            })

    # Parse the Connections section
    # Format: source --[relationship]--> target
    # Also handles: source --[relationship]--&gt; target (HTML encoded)
    connections_section = ""
    if "Connections:" in raw_text:
        connections_section = raw_text.split("Connections:")[-1]

    # Match patterns like: source --[rel]--> target or source --[rel]--&gt; target
    conn_pattern = r'([^\n-]+)\s*--\[([^\]]+)\]--(?:>|&gt;)\s*([^\n]+)'
    for match in re.finditer(conn_pattern, connections_section):
        source = match.group(1).strip()
        relationship = match.group(2).strip()
        target = match.group(3).strip()

        if source and target:
            connections.append({
                "source": source[:100],
                "relationship": relationship,
                "target": target[:100]
            })

    # Deduplicate nodes by name
    seen_nodes = set()
    unique_nodes = []
    for node in nodes:
        if node["name"] not in seen_nodes:
            seen_nodes.add(node["name"])
            unique_nodes.append(node)

    return {
        "nodes": unique_nodes,
        "connections": connections,
        "raw_context": raw_text
    }


# Helper function to serialize objects for tracing
def serialize_for_trace(obj):
    """Serialize an object for LangFuse tracing."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: serialize_for_trace(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_for_trace(item) for item in obj]
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    if hasattr(obj, 'dict'):
        return obj.dict()
    if hasattr(obj, '__dict__'):
        return {k: serialize_for_trace(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    return str(obj)


@observe(name="retrieve_context", as_type="retriever")
async def retrieve_context(query: str, query_type, top_k: int):
    """Retrieve context documents from the knowledge graph. Traced as retriever."""
    graph_context = {"nodes": [], "connections": [], "raw_context": ""}
    raw_results = []

    try:
        context_results = await cognee.search(
            query_text=query,
            query_type=query_type,
            top_k=top_k,
            only_context=True
        )

        # Serialize raw results for tracing
        raw_results = [serialize_for_trace(chunk) for chunk in (context_results or [])]

        # Parse the context to extract nodes and relationships
        graph_context = parse_graph_context(context_results)

        # Update the current observation with retrieval details
        langfuse_ctx.update_current_observation(
            input={"query": query, "top_k": top_k, "search_type": str(query_type)},
            output={
                "nodes": graph_context["nodes"],
                "connections": graph_context["connections"],
                "node_count": len(graph_context["nodes"]),
                "connection_count": len(graph_context["connections"])
            },
            metadata={
                "raw_results_count": len(raw_results),
                "raw_context_preview": graph_context["raw_context"][:1000] if graph_context["raw_context"] else "No raw context"
            }
        )
    except Exception as e:
        langfuse_ctx.update_current_observation(
            input={"query": query, "top_k": top_k},
            output={"error": str(e)},
            level="WARNING"
        )

    return graph_context, raw_results


@observe(name="generate_answer", as_type="generation")
async def generate_answer(query: str, query_type, top_k: int, graph_context: dict):
    """Generate answer using LLM with knowledge graph context. Traced as generation."""
    try:
        # Get the actual answer from Cognee
        results = await cognee.search(
            query_text=query,
            query_type=query_type,
            top_k=top_k
        )

        # Extract the answer from results
        if results:
            if isinstance(results[0], str):
                answer = results[0]
            elif hasattr(results[0], 'dict'):
                r = results[0].dict()
                answer = r.get('text') or r.get('content') or str(r)
            else:
                answer = str(results[0])
        else:
            answer = "No relevant information found in the knowledge base."

        # Get LLM config from environment for metadata
        llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

        # Build context preview for tracing
        context_preview = []
        for node in graph_context.get("nodes", [])[:5]:
            context_preview.append(f"{node['name']}: {node['content'][:100]}")
        for conn in graph_context.get("connections", [])[:5]:
            context_preview.append(f"{conn['source']} --[{conn['relationship']}]--> {conn['target']}")

        # Update generation with details
        langfuse_ctx.update_current_observation(
            input={
                "query": query,
                "context_nodes": len(graph_context.get("nodes", [])),
                "context_connections": len(graph_context.get("connections", [])),
                "context_preview": context_preview
            },
            output=answer,
            model=llm_model,
            metadata={
                "search_type": str(query_type),
                "top_k": top_k,
                "answer_length": len(answer)
            }
        )

        return answer

    except Exception as e:
        langfuse_ctx.update_current_observation(
            input={"query": query},
            output={"error": str(e)},
            level="ERROR"
        )
        raise


@app.post("/query")
@observe(name="query_knowledge")
async def query_knowledge(request: QueryRequest):
    """Query the knowledge graph using RAG. Traced by LangFuse with full details."""
    try:
        from cognee import SearchType

        # Map search type string to enum
        search_type_map = {
            "GRAPH_COMPLETION": SearchType.GRAPH_COMPLETION,
            "RAG_COMPLETION": SearchType.RAG_COMPLETION,
            "CHUNKS": SearchType.CHUNKS,
            "CHUNKS_LEXICAL": SearchType.CHUNKS_LEXICAL,
            "SUMMARIES": SearchType.SUMMARIES,
            "CODE": SearchType.CODE,
            "GRAPH_COMPLETION_COT": SearchType.GRAPH_COMPLETION_COT,
        }
        query_type = search_type_map.get(request.search_type, SearchType.GRAPH_COMPLETION)

        # Update the main trace with input
        langfuse_ctx.update_current_trace(
            input={
                "query": request.query,
                "search_type": request.search_type,
                "top_k": request.top_k
            },
            metadata={
                "endpoint": "/query",
                "search_type_enum": str(query_type)
            }
        )

        # Step 1: Retrieve context from knowledge graph (traced as retriever)
        graph_context, raw_results = await retrieve_context(
            query=request.query,
            query_type=query_type,
            top_k=request.top_k
        )

        # Step 2: Generate answer using LLM (traced as generation)
        answer = await generate_answer(
            query=request.query,
            query_type=query_type,
            top_k=request.top_k,
            graph_context=graph_context
        )

        # Build response with structured graph context
        response = {
            "query": request.query,
            "answer": answer,
            "search_type": request.search_type,
            # Structured graph context for display
            "graph_context": {
                "nodes": graph_context.get("nodes", []),
                "connections": graph_context.get("connections", [])
            },
            # Legacy field for backwards compatibility
            "retrieved_documents": [
                {"text": f"{node['name']}: {node['content']}"}
                for node in graph_context.get("nodes", [])
                if node.get("content")
            ]
        }

        # Update trace with final output
        langfuse_ctx.update_current_trace(
            output={
                "answer": answer,
                "nodes_retrieved": len(graph_context.get("nodes", [])),
                "connections_retrieved": len(graph_context.get("connections", []))
            }
        )

        # Flush traces before returning
        langfuse_flush()
        return response

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"

        # Update trace with error
        langfuse_ctx.update_current_trace(
            output={"error": str(e)},
            metadata={"traceback": traceback.format_exc()[:1000]}
        )

        # Flush traces even on error
        langfuse_flush()
        raise HTTPException(status_code=500, detail=error_detail)


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
