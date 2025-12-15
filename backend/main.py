"""
Cognee Knowledge Graph Backend API
Provides endpoints for document upload, knowledge graph building, and intelligent queries.
Includes LangChain agent orchestration and LangFuse observability for tracing.
"""
import os
import asyncio
import json
from pathlib import Path
from typing import Optional, List, Any
from contextlib import asynccontextmanager

# Apply nest_asyncio to allow nested event loops
# This is required because LangChain tools run in thread pool executors
# but need to call async Cognee/Neo4j functions that are tied to the main event loop
import nest_asyncio
nest_asyncio.apply()
print("✓ nest_asyncio applied for nested event loop support")

# Configure Cognee data directories BEFORE importing cognee
# This ensures the database is created in the mounted volume
COGNEE_DATA_DIR = Path("/root/.cognee_system")
COGNEE_DATA_DIR.mkdir(parents=True, exist_ok=True)
(COGNEE_DATA_DIR / "databases").mkdir(exist_ok=True)

# Patch Cognee's observability to work with LangFuse SDK v3
# Cognee 0.4.1 uses SDK v2 API (langfuse.decorators) which doesn't exist in v3
# We create a compatibility shim before importing cognee
import sys
from types import ModuleType

# Create the langfuse.decorators module shim for Cognee compatibility
if 'langfuse.decorators' not in sys.modules:
    # Create a no-op observe decorator that matches Cognee's expected interface
    def _cognee_observe(*args, **kwargs):
        """Compatibility shim for Cognee's langfuse.decorators.observe"""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        def decorator(func):
            return func
        return decorator

    # Create the fake module
    decorators_module = ModuleType('langfuse.decorators')
    decorators_module.observe = _cognee_observe

    # Register it so imports find it
    sys.modules['langfuse.decorators'] = decorators_module
    print("✓ Created langfuse.decorators compatibility shim for Cognee")

import cognee

# Now configure cognee to use our data directory
cognee.config.system_root_directory(str(COGNEE_DATA_DIR))
print(f"✓ Cognee system directory set to: {COGNEE_DATA_DIR}")
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import aiofiles

# LangFuse Observability - SDK v3
# No-op decorator by default - works without any dependencies
def _noop_observe(*args, **kwargs):
    """No-op decorator when observability is disabled or unavailable."""
    def decorator(func):
        return func
    return decorator if not args else args[0]

class _NoopClient:
    """No-op client when observability is disabled."""
    @staticmethod
    def update_current_span(**kwargs):
        pass
    @staticmethod
    def update_current_generation(**kwargs):
        pass
    @staticmethod
    def update_current_trace(**kwargs):
        pass
    @staticmethod
    def get_current_trace_id():
        return None
    @staticmethod
    def get_current_observation_id():
        return None
    @staticmethod
    def flush():
        pass

observe = _noop_observe
langfuse_client = _NoopClient()
OBSERVABILITY_ENABLED = False

# Only attempt to import langfuse if explicitly enabled AND keys are set
# Note: We use LANGFUSE_ENABLED instead of MONITORING_TOOL because Cognee uses
# MONITORING_TOOL internally with SDK v2, while we use SDK v3 directly
_langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "").lower() in ("true", "1", "yes")
if _langfuse_enabled:
    _secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "")
    _public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
    _host = os.environ.get("LANGFUSE_HOST", "")

    if _secret_key and _public_key and not _secret_key.startswith("sk-lf-your"):
        try:
            # LangFuse SDK v3 import path
            from langfuse import observe as langfuse_observe, get_client

            observe = langfuse_observe
            langfuse_client = get_client()
            OBSERVABILITY_ENABLED = True
            print(f"✓ LangFuse observability enabled (SDK v3)")
            print(f"  Host: {_host}")
            print(f"  Public Key: {_public_key[:20]}...")
        except ImportError as e:
            print(f"⚠ LangFuse import failed: {e} - running without observability")
    else:
        print("⚠ LangFuse keys not configured - running without observability")

# ===================
# LangChain Agent Setup
# ===================
LANGCHAIN_ENABLED = False
langchain_agent = None
langfuse_callback_handler = None
LangfuseCallbackHandler = None  # Will be set if langfuse.langchain is available

try:
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool
    from langchain.agents import create_agent

    # LangFuse callback for LangChain (SDK v3)
    # Note: We create callback handlers per-request to link to the trace context
    if OBSERVABILITY_ENABLED:
        try:
            from langfuse.langchain import CallbackHandler as _LangfuseCallbackHandler
            LangfuseCallbackHandler = _LangfuseCallbackHandler
            langfuse_callback_handler = True  # Flag to indicate callback is available
            print("✓ LangFuse callback handler for LangChain available (SDK v3)")
        except ImportError as e:
            print(f"⚠ LangFuse callback handler not available for LangChain: {e}")

    LANGCHAIN_ENABLED = True
    print("✓ LangChain agent framework loaded")
except ImportError as e:
    print(f"⚠ LangChain not available: {e}")

# Configuration
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global state for tracking processing
processing_status = {"status": "idle", "message": ""}

# Custom prompt for entity extraction with type classification
# This instructs the LLM to classify entities into specific types
ENTITY_EXTRACTION_PROMPT = """
Extract entities and relationships from the text. For each entity, classify it into one of these types:

ENTITY TYPES:
- Person: Individual people, characters, named individuals
- Organization: Companies, institutions, teams, groups, agencies
- Location: Places, cities, countries, addresses, geographical features
- Concept: Abstract ideas, theories, methodologies, principles
- Technology: Software, hardware, tools, platforms, programming languages
- Product: Physical or digital products, services, offerings
- Event: Meetings, conferences, incidents, historical events
- Document: Reports, papers, articles, books, specifications
- Date: Specific dates, time periods, deadlines
- Metric: Numbers, statistics, measurements, KPIs

RULES:
1. Always assign the most specific type that fits
2. Extract relationships between entities (e.g., "works_at", "located_in", "created_by")
3. Include descriptive attributes when available
4. Preserve the original names exactly as they appear in the text

For each entity, provide:
- name: The entity's name
- type: One of the types listed above
- description: Brief description of the entity based on context
"""

# Store the main event loop for cross-thread async calls
# This is needed because LangChain tools run in thread pool executors
# but need to call async Cognee/Neo4j functions bound to the main loop
_main_event_loop = None


class QueryRequest(BaseModel):
    query: str
    search_type: str = "GRAPH_COMPLETION"
    top_k: int = 10


class AddTextRequest(BaseModel):
    text: str
    dataset_name: str = "default"


class AgentQueryRequest(BaseModel):
    """Request model for agent-based queries."""
    query: str = Field(..., description="The user's question or request")
    use_agent: bool = Field(default=True, description="Whether to use the LangChain agent")
    verbose: bool = Field(default=False, description="Enable verbose agent output")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Cognee on startup."""
    global _main_event_loop

    # Capture the main event loop for cross-thread async calls
    _main_event_loop = asyncio.get_running_loop()
    print("✓ Main event loop captured for cross-thread async calls")

    # Initialize Cognee database (creates tables if they don't exist)
    try:
        from cognee.infrastructure.databases.relational import create_db_and_tables
        await create_db_and_tables()
        print("✓ Cognee database initialized")
    except Exception as e:
        print(f"⚠ Cognee setup warning (may already be initialized): {e}")

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


@observe(name="process_documents", as_type="chain")
async def process_documents(file_paths: list[str], dataset_name: str):
    """Process multiple documents with Cognee. Traced by LangFuse."""
    global processing_status

    try:
        total = len(file_paths)
        for i, file_path in enumerate(file_paths, 1):
            processing_status = {"status": "processing", "message": f"Adding document {i}/{total}: {Path(file_path).name}"}
            await cognee.add(file_path, dataset_name=dataset_name)

        processing_status = {"status": "processing", "message": "Building knowledge graph with entity classification..."}
        await cognee.cognify(custom_prompt=ENTITY_EXTRACTION_PROMPT)

        processing_status = {"status": "processing", "message": "Adding memory algorithms..."}
        await cognee.memify()

        processing_status = {"status": "completed", "message": f"Processed {total} document(s) successfully"}

    except Exception as e:
        processing_status = {"status": "error", "message": str(e)}
    finally:
        # Flush traces to ensure they're sent to LangFuse
        langfuse_client.flush()


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


@observe(name="process_text", as_type="chain")
async def process_text(text: str, dataset_name: str):
    """Process text with Cognee. Traced by LangFuse."""
    global processing_status

    try:
        await cognee.add(text, dataset_name=dataset_name)
        processing_status = {"status": "processing", "message": "Building knowledge graph with entity classification..."}
        await cognee.cognify(custom_prompt=ENTITY_EXTRACTION_PROMPT)
        # processing_status = {"status": "processing", "message": "Adding memory algorithms..."}
        # await cognee.memify()
        processing_status = {"status": "completed", "message": "Text processed successfully"}
    except Exception as e:
        processing_status = {"status": "error", "message": str(e)}
    finally:
        # Flush traces to ensure they're sent to LangFuse
        langfuse_client.flush()


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

    # Valid entity types for classification
    VALID_ENTITY_TYPES = {
        "Person", "Organization", "Location", "Concept", "Technology",
        "Product", "Event", "Document", "Date", "Metric"
    }

    def extract_entity_type(header, content):
        """Extract entity type from node header tags or content."""
        # Check for tags in header like [Person] or [Organization]
        tag_match = re.search(r'\[([^\]]+)\]', header)
        if tag_match:
            tag = tag_match.group(1).strip()
            # Check if tag matches a valid entity type
            for valid_type in VALID_ENTITY_TYPES:
                if valid_type.lower() in tag.lower():
                    return valid_type

        # Check content for type hints (order matters - more specific checks first)
        content_lower = content.lower() if content else ""
        name_lower = header.lower() if header else ""

        # Person detection (check first - titles and roles indicate persons)
        person_titles = ['ceo', 'cto', 'cfo', 'founder', 'director', 'manager', 'president',
                        'chairman', 'officer', 'executive', 'employee', 'worker', 'staff']
        person_words = ['person', 'individual', 'human', 'man', 'woman', 'people']

        # Check if description mentions someone IS/WAS a title (strong person indicator)
        for title in person_titles:
            if f'is {title}' in content_lower or f'the {title}' in content_lower or f'as {title}' in content_lower:
                return 'Person'
            # Check if description starts with title (e.g., "CEO of...")
            if content_lower.startswith(title):
                return 'Person'

        if any(word in content_lower for word in person_words):
            return 'Person'

        # Location detection (check before Organization - cities/countries are locations)
        location_words = ['city', 'country', 'location', 'place', 'region', 'address',
                         'state', 'province', 'town', 'village', 'capital', 'located in']
        if any(word in content_lower for word in location_words):
            return 'Location'

        # Common city/country names in the entity name itself
        known_locations = ['francisco', 'york', 'angeles', 'london', 'paris', 'tokyo',
                          'berlin', 'chicago', 'boston', 'seattle', 'amsterdam']
        if any(loc in name_lower for loc in known_locations):
            return 'Location'

        # Organization detection
        org_words = ['company', 'organization', 'corporation', 'team', 'agency',
                    'institution', 'firm', 'enterprise', 'business', 'inc', 'llc', 'ltd']
        if any(word in content_lower for word in org_words):
            return 'Organization'

        # Technology detection
        tech_words = ['software', 'tool', 'platform', 'framework', 'language', 'api',
                     'system', 'application', 'app', 'technology', 'tech']
        if any(word in content_lower for word in tech_words):
            return 'Technology'

        # Product detection
        product_words = ['product', 'service', 'offering', 'solution', 'feature']
        if any(word in content_lower for word in product_words):
            return 'Product'

        # Event detection
        event_words = ['event', 'meeting', 'conference', 'summit', 'workshop',
                      'seminar', 'webinar', 'incident', 'ceremony']
        if any(word in content_lower for word in event_words):
            return 'Event'

        # Concept detection
        concept_words = ['concept', 'idea', 'theory', 'methodology', 'principle',
                        'approach', 'strategy', 'philosophy']
        if any(word in content_lower for word in concept_words):
            return 'Concept'

        # Date/Time detection
        date_words = ['date', 'time', 'period', 'month', 'year', 'day', 'week',
                     'quarter', 'deadline', 'schedule']
        if any(word in content_lower for word in date_words):
            return 'Date'

        # Metric detection
        metric_words = ['metric', 'number', 'statistic', 'kpi', 'measurement',
                       'percentage', 'rate', 'count', 'total', 'average']
        if any(word in content_lower for word in metric_words):
            return 'Metric'

        return "Entity"

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

        # Extract entity type from header tags or content
        entity_type = extract_entity_type(node_header, node_content)

        # Skip nodes with None or empty content
        if node_content and node_content.lower() != 'none':
            nodes.append({
                "name": node_name[:100],
                "type": entity_type,
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

        # Update the current span with retrieval details (SDK v3 uses update_current_span)
        langfuse_client.update_current_span(
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
        langfuse_client.update_current_span(
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

        # Update generation with details (SDK v3 uses update_current_generation)
        langfuse_client.update_current_generation(
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
        langfuse_client.update_current_span(
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
        langfuse_client.update_current_trace(
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
        langfuse_client.update_current_trace(
            output={
                "answer": answer,
                "nodes_retrieved": len(graph_context.get("nodes", [])),
                "connections_retrieved": len(graph_context.get("connections", []))
            }
        )

        # Flush traces before returning
        langfuse_client.flush()
        return response

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"

        # Update trace with error
        langfuse_client.update_current_trace(
            output={"error": str(e)},
            metadata={"traceback": traceback.format_exc()[:1000]}
        )

        # Flush traces even on error
        langfuse_client.flush()
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/graph")
async def get_graph(show_all: bool = False):
    """Get the knowledge graph for visualization.

    By default, only shows Entity nodes and their relationships.
    Set show_all=true to see all node types (documents, chunks, summaries).

    Entity types are extracted from the 'type' property set during cognify:
    Person, Organization, Location, Concept, Technology, Product, Event, Document, Date, Metric
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

        # Valid entity types from our custom prompt
        VALID_ENTITY_TYPES = {
            "Person", "Organization", "Location", "Concept", "Technology",
            "Product", "Event", "Document", "Date", "Metric"
        }

        def get_entity_type(node_props, labels):
            """Extract the entity type from node properties or labels.

            Priority:
            1. 'type' property if it's a valid entity type
            2. 'entity_type' property
            3. Neo4j label (excluding __Node__ and Entity)
            4. Content-based heuristics
            5. Default to 'Entity'
            """
            # Check type property first (Cognee stores entity classification here)
            node_type = node_props.get('type', '')
            if node_type in VALID_ENTITY_TYPES:
                return node_type

            # Check entity_type property
            entity_type = node_props.get('entity_type', '')
            if entity_type in VALID_ENTITY_TYPES:
                return entity_type

            # Check if type property contains a valid type (case-insensitive)
            if node_type:
                for valid_type in VALID_ENTITY_TYPES:
                    if valid_type.lower() in node_type.lower():
                        return valid_type

            # Fall back to Neo4j labels
            for label in labels:
                if label not in ("__Node__", "Entity") and label in VALID_ENTITY_TYPES:
                    return label

            # Check labels for partial matches
            for label in labels:
                if label not in ("__Node__", "Entity"):
                    for valid_type in VALID_ENTITY_TYPES:
                        if valid_type.lower() in label.lower():
                            return valid_type

            # Content-based heuristics (order matters - check specific types first)
            name = str(node_props.get('name', '')).lower()
            description = str(node_props.get('description', '')).lower()

            # Metric detection (check first - numbers, ages, attendance)
            metric_patterns = ['attendance', 'number', 'count', 'total', 'approximate', 'percentage',
                              'over ', 'about ', 'around ', 'approximately']
            if any(word in description for word in metric_patterns):
                return 'Metric'
            # Age values
            if 'years old' in name or 'age value' in description:
                return 'Metric'
            # Pattern: "over X people", "500 people", etc.
            if 'people' in name or 'people' in description.split('.')[0]:
                return 'Metric'

            # Person detection (check early - titles indicate persons)
            person_titles = ['ceo', 'cto', 'cfo', 'founder', 'director', 'manager', 'president',
                            'chairman', 'officer', 'executive', 'employee', 'coworker', 'worker']
            for title in person_titles:
                if f'is {title}' in description or f'the {title}' in description or description.startswith(title):
                    return 'Person'
            # Person patterns: age + working/relationship
            if 'years old' in description and any(word in description for word in ['working', 'works', 'worked', 'relationship', 'coworker', 'interested']):
                return 'Person'
            if any(word in description for word in ['person', 'individual', 'human']):
                return 'Person'

            # Date detection (specific patterns - NOT matching "years old")
            date_patterns = ['time period', 'period when', 'last month', 'next month', 'this year',
                            'last year', 'next year', 'deadline', 'scheduled for']
            if any(pattern in description for pattern in date_patterns):
                return 'Date'
            if name in ['last month', 'this month', 'next month', 'last year', 'this year', 'next year']:
                return 'Date'

            # Location detection
            location_words = ['city', 'country', 'location', 'place', 'region', 'address', 'state', 'capital', 'base location']
            if any(word in description for word in location_words):
                return 'Location'
            # Known location names
            known_locations = ['francisco', 'york', 'angeles', 'london', 'paris', 'tokyo', 'berlin', 'chicago', 'boston']
            if any(loc in name for loc in known_locations):
                return 'Location'

            # Product detection (check BEFORE Event - products may mention launch events)
            # Name-based product detection (high priority)
            if any(word in name for word in ['assist', 'app', 'tool', 'software', 'platform', 'product']):
                return 'Product'
            if description.startswith('product') or description.startswith('new product') or description.startswith('a product'):
                return 'Product'

            # Event detection (check BEFORE Organization - many events mention organizations)
            event_words = ['conference', 'event', 'meeting', 'summit', 'workshop', 'seminar']
            # Only match Event if the NAME suggests it's an event, or it's clearly about the event itself
            if any(word in name for word in ['conference', 'summit', 'event', 'meeting', 'annual']):
                return 'Event'
            # Match if description is ABOUT an event (not just mentioning one)
            if any(word in description[:30] for word in event_words):
                return 'Event'
            # Pattern: "new X announced by", "X developed by", "X created by"
            if any(pattern in description for pattern in ['announced by', 'developed by', 'created by', 'made by', 'built by']):
                return 'Product'
            # "New product announced" pattern
            if 'new product' in description or 'announced' in description[:50]:
                return 'Product'
            # "X products developed by" - the entity IS a product type
            if 'products' in name or (description.startswith(name) and 'products' in description[:len(name)+15]):
                return 'Product'

            # Organization detection - entity IS the org (not just mentions one)
            # Check if description says THIS entity is a company/org
            org_patterns = [
                'is a company', 'is an organization', 'is a corporation', 'is a firm',
                'a company', 'technology company', 'software company', 'is a team'
            ]
            if description.startswith(('a ', 'an ', 'the ')) and any(word in description[:50] for word in ['company', 'organization', 'corporation', 'firm', 'enterprise']):
                return 'Organization'
            if any(pattern in description for pattern in org_patterns):
                return 'Organization'
            # Name ends with common org suffixes
            if any(name.endswith(suffix) for suffix in [' inc', ' corp', ' corporation', ' llc', ' ltd', ' company']):
                return 'Organization'
            # Fallback: only match if "company" etc appears at the START of description
            org_words = ['company', 'organization', 'corporation', 'firm', 'enterprise']
            first_sentence = description.split('.')[0] if '.' in description else description
            if any(f' is a {word}' in first_sentence or first_sentence.startswith(word) for word in org_words):
                return 'Organization'

            # Technology detection
            tech_words = ['artificial intelligence', 'machine learning', 'programming language',
                         'framework', 'api', 'protocol']
            if any(word in description for word in tech_words):
                return 'Technology'
            if name in ['ai', 'ml', 'python', 'javascript', 'java', 'react', 'api']:
                return 'Technology'
            # AI as area of interest
            if 'area of interest' in description or 'interested in' in description:
                if 'ai' in description or 'artificial intelligence' in description:
                    return 'Technology'

            return "Entity"

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
                        # Get entity type from properties or labels
                        node_type = get_entity_type(dict(n), n_labels)
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
                        node_type = get_entity_type(dict(m), m_labels)
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
    """Reset the entire knowledge base with comprehensive cleanup.

    Clears:
    1. Neo4j knowledge graph (all nodes and relationships)
    2. Uploaded files
    3. Cognee internal data and caches

    Returns step-by-step progress for frontend display.
    """
    global processing_status

    steps = []
    errors = []

    # Step 1: Clear Neo4j graph
    try:
        from neo4j import AsyncGraphDatabase

        uri = os.environ.get("GRAPH_DATABASE_URL", "bolt://neo4j:7687")
        user = os.environ.get("GRAPH_DATABASE_USERNAME", "neo4j")
        password = os.environ.get("GRAPH_DATABASE_PASSWORD", "cogneepassword")

        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        async with driver.session() as session:
            await session.run("MATCH (n) DETACH DELETE n")
        await driver.close()

        steps.append({"step": "neo4j", "status": "success", "message": "Neo4j graph cleared"})
    except Exception as e:
        steps.append({"step": "neo4j", "status": "error", "message": str(e)})
        errors.append(f"Neo4j: {str(e)}")

    # Step 2: Clear uploaded files
    try:
        cleared_files = 0
        for file in UPLOAD_DIR.iterdir():
            if file.is_file() and not file.name.startswith('.'):
                file.unlink()
                cleared_files += 1

        steps.append({"step": "uploads", "status": "success", "message": f"Cleared {cleared_files} uploaded file(s)"})
    except Exception as e:
        steps.append({"step": "uploads", "status": "error", "message": str(e)})
        errors.append(f"Uploads: {str(e)}")

    # Step 3: Clear Cognee data
    try:
        await cognee.prune.prune_data()
        await cognee.prune.prune_system(metadata=True)

        steps.append({"step": "cognee", "status": "success", "message": "Cognee data cleared"})
    except Exception as e:
        steps.append({"step": "cognee", "status": "error", "message": str(e)})
        errors.append(f"Cognee: {str(e)}")

    # Reset processing status
    processing_status = {"status": "idle", "message": ""}

    # Return detailed response
    success = len(errors) == 0
    return {
        "success": success,
        "message": "Knowledge base reset successfully" if success else f"Reset completed with errors: {'; '.join(errors)}",
        "steps": steps
    }


@app.delete("/reset-langfuse-traces")
async def reset_langfuse_traces():
    """Clear all LangFuse traces while preserving project, users, and API keys.

    Clears ClickHouse tables:
    - traces
    - observations
    - scores
    - analytics_traces
    - analytics_observations
    - analytics_scores
    - event_log

    Returns step-by-step progress for frontend display.
    """
    import httpx

    steps = []
    errors = []

    # ClickHouse connection details (from docker-compose)
    clickhouse_url = "http://langfuse-clickhouse:8123"
    clickhouse_user = "clickhouse"
    clickhouse_password = "clickhousepassword"

    # Tables to truncate (only actual tables, not views)
    # Note: analytics_* are views derived from main tables and can't be truncated
    tables = [
        "traces",
        "observations",
        "scores",
        "event_log"
    ]

    async with httpx.AsyncClient() as client:
        for table in tables:
            try:
                response = await client.post(
                    clickhouse_url,
                    params={
                        "user": clickhouse_user,
                        "password": clickhouse_password,
                        "query": f"TRUNCATE TABLE IF EXISTS {table}"
                    },
                    timeout=30.0
                )

                if response.status_code == 200:
                    steps.append({
                        "step": table,
                        "status": "success",
                        "message": f"Cleared {table}"
                    })
                else:
                    error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                    steps.append({
                        "step": table,
                        "status": "error",
                        "message": error_msg
                    })
                    errors.append(f"{table}: {error_msg}")

            except Exception as e:
                steps.append({
                    "step": table,
                    "status": "error",
                    "message": str(e)
                })
                errors.append(f"{table}: {str(e)}")

    success = len(errors) == 0
    return {
        "success": success,
        "message": "LangFuse traces cleared successfully" if success else f"Completed with errors: {'; '.join(errors)}",
        "steps": steps,
        "note": "Project, users, and API keys are preserved. Refresh LangFuse UI to see changes."
    }


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


# ===================
# LangChain Agent Tools and Endpoints
# ===================

if LANGCHAIN_ENABLED:
    @tool
    def search_knowledge_graph(query: str) -> str:
        """
        Search the knowledge graph for information about entities, relationships, and facts.
        Use this tool when the user asks questions about people, organizations, concepts,
        or relationships stored in the knowledge base.

        Args:
            query: The search query to find relevant information in the knowledge graph.

        Returns:
            A string containing relevant nodes and their relationships from the knowledge graph.
        """
        import asyncio
        from cognee import SearchType

        try:
            # Schedule the coroutine on the main event loop from this thread
            # This avoids "attached to a different loop" errors with Neo4j
            if _main_event_loop is None:
                return "Error: Main event loop not initialized"

            future = asyncio.run_coroutine_threadsafe(
                _async_search_knowledge(query),
                _main_event_loop
            )
            # Wait for the result with a timeout
            result = future.result(timeout=60)
            return result
        except Exception as e:
            return f"Error searching knowledge graph: {str(e)}"

    async def _async_search_knowledge(query: str) -> str:
        """Async helper for knowledge graph search."""
        from cognee import SearchType

        try:
            # Get context from knowledge graph
            context_results = await cognee.search(
                query_text=query,
                query_type=SearchType.GRAPH_COMPLETION,
                top_k=10,
                only_context=True
            )

            if not context_results:
                return "No relevant information found in the knowledge graph."

            # Parse the context
            graph_context = parse_graph_context(context_results)

            # Format the response
            response_parts = []

            if graph_context["nodes"]:
                response_parts.append("**Relevant Entities:**")
                for node in graph_context["nodes"][:10]:
                    response_parts.append(f"- {node['name']}: {node['content'][:200]}")

            if graph_context["connections"]:
                response_parts.append("\n**Relationships:**")
                for conn in graph_context["connections"][:10]:
                    response_parts.append(f"- {conn['source']} --[{conn['relationship']}]--> {conn['target']}")

            if not response_parts:
                return "No structured information found. The knowledge graph may be empty."

            return "\n".join(response_parts)

        except Exception as e:
            return f"Error during search: {str(e)}"

    @tool
    def get_answer_from_knowledge(question: str) -> str:
        """
        Get a direct answer to a question using the knowledge graph and LLM.
        Use this tool when you need a comprehensive answer that synthesizes
        information from multiple sources in the knowledge base.

        Args:
            question: The question to answer using the knowledge graph.

        Returns:
            A synthesized answer based on the knowledge graph content.
        """
        import asyncio

        try:
            # Schedule the coroutine on the main event loop from this thread
            # This avoids "attached to a different loop" errors with Neo4j
            if _main_event_loop is None:
                return "Error: Main event loop not initialized"

            future = asyncio.run_coroutine_threadsafe(
                _async_get_answer(question),
                _main_event_loop
            )
            # Wait for the result with a timeout
            result = future.result(timeout=60)
            return result
        except Exception as e:
            return f"Error getting answer: {str(e)}"

    async def _async_get_answer(question: str) -> str:
        """Async helper for getting answers."""
        from cognee import SearchType

        try:
            results = await cognee.search(
                query_text=question,
                query_type=SearchType.GRAPH_COMPLETION,
                top_k=10
            )

            if results:
                if isinstance(results[0], str):
                    return results[0]
                elif hasattr(results[0], 'dict'):
                    r = results[0].dict()
                    return r.get('text') or r.get('content') or str(r)
                else:
                    return str(results[0])
            else:
                return "I couldn't find relevant information in the knowledge base to answer this question."

        except Exception as e:
            return f"Error generating answer: {str(e)}"

    @tool
    def list_available_documents() -> str:
        """
        List all documents that have been uploaded to the knowledge base.
        Use this tool when the user wants to know what information sources are available.

        Returns:
            A list of uploaded document names and their sizes.
        """
        try:
            documents = []
            for file in UPLOAD_DIR.iterdir():
                if file.is_file() and not file.name.startswith('.'):
                    stat = file.stat()
                    size_kb = stat.st_size / 1024
                    documents.append(f"- {file.name} ({size_kb:.1f} KB)")

            if documents:
                return "**Available Documents:**\n" + "\n".join(documents)
            else:
                return "No documents have been uploaded to the knowledge base yet."
        except Exception as e:
            return f"Error listing documents: {str(e)}"

    def create_knowledge_agent():
        """Create a LangChain agent for knowledge graph queries."""
        if not LANGCHAIN_ENABLED:
            return None

        try:
            # Initialize the LLM model name
            llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")

            # Define the tools
            tools = [search_knowledge_graph, get_answer_from_knowledge, list_available_documents]

            # Create the agent using the new LangChain 1.x API
            agent = create_agent(
                model=llm_model,
                tools=tools,
                system_prompt="""You are a helpful knowledge assistant with access to a knowledge graph database.
Your role is to help users find information and answer questions based on the stored knowledge.

When answering questions:
1. First use the search_knowledge_graph tool to find relevant entities and relationships
2. If you need a synthesized answer, use get_answer_from_knowledge tool
3. Always cite the sources (entities/relationships) you found
4. If no relevant information is found, clearly state that

Be concise but thorough in your responses."""
            )

            return agent

        except Exception as e:
            print(f"⚠ Error creating knowledge agent: {e}")
            import traceback
            traceback.print_exc()
            return None

    # Create the agent on startup
    knowledge_agent = create_knowledge_agent()
    if knowledge_agent:
        print("✓ LangChain knowledge agent initialized")

    @app.post("/agent/query")
    @observe(name="agent_query", as_type="agent")
    async def agent_query(request: AgentQueryRequest):
        """
        Query the knowledge base using the LangChain agent.
        The agent can search, reason, and synthesize information from the knowledge graph.
        """
        if not LANGCHAIN_ENABLED or not knowledge_agent:
            raise HTTPException(
                status_code=503,
                detail="LangChain agent is not available. Check if LangChain is installed."
            )

        try:
            # Create a new callback handler linked to the current trace context
            callbacks = []
            if OBSERVABILITY_ENABLED and LangfuseCallbackHandler:
                try:
                    # Get trace context from langfuse SDK v3 client
                    trace_id = langfuse_client.get_current_trace_id()
                    observation_id = langfuse_client.get_current_observation_id()

                    if trace_id:
                        # Create trace context to link LangChain operations
                        trace_context = {"trace_id": trace_id}
                        if observation_id:
                            trace_context["parent_span_id"] = observation_id

                        # Create callback handler with trace context
                        handler = LangfuseCallbackHandler(
                            trace_context=trace_context,
                            update_trace=True
                        )
                        callbacks.append(handler)
                        print(f"✓ LangChain callback linked to trace: {trace_id[:8]}...")
                    else:
                        # No active trace, create standalone callback
                        handler = LangfuseCallbackHandler()
                        callbacks.append(handler)
                        print("⚠ No active trace, using standalone callback")
                except Exception as e:
                    # Fall back to standalone callback handler
                    handler = LangfuseCallbackHandler()
                    callbacks.append(handler)
                    print(f"⚠ Using standalone LangChain callback: {e}")

            # Update trace with input
            langfuse_client.update_current_trace(
                input={"query": request.query},
                metadata={"endpoint": "/agent/query", "use_agent": request.use_agent}
            )

            # Run the agent using the new LangChain 1.x API
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: knowledge_agent.invoke(
                    {"messages": [{"role": "user", "content": request.query}]},
                    config={"callbacks": callbacks} if callbacks else {}
                )
            )

            # Extract the response from messages
            messages = result.get("messages", [])
            answer = "No response generated"
            steps = []

            for msg in messages:
                # Get the final assistant message as the answer
                if hasattr(msg, 'content') and hasattr(msg, 'type'):
                    if msg.type == "ai" and msg.content:
                        answer = msg.content
                    elif msg.type == "tool":
                        steps.append({
                            "tool": getattr(msg, 'name', 'unknown'),
                            "input": "",
                            "output": str(msg.content)[:500]
                        })
                elif isinstance(msg, dict):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        answer = msg["content"]

            response = {
                "query": request.query,
                "answer": answer,
                "agent_used": True,
                "steps": steps,
                "tools_called": len(steps)
            }

            # Update trace with output
            langfuse_client.update_current_trace(
                output={
                    "answer": answer,
                    "tools_called": len(steps)
                }
            )

            langfuse_client.flush()
            return response

        except Exception as e:
            import traceback
            langfuse_client.update_current_trace(
                output={"error": str(e)},
                metadata={"traceback": traceback.format_exc()[:1000]}
            )
            langfuse_client.flush()
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/agent/status")
    async def agent_status():
        """Check if the LangChain agent is available and its configuration."""
        return {
            "langchain_enabled": LANGCHAIN_ENABLED,
            "agent_available": knowledge_agent is not None,
            "langfuse_callback": langfuse_callback_handler is not None,
            "llm_model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "tools": ["search_knowledge_graph", "get_answer_from_knowledge", "list_available_documents"]
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
