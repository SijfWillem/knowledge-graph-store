"""
Cognee RAG Frontend
Streamlit UI for document upload, chat, and knowledge graph visualization.
"""
import os
import time
import requests
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="Cognee Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 100%;
    }
    .status-processing {
        color: #ff9800;
    }
    .status-completed {
        color: #4caf50;
    }
    .status-error {
        color: #f44336;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if backend is healthy."""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_processing_status():
    """Get current processing status."""
    try:
        response = requests.get(f"{BACKEND_URL}/status", timeout=5)
        return response.json()
    except:
        return {"status": "unknown", "message": "Cannot reach backend"}


def upload_documents(files, dataset_name):
    """Upload multiple documents to the backend."""
    try:
        files_data = [("files", (f.name, f.getvalue(), f.type)) for f in files]
        response = requests.post(
            f"{BACKEND_URL}/upload",
            files=files_data,
            params={"dataset_name": dataset_name},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_documents():
    """Get list of uploaded documents."""
    try:
        response = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        return response.json()
    except Exception as e:
        return {"documents": [], "error": str(e)}


def add_text(text, dataset_name):
    """Add text to the knowledge base."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/add-text",
            json={"text": text, "dataset_name": dataset_name},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def query_knowledge(query, search_type):
    """Query the knowledge base."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/query",
            json={"query": query, "search_type": search_type},
            timeout=60
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def query_agent(query):
    """Query the knowledge base using the LangChain agent."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/agent/query",
            json={"query": query, "use_agent": True},
            timeout=120
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def get_agent_status():
    """Check if the LangChain agent is available."""
    try:
        response = requests.get(f"{BACKEND_URL}/agent/status", timeout=5)
        return response.json()
    except:
        return {"langchain_enabled": False, "agent_available": False}


def get_graph_data():
    """Get knowledge graph data."""
    try:
        response = requests.get(f"{BACKEND_URL}/graph", timeout=30)
        return response.json()
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


def reset_knowledge_base():
    """Reset the knowledge base with comprehensive cleanup.

    Returns a response with step-by-step progress:
    - Neo4j graph clearing
    - Uploaded files clearing
    - Cognee data clearing
    """
    try:
        response = requests.delete(f"{BACKEND_URL}/reset", timeout=60)
        return response.json()
    except Exception as e:
        return {"error": str(e), "success": False, "steps": []}


def reset_langfuse_traces():
    """Reset LangFuse traces while preserving project and API keys.

    Returns a response with step-by-step progress for each ClickHouse table.
    """
    try:
        response = requests.delete(f"{BACKEND_URL}/reset-langfuse-traces", timeout=120)
        return response.json()
    except Exception as e:
        return {"error": str(e), "success": False, "steps": []}


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_data" not in st.session_state:
    st.session_state.graph_data = None


# Sidebar
with st.sidebar:
    st.title("🧠 Cognee RAG")
    st.markdown("---")

    # Backend status
    backend_healthy = check_backend_health()
    if backend_healthy:
        st.success("Backend: Connected")
    else:
        st.error("Backend: Disconnected")

    st.markdown("---")

    # Dataset selection
    dataset_name = st.text_input("Dataset Name", value="default")

    st.markdown("---")

    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=["txt", "pdf", "md", "json", "csv", "docx"],
        help="Upload one or more documents to build the knowledge graph",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Upload & Process"):
        with st.spinner(f"Uploading {len(uploaded_files)} file(s)..."):
            result = upload_documents(uploaded_files, dataset_name)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success(result.get("message", "Uploaded!"))

    # Show loaded documents
    st.markdown("---")
    st.subheader("📚 Loaded Documents")
    docs = get_documents()
    if docs.get("documents"):
        for doc in docs["documents"]:
            size_kb = doc["size"] / 1024
            st.text(f"• {doc['name']} ({size_kb:.1f} KB)")
    else:
        st.caption("No documents loaded yet")

    st.markdown("---")

    # Text input
    st.subheader("📝 Add Text")
    text_input = st.text_area(
        "Enter text to add to knowledge base",
        height=150,
        placeholder="Paste or type text here..."
    )

    if text_input and st.button("Add Text"):
        with st.spinner("Adding text..."):
            result = add_text(text_input, dataset_name)
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Text added for processing!")

    st.markdown("---")

    # Processing status
    st.subheader("⏳ Status")
    status = get_processing_status()
    status_text = status.get("status", "unknown")
    status_message = status.get("message", "")

    if status_text == "processing":
        st.warning(f"🔄 {status_message}")
    elif status_text == "completed":
        st.success(f"✅ {status_message}")
    elif status_text == "error":
        st.error(f"❌ {status_message}")
    else:
        st.info(f"💤 {status_text}")

    if st.button("🔄 Refresh Status"):
        st.rerun()

    st.markdown("---")

    # Reset button with progress display
    st.subheader("🗑️ Reset Knowledge Base")
    st.caption("Clears Neo4j graph, uploaded files, and Cognee data")

    if "reset_confirmed" not in st.session_state:
        st.session_state.reset_confirmed = False

    if st.button("Reset Knowledge Base", type="secondary"):
        st.session_state.reset_confirmed = True

    if st.session_state.reset_confirmed:
        confirm = st.checkbox("I understand this will delete all data", value=False)

        if confirm:
            if st.button("Confirm Reset", type="primary"):
                # Create progress display
                progress_container = st.container()

                with progress_container:
                    st.markdown("**Resetting Knowledge Base...**")

                    # Show initial progress
                    step_status = st.empty()
                    step_status.info("⏳ Starting reset...")

                    # Progress bar
                    progress_bar = st.progress(0)

                    # Step indicators
                    neo4j_status = st.empty()
                    uploads_status = st.empty()
                    cognee_status = st.empty()

                    neo4j_status.markdown("⏳ Neo4j graph...")
                    uploads_status.markdown("⏳ Uploaded files...")
                    cognee_status.markdown("⏳ Cognee data...")

                    progress_bar.progress(10)

                    # Call reset endpoint
                    result = reset_knowledge_base()

                    if "error" in result and not result.get("steps"):
                        step_status.error(f"❌ Error: {result['error']}")
                    else:
                        # Update progress based on steps
                        steps = result.get("steps", [])
                        progress_bar.progress(30)

                        for step in steps:
                            step_name = step.get("step", "")
                            status = step.get("status", "")
                            message = step.get("message", "")

                            if step_name == "neo4j":
                                progress_bar.progress(50)
                                if status == "success":
                                    neo4j_status.markdown(f"✅ {message}")
                                else:
                                    neo4j_status.markdown(f"❌ {message}")

                            elif step_name == "uploads":
                                progress_bar.progress(70)
                                if status == "success":
                                    uploads_status.markdown(f"✅ {message}")
                                else:
                                    uploads_status.markdown(f"❌ {message}")

                            elif step_name == "cognee":
                                progress_bar.progress(90)
                                if status == "success":
                                    cognee_status.markdown(f"✅ {message}")
                                else:
                                    cognee_status.markdown(f"❌ {message}")

                        progress_bar.progress(100)

                        # Final status
                        if result.get("success", False):
                            step_status.success("✅ Knowledge base reset successfully!")
                            # Clear session state
                            st.session_state.messages = []
                            st.session_state.graph_data = None
                            st.session_state.reset_confirmed = False
                        else:
                            step_status.warning(f"⚠️ {result.get('message', 'Reset completed with some issues')}")

                        # Add rerun button
                        if st.button("Done"):
                            st.session_state.reset_confirmed = False
                            st.rerun()
        else:
            if st.button("Cancel"):
                st.session_state.reset_confirmed = False
                st.rerun()

    st.markdown("---")

    # LangFuse Traces Reset button with progress display
    st.subheader("📊 Reset LangFuse Traces")
    st.caption("Clears all traces, keeps project & API keys")

    if "langfuse_reset_confirmed" not in st.session_state:
        st.session_state.langfuse_reset_confirmed = False

    if st.button("Reset LangFuse Traces", type="secondary"):
        st.session_state.langfuse_reset_confirmed = True

    if st.session_state.langfuse_reset_confirmed:
        lf_confirm = st.checkbox("I understand this will delete all traces", value=False, key="lf_confirm")

        if lf_confirm:
            if st.button("Confirm LangFuse Reset", type="primary", key="lf_reset_btn"):
                # Create progress display
                lf_progress_container = st.container()

                with lf_progress_container:
                    st.markdown("**Clearing LangFuse Traces...**")

                    # Show initial progress
                    lf_step_status = st.empty()
                    lf_step_status.info("⏳ Connecting to ClickHouse...")

                    # Progress bar
                    lf_progress_bar = st.progress(0)

                    # Step indicators for each table
                    table_statuses = {}
                    tables = ["traces", "observations", "scores", "event_log"]

                    for table in tables:
                        table_statuses[table] = st.empty()
                        table_statuses[table].markdown(f"⏳ {table}...")

                    lf_progress_bar.progress(10)

                    # Call reset endpoint
                    result = reset_langfuse_traces()

                    if "error" in result and not result.get("steps"):
                        lf_step_status.error(f"❌ Error: {result['error']}")
                    else:
                        # Update progress based on steps
                        steps = result.get("steps", [])
                        total_steps = len(tables)

                        for i, step in enumerate(steps):
                            step_name = step.get("step", "")
                            status = step.get("status", "")
                            message = step.get("message", "")

                            # Update progress bar
                            progress = int(10 + (90 * (i + 1) / total_steps))
                            lf_progress_bar.progress(progress)

                            # Update step status
                            if step_name in table_statuses:
                                if status == "success":
                                    table_statuses[step_name].markdown(f"✅ {message}")
                                else:
                                    table_statuses[step_name].markdown(f"❌ {message}")

                        lf_progress_bar.progress(100)

                        # Final status
                        if result.get("success", False):
                            lf_step_status.success("✅ LangFuse traces cleared successfully!")
                            st.info(result.get("note", "Refresh LangFuse UI to see changes."))
                            st.session_state.langfuse_reset_confirmed = False
                        else:
                            lf_step_status.warning(f"⚠️ {result.get('message', 'Completed with some issues')}")

                        # Add rerun button
                        if st.button("Done", key="lf_done_btn"):
                            st.session_state.langfuse_reset_confirmed = False
                            st.rerun()
        else:
            if st.button("Cancel", key="lf_cancel_btn"):
                st.session_state.langfuse_reset_confirmed = False
                st.rerun()


# Main content area
tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Knowledge Graph"])

# Chat tab
with tab1:
    # Header and options at top
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.markdown("### Chat with your Knowledge Base")
    with col2:
        # Check agent status
        agent_status = get_agent_status()
        use_agent = st.toggle(
            "Use Agent",
            value=False,
            help="Use LangChain agent for multi-step reasoning",
            disabled=not agent_status.get("agent_available", False)
        )
        if not agent_status.get("agent_available", False):
            st.caption("Agent unavailable")
    with col3:
        search_type = st.selectbox(
            "Search Type",
            ["GRAPH_COMPLETION", "CHUNKS", "CHUNKS_LEXICAL", "SUMMARIES", "CODE", "GRAPH_COMPLETION_COT"],
            help="GRAPH_COMPLETION: Q&A using knowledge graph (default)\nCHUNKS: Raw text chunks\nCHUNKS_LEXICAL: Token-based lexical search\nSUMMARIES: Document summaries\nCODE: Code-specific search\nGRAPH_COMPLETION_COT: Chain-of-thought reasoning",
            disabled=use_agent
        )

    # Chat messages in a scrollable container
    chat_container = st.container(height=450)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Show graph context for assistant messages if stored
                if message["role"] == "assistant":
                    graph_ctx = message.get("graph_context", {})
                    nodes = graph_ctx.get("nodes", [])
                    connections = graph_ctx.get("connections", [])

                    if nodes or connections:
                        with st.expander("🔗 Knowledge Graph Context", expanded=False):
                            if nodes:
                                st.markdown("**Retrieved Nodes:**")
                                for node in nodes:
                                    node_name = node.get("name", "Unknown")
                                    node_type = node.get("type", "Entity")
                                    node_content = node.get("content", "")
                                    st.markdown(f"- **{node_name}** ({node_type})")
                                    if node_content:
                                        st.markdown(f"  > {node_content[:200]}{'...' if len(node_content) > 200 else ''}")

                            if connections:
                                st.markdown("**Relationships:**")
                                for conn in connections:
                                    src = conn.get("source", "?")
                                    rel = conn.get("relationship", "related_to")
                                    tgt = conn.get("target", "?")
                                    st.markdown(f"- `{src}` → **{rel}** → `{tgt}`")

                    # Show agent steps if available
                    if "agent_steps" in message and message["agent_steps"]:
                        with st.expander("🤖 Agent Reasoning Steps", expanded=False):
                            for i, step in enumerate(message["agent_steps"], 1):
                                st.markdown(f"**Step {i}: {step.get('tool', 'Unknown tool')}**")
                                if step.get("input"):
                                    st.markdown(f"*Input:* `{step['input']}`")
                                if step.get("output"):
                                    st.markdown(f"*Output:* {step['output'][:300]}{'...' if len(step.get('output', '')) > 300 else ''}")
                                st.markdown("---")

                    # Fallback to legacy retrieved_docs if no graph context
                    elif "retrieved_docs" in message:
                        docs = message["retrieved_docs"]
                        if docs:
                            with st.expander("📄 Retrieved Documents", expanded=False):
                                for i, doc in enumerate(docs, 1):
                                    st.markdown(f"**Source {i}**")
                                    st.markdown(f"> {doc.get('text', 'No text available')}")

    # Chat input (stays below the scrollable container)
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            if use_agent:
                # Use LangChain agent
                with st.spinner("Agent is reasoning..."):
                    result = query_agent(prompt)

                    if "error" in result:
                        response = f"Error: {result['error']}"
                        graph_context = {}
                        agent_steps = []
                        retrieved_docs = []
                    elif "answer" in result:
                        response = result["answer"]
                        graph_context = {}
                        agent_steps = result.get("steps", [])
                        retrieved_docs = []
                    else:
                        response = "No answer could be generated. Try adding more documents."
                        graph_context = {}
                        agent_steps = []
                        retrieved_docs = []

                    st.markdown(response)

                    # Show agent steps
                    if agent_steps:
                        with st.expander("🤖 Agent Reasoning Steps", expanded=True):
                            for i, step in enumerate(agent_steps, 1):
                                st.markdown(f"**Step {i}: {step.get('tool', 'Unknown tool')}**")
                                if step.get("input"):
                                    st.markdown(f"*Input:* `{step['input']}`")
                                if step.get("output"):
                                    st.markdown(f"*Output:* {step['output'][:300]}{'...' if len(step.get('output', '')) > 300 else ''}")
                                st.markdown("---")

                    # Store message with agent steps
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "agent_steps": agent_steps,
                        "graph_context": {},
                        "retrieved_docs": []
                    })
            else:
                # Use direct query
                with st.spinner("Querying knowledge graph..."):
                    result = query_knowledge(prompt, search_type)

                    if "error" in result:
                        response = f"Error: {result['error']}"
                        graph_context = {}
                        retrieved_docs = []
                    elif "answer" in result:
                        response = result["answer"]
                        graph_context = result.get("graph_context", {})
                        retrieved_docs = result.get("retrieved_documents", [])
                    else:
                        response = "No answer could be generated. Try adding more documents."
                        graph_context = {}
                        retrieved_docs = []

                    st.markdown(response)

                    # Show graph context (nodes and connections)
                    nodes = graph_context.get("nodes", [])
                    connections = graph_context.get("connections", [])

                    if nodes or connections:
                        with st.expander("🔗 Knowledge Graph Context", expanded=True):
                            if nodes:
                                st.markdown("**Retrieved Nodes:**")
                                for node in nodes:
                                    node_name = node.get("name", "Unknown")
                                    node_type = node.get("type", "Entity")
                                    node_content = node.get("content", "")
                                    st.markdown(f"- **{node_name}** ({node_type})")
                                    if node_content:
                                        st.markdown(f"  > {node_content[:200]}{'...' if len(node_content) > 200 else ''}")

                            if connections:
                                st.markdown("**Relationships:**")
                                for conn in connections:
                                    src = conn.get("source", "?")
                                    rel = conn.get("relationship", "related_to")
                                    tgt = conn.get("target", "?")
                                    st.markdown(f"- `{src}` → **{rel}** → `{tgt}`")

                    # Fallback to legacy retrieved_docs if no graph context
                    elif retrieved_docs:
                        with st.expander("📄 Retrieved Documents", expanded=False):
                            for i, doc in enumerate(retrieved_docs, 1):
                                st.markdown(f"**Source {i}**")
                                st.markdown(f"> {doc.get('text', 'No text available')}")

                    # Store message with graph context
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "graph_context": graph_context,
                        "retrieved_docs": retrieved_docs
                    })

# Graph visualization tab
with tab2:
    st.header("Knowledge Graph Visualization")

    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🔄 Refresh Graph"):
            with st.spinner("Loading graph..."):
                st.session_state.graph_data = get_graph_data()

    # Auto-load graph if not loaded yet
    if st.session_state.graph_data is None:
        st.session_state.graph_data = get_graph_data()

    # Display graph
    graph_data = st.session_state.graph_data

    if graph_data:
        nodes_data = graph_data.get("nodes", [])
        edges_data = graph_data.get("edges", [])

        if nodes_data:
            st.info(f"Nodes: {len(nodes_data)} | Edges: {len(edges_data)}")

            # Create nodes and edges for agraph
            nodes = []
            edges = []

            # Color mapping for node types (matching backend entity classification)
            # Using brighter colors that work well in both light and dark mode
            type_colors = {
                # Entity types from custom extraction prompt
                "Person": "#FF4081",           # Pink - People (brighter)
                "Organization": "#00E5FF",     # Cyan - Companies (brighter)
                "Location": "#A1887F",         # Brown - Places (lighter)
                "Concept": "#CE93D8",          # Purple - Abstract ideas (lighter)
                "Technology": "#7C4DFF",       # Indigo - Software (brighter)
                "Product": "#FF6E40",          # Deep Orange - Products (brighter)
                "Event": "#FFCA28",            # Amber - Events (brighter)
                "Document": "#448AFF",         # Blue - Documents (brighter)
                "Date": "#1DE9B6",             # Teal - Dates (brighter)
                "Metric": "#B2FF59",           # Light Green - Numbers (brighter)
                # Default and legacy types
                "Entity": "#69F0AE",           # Green - Generic entity (brighter)
                "Chunk": "#FFB74D",            # Orange - Text chunks (brighter)
                "default": "#90A4AE"           # Gray - Unknown (lighter)
            }

            # Node size - slightly larger for better visibility
            NODE_SIZE = 25

            for node in nodes_data:
                node_type = node.get("type", "Entity")
                color = type_colors.get(node_type, type_colors["default"])
                nodes.append(Node(
                    id=node["id"],
                    label=node.get("label", node["id"])[:20],
                    size=NODE_SIZE,
                    color=color,
                    title=f"Type: {node_type}\n{str(node.get('data', ''))[:200]}"
                ))

            for edge in edges_data:
                edges.append(Edge(
                    source=edge["source"],
                    target=edge["target"],
                    label=edge.get("label", "")[:12],
                    color="#999999"
                ))

            # Graph configuration with vis.js options via kwargs
            config = Config(
                width=1200,
                height=700,
                directed=True,
                physics=True,
                hierarchical=False,
                # Physics options for better spacing
                solver="barnesHut",
                minVelocity=0.75,
                stabilization=True,
                fit=True,
                # Node spacing
                nodeSpacing=150,
                # Add edges config for curved edges
                edges={
                    "smooth": {
                        "enabled": True,
                        "type": "curvedCW",
                        "roundness": 0.2
                    },
                    "arrows": {
                        "to": {"enabled": True, "scaleFactor": 0.5}
                    },
                    "color": "#888888",
                    "font": {"size": 10, "align": "middle"}
                }
            )

            # Render graph
            return_value = agraph(nodes=nodes, edges=edges, config=config)

            # Legend - using Streamlit columns for reliable rendering
            st.markdown("#### Legend")
            # Filter out 'default' from legend and only show types that might appear
            legend_types = {k: v for k, v in type_colors.items() if k != "default"}
            legend_items = list(legend_types.items())

            # Display in rows of 6 columns
            cols_per_row = 6
            for row_start in range(0, len(legend_items), cols_per_row):
                row_items = legend_items[row_start:row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for i, (node_type, color) in enumerate(row_items):
                    with cols[i]:
                        st.markdown(f'<span style="color:{color}; font-size:20px;">●</span> {node_type}', unsafe_allow_html=True)

        else:
            st.warning("No graph data available. Upload documents and process them first.")

        if "error" in graph_data:
            st.error(f"Graph Error: {graph_data['error']}")
    else:
        st.info("Click 'Load Graph' to visualize the knowledge graph.")


# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888;'>"
    "Powered by <a href='https://github.com/topoteretes/cognee'>Cognee</a> | "
    "Hybrid RAG with Knowledge Graphs"
    "</div>",
    unsafe_allow_html=True
)
