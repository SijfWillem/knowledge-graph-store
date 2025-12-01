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
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
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


def get_graph_data():
    """Get knowledge graph data."""
    try:
        response = requests.get(f"{BACKEND_URL}/graph", timeout=30)
        return response.json()
    except Exception as e:
        return {"nodes": [], "edges": [], "error": str(e)}


def reset_knowledge_base():
    """Reset the knowledge base."""
    try:
        response = requests.delete(f"{BACKEND_URL}/reset", timeout=30)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


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

    # Reset button
    if st.button("🗑️ Reset Knowledge Base", type="secondary"):
        if st.checkbox("Confirm reset"):
            result = reset_knowledge_base()
            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.success("Knowledge base reset!")
                st.session_state.messages = []
                st.session_state.graph_data = None
                st.rerun()


# Main content area
tab1, tab2 = st.tabs(["💬 Chat", "🕸️ Knowledge Graph"])

# Chat tab
with tab1:
    st.header("Chat with your Knowledge Base")

    # Search type selector
    col1, col2 = st.columns([3, 1])
    with col2:
        search_type = st.selectbox(
            "Search Type",
            ["GRAPH_COMPLETION", "RAG_COMPLETION", "CHUNKS", "SUMMARIES"],
            help="GRAPH_COMPLETION: Q&A using knowledge graph\nRAG_COMPLETION: Standard RAG\nCHUNKS: Raw text chunks\nSUMMARIES: Document summaries"
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Querying knowledge graph..."):
                result = query_knowledge(prompt, search_type)

                if "error" in result:
                    response = f"Error: {result['error']}"
                elif "answer" in result:
                    # Display the GraphRAG answer
                    response = result["answer"]

                    # Show graph relationships as sources
                    sources = result.get("sources", [])
                    if sources:
                        response += "\n\n---\n**Knowledge Graph Sources:**\n"
                        for src in sources[:5]:
                            response += f"- `{src}`\n"
                else:
                    response = "No answer could be generated. Try adding more documents."

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

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

            # Color mapping for node types
            type_colors = {
                "Entity": "#4CAF50",
                "Document": "#2196F3",
                "Chunk": "#FF9800",
                "Concept": "#9C27B0",
                "Person": "#E91E63",
                "Organization": "#00BCD4",
                "Location": "#795548",
                "default": "#607D8B"
            }

            for node in nodes_data:
                node_type = node.get("type", "Entity")
                color = type_colors.get(node_type, type_colors["default"])
                nodes.append(Node(
                    id=node["id"],
                    label=node.get("label", node["id"])[:30],
                    size=25,
                    color=color,
                    title=f"Type: {node_type}\n{str(node.get('data', ''))[:200]}"
                ))

            for edge in edges_data:
                edges.append(Edge(
                    source=edge["source"],
                    target=edge["target"],
                    label=edge.get("label", "")[:20],
                    color="#888888"
                ))

            # Graph configuration
            config = Config(
                width=1200,
                height=600,
                directed=True,
                physics=True,
                hierarchical=False,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=False,
                node={"labelProperty": "label"},
                link={"labelProperty": "label", "renderLabel": True}
            )

            # Render graph
            return_value = agraph(nodes=nodes, edges=edges, config=config)

            # Legend
            st.markdown("### Legend")
            legend_cols = st.columns(len(type_colors))
            for i, (node_type, color) in enumerate(type_colors.items()):
                with legend_cols[i % len(legend_cols)]:
                    st.markdown(
                        f'<span style="color:{color}">●</span> {node_type}',
                        unsafe_allow_html=True
                    )

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
