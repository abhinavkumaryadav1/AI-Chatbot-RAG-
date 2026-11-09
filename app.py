"""
SmartRAG - Document-Based AI Chatbot
A modern, local Retrieval-Augmented Generation (RAG) assistant.
Built with Streamlit, PyMuPDF, Sentence Transformers, FAISS, and Ollama.
"""

import os
import time
from datetime import datetime
from typing import List, Dict, Any
import streamlit as st

# Application modules
import config
from document_processor import DocumentProcessor, DocumentProcessingError
from text_splitter import ChunkSplitter
from embedding_manager import EmbeddingManager
from vector_store import FAISSVectorStore
from ollama_client import OllamaClient, OllamaClientError
import prompt_templates

# Page Configuration
st.set_page_config(
    page_title=f"{config.APP_NAME} - Document AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Modern Dark Theme CSS
st.markdown(config.CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------------
# Cached Resources (Embedding Model & Ollama Client)
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def get_embedding_manager() -> EmbeddingManager:
    """Loads and caches the Sentence Transformers model locally."""
    return EmbeddingManager(config.DEFAULT_EMBEDDING_MODEL)


@st.cache_resource
def get_ollama_client() -> OllamaClient:
    """Initializes the Ollama HTTP API client."""
    return OllamaClient(host=config.DEFAULT_OLLAMA_HOST, default_model=config.DEFAULT_OLLAMA_MODEL)


# -------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = FAISSVectorStore(dimension=config.EMBEDDING_DIMENSION)

if "uploaded_file_records" not in st.session_state:
    # Maps filename -> {'bytes': bytes, 'size_kb': float, 'page_count': int, 'ext': str}
    st.session_state.uploaded_file_records = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = ["Current Session"]

# Pre-populate sample documents on first run if index is empty
if "initialized_samples" not in st.session_state:
    st.session_state.initialized_samples = True
    sample_pdf = os.path.join("sample_documents", "Employee_Handbook.pdf")
    sample_txt = os.path.join("sample_documents", "HR_Guidelines.txt")

    if os.path.exists(sample_pdf):
        try:
            with open(sample_pdf, "rb") as f:
                pdf_b = f.read()
            st.session_state.uploaded_file_records["Employee_Handbook.pdf"] = {
                "bytes": pdf_b,
                "size_kb": round(len(pdf_b) / 1024, 1),
                "page_count": 3,
                "ext": "pdf"
            }
        except Exception:
            pass

    if os.path.exists(sample_txt):
        try:
            with open(sample_txt, "rb") as f:
                txt_b = f.read()
            st.session_state.uploaded_file_records["HR_Guidelines.txt"] = {
                "bytes": txt_b,
                "size_kb": round(len(txt_b) / 1024, 1),
                "page_count": 1,
                "ext": "txt"
            }
        except Exception:
            pass

    # Build initial index from samples
    if st.session_state.uploaded_file_records:
        try:
            emb_mgr = get_embedding_manager()
            splitter = ChunkSplitter(config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP)
            all_chunks = []
            for fname, record in st.session_state.uploaded_file_records.items():
                pages = DocumentProcessor.process_file(record["bytes"], fname)
                chunks = splitter.split_pages(pages)
                all_chunks.extend(chunks)

            if all_chunks:
                embeddings = emb_mgr.encode_texts([c["text"] for c in all_chunks])
                st.session_state.vector_store.add_chunks(all_chunks, embeddings)
        except Exception as e:
            st.error(f"Error loading initial sample index: {e}")


# -------------------------------------------------------------
# Document Indexing Helper Functions
# -------------------------------------------------------------
def rebuild_entire_index(chunk_size: int, chunk_overlap: int):
    """Rebuilds the entire FAISS vector index from all current uploaded records."""
    emb_mgr = get_embedding_manager()
    st.session_state.vector_store.clear()

    if not st.session_state.uploaded_file_records:
        return

    splitter = ChunkSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for fname, record in st.session_state.uploaded_file_records.items():
        try:
            pages = DocumentProcessor.process_file(record["bytes"], fname)
            record["page_count"] = len(pages)
            chunks = splitter.split_pages(pages)
            all_chunks.extend(chunks)
        except Exception as err:
            st.warning(f"Could not process '{fname}': {err}")

    if all_chunks:
        embeddings = emb_mgr.encode_texts([c["text"] for c in all_chunks])
        st.session_state.vector_store.add_chunks(all_chunks, embeddings)


def remove_document(filename: str, chunk_size: int, chunk_overlap: int):
    """Removes a document from storage and triggers index rebuilding."""
    if filename in st.session_state.uploaded_file_records:
        del st.session_state.uploaded_file_records[filename]
        rebuild_entire_index(chunk_size, chunk_overlap)
        st.toast(f"Removed '{filename}' and rebuilt vector index.", icon="🗑️")


# -------------------------------------------------------------
# Sidebar Layout
# -------------------------------------------------------------
with st.sidebar:
    # Branding
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">📚</div>
            <div>
                <div class="brand-title">SmartRAG</div>
                <div class="brand-subtitle">Document-Based AI Chatbot</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # New Chat Button
    if st.button("➕  New Chat", use_container_width=True, type="primary"):
        st.session_state.chat_history = []
        st.session_state.pending_query = None
        st.rerun()

    # Chat Sessions List (ChatGPT/Claude style sidebar)
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#64748b; margin-top:14px; margin-bottom:4px;'>CHAT SESSIONS</p>", unsafe_allow_html=True)
    st.caption("💬 Current Session (Active)")

    st.markdown("---")

    # Document Upload Section
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#64748b; margin-bottom:6px;'>UPLOADED DOCUMENTS</p>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload multiple documents to build your local RAG knowledge base.",
        label_visibility="collapsed"
    )

    # Process newly uploaded files
    if uploaded_files:
        files_added = False
        for f in uploaded_files:
            if f.name not in st.session_state.uploaded_file_records:
                file_bytes = f.read()
                ext = "pdf" if f.name.lower().endswith(".pdf") else "txt"
                st.session_state.uploaded_file_records[f.name] = {
                    "bytes": file_bytes,
                    "size_kb": round(len(file_bytes) / 1024, 1),
                    "page_count": 1,
                    "ext": ext
                }
                files_added = True

        if files_added:
            with st.spinner("Indexing new documents into FAISS..."):
                rebuild_entire_index(config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP)
            st.toast("Documents indexed successfully!", icon="✅")
            st.rerun()

    # Uploaded Documents List
    if st.session_state.uploaded_file_records:
        for doc_name, info in list(st.session_state.uploaded_file_records.items()):
            col_info, col_del = st.columns([5, 1])
            with col_info:
                badge_class = "pdf" if info["ext"] == "pdf" else "txt"
                pages_str = f" • {info['page_count']} pgs" if info['ext'] == "pdf" else ""
                st.markdown(
                    f"""
                    <div class="doc-card">
                        <div>
                            <span class="doc-badge {badge_class}">{info['ext']}</span>
                            <span style="font-size:0.82rem; font-weight:600; margin-left:6px;">{doc_name}</span>
                            <div style="font-size:0.7rem; color:#94a3b8; margin-top:2px;">{info['size_kb']} KB{pages_str}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_del:
                if st.button("🗑️", key=f"del_{doc_name}", help=f"Remove {doc_name}"):
                    remove_document(doc_name, config.DEFAULT_CHUNK_SIZE, config.DEFAULT_CHUNK_OVERLAP)
                    st.rerun()

        # Index Summary Status
        total_chunks = st.session_state.vector_store.total_chunks
        st.markdown(
            f"<div style='font-size:0.72rem; color:#10b981; background:rgba(16, 185, 129, 0.1); padding:6px 10px; border-radius:6px; border:1px solid rgba(16, 185, 129, 0.2); margin-top:6px;'>"
            f"⚡ <b>Index Active:</b> {total_chunks} chunks indexed across {len(st.session_state.uploaded_file_records)} doc(s)"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("No documents uploaded yet. Upload a PDF or TXT to start querying.")

    st.markdown("---")

    # Settings Accordion
    with st.expander("⚙️ Settings & Hyperparameters", expanded=False):
        # Ollama Model Selection
        ollama_cli = get_ollama_client()
        available_models = ollama_cli.list_models()
        model_options = list(set([config.DEFAULT_OLLAMA_MODEL] + config.SUPPORTED_MODELS + available_models))

        selected_model = st.selectbox(
            "Ollama LLM Model",
            options=model_options,
            index=0,
            help="Choose the local Ollama model to use for generating answers."
        )

        chunk_size = st.slider(
            "Chunk Size (characters)",
            min_value=200,
            max_value=2000,
            value=config.DEFAULT_CHUNK_SIZE,
            step=50,
            help="Character length for each segmented text passage."
        )

        chunk_overlap = st.slider(
            "Chunk Overlap (characters)",
            min_value=0,
            max_value=400,
            value=config.DEFAULT_CHUNK_OVERLAP,
            step=25,
            help="Overlap characters between successive chunks to preserve context."
        )

        top_k = st.slider(
            "Retrieved Chunks (Top-K)",
            min_value=1,
            max_value=8,
            value=config.DEFAULT_TOP_K,
            step=1,
            help="Maximum number of relevant chunks retrieved per query."
        )

        similarity_threshold = st.slider(
            "Similarity Confidence Threshold",
            min_value=0.0,
            max_value=0.9,
            value=config.DEFAULT_SIMILARITY_THRESHOLD,
            step=0.05,
            help="Minimum cosine similarity score required for context chunks."
        )

        if st.button("Re-chunk & Re-index", use_container_width=True):
            with st.spinner("Re-chunking and indexing..."):
                rebuild_entire_index(chunk_size, chunk_overlap)
            st.success("Re-indexed successfully!")
            st.rerun()

    # Ollama Connection Status Widget
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:#64748b; margin-top:14px; margin-bottom:6px;'>LOCAL LLM STATUS</p>", unsafe_allow_html=True)
    ollama_cli = get_ollama_client()
    is_connected, status_msg = ollama_cli.check_connection()

    if is_connected:
        st.markdown(
            f"""
            <div class="status-pill online">
                <div class="status-dot online"></div>
                <span>Ollama Online ({selected_model})</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="status-pill offline">
                <div class="status-dot offline"></div>
                <span>Ollama Disconnected</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(f"⚠️ {status_msg}")
        st.info("💡 Run `ollama serve` or open Ollama from your Start menu.")

    if st.button("🔄 Refresh Ollama Status", use_container_width=True):
        st.rerun()

    # Clear Chat Button
    if st.session_state.chat_history:
        if st.button("🗑️  Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.pending_query = None
            st.rerun()

    # Mascot Branding Footer
    st.markdown(
        """
        <div class="assistant-mascot-card">
            <div style="font-size: 2rem; margin-bottom:4px;">🤖</div>
            <div class="title">Your Document Knowledge Assistant</div>
            <div class="subtitle">100% Local • Zero Cloud APIs • FAISS + Ollama</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------------------
# Main Chat Area Header
# -------------------------------------------------------------
status_badge = (
    '<span class="status-pill online"><span class="status-dot online"></span> Local LLM (Ollama)</span>'
    if is_connected else
    '<span class="status-pill offline"><span class="status-dot offline"></span> Local LLM Offline</span>'
)

st.markdown(
    f"""
    <div class="chat-header">
        <div>
            <div style="font-size: 1.25rem; font-weight: 700; color: #ffffff;">{config.APP_NAME}</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">{config.APP_SUBTITLE} — Ask questions and get document-grounded answers</div>
        </div>
        <div>
            {status_badge}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------------
# Welcome Screen (When Chat History is Empty)
# -------------------------------------------------------------
if not st.session_state.chat_history:
    st.markdown(
        """
        <div class="welcome-hero">
            <div style="font-size: 3.5rem; margin-bottom: 12px;">📑 ➔ 🤖</div>
            <h1>Document Intelligence, Locally Grounded</h1>
            <p>Upload PDFs or text files to chat with your knowledge base. SmartRAG retrieves verified source passages using dense semantic embeddings and grounds every response with exact document citations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #a78bfa; margin-bottom: 8px;'>Try asking about the loaded documents:</p>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    sample_questions = [
        "What is the annual and sick leave entitlement for employees?",
        "Can I work from home permanently?",
        "How many days in advance must leave requests be submitted?",
        "What is the health insurance coverage and wellness stipend?",
    ]

    for idx, sq in enumerate(sample_questions):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if st.button(f"🔍 {sq}", key=f"sq_{idx}", use_container_width=True):
                st.session_state.pending_query = sq
                st.rerun()


# -------------------------------------------------------------
# Render Chat Messages
# -------------------------------------------------------------
for msg in st.session_state.chat_history:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    timestamp = msg.get("timestamp", "")
    sources = msg.get("sources", [])

    if role == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(f"**You** <span style='font-size:0.75rem; color:#94a3b8; margin-left:8px;'>{timestamp}</span>", unsafe_allow_html=True)
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**SmartRAG** <span style='font-size:0.75rem; color:#94a3b8; margin-left:8px;'>{timestamp}</span>", unsafe_allow_html=True)
            st.markdown(content)

            # Display Source Citations
            if sources:
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                # Group distinct sources for badge display
                source_tags = []
                for s in sources:
                    doc = s.get("source", "Document")
                    page = s.get("page", "")
                    tag = f"{doc} (Page {page})" if page else doc
                    if tag not in source_tags:
                        source_tags.append(tag)

                tag_html = " ".join([f"<span class='citation-tag'>📄 {t}</span>" for t in source_tags])
                st.markdown(f"<div style='margin-bottom:6px;'><b>Sources:</b> {tag_html}</div>", unsafe_allow_html=True)

                with st.expander("🔍 View Retrieved Context Passages", expanded=False):
                    for idx, chunk in enumerate(sources, 1):
                        score = chunk.get("score", 0.0)
                        source_file = chunk.get("source", "Unknown")
                        page_num = chunk.get("page", "N/A")
                        text_passage = chunk.get("text", "")

                        st.markdown(
                            f"""
                            <div class="citation-container">
                                <div class="citation-header">
                                    <span>#{idx} • {source_file} (Page {page_num})</span>
                                    <span style="margin-left:auto; font-size:0.75rem; color:#34d399;">Similarity: {score:.3f}</span>
                                </div>
                                <div style="font-style: italic; margin-top:4px;">"{text_passage}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )


# -------------------------------------------------------------
# Chat Input & RAG Execution Pipeline
# -------------------------------------------------------------
user_query = st.chat_input("Type your question here...")

# Handle sample button clicks
if st.session_state.pending_query:
    user_query = st.session_state.pending_query
    st.session_state.pending_query = None

if user_query:
    current_time = datetime.now().strftime("%I:%M %p")

    # 1. Append user message to state
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_query,
        "timestamp": current_time,
        "sources": []
    })

    # Render user bubble immediately
    with st.chat_message("user", avatar="👤"):
        st.markdown(f"**You** <span style='font-size:0.75rem; color:#94a3b8; margin-left:8px;'>{current_time}</span>", unsafe_allow_html=True)
        st.markdown(user_query)

    # 2. Execute RAG Retrieval & Generation
    with st.chat_message("assistant", avatar="🤖"):
        assistant_time = datetime.now().strftime("%I:%M %p")
        response_placeholder = st.empty()
        status_container = st.status("🧠 Processing query through SmartRAG pipeline...", expanded=True)

        # Step A: Validate documents exist in store
        vector_store: FAISSVectorStore = st.session_state.vector_store
        if vector_store.total_chunks == 0:
            status_container.update(label="❌ No documents indexed", state="error", expanded=False)
            fallback_ans = "Please upload at least one PDF or TXT document first to ask questions."
            response_placeholder.markdown(fallback_ans)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": fallback_ans,
                "timestamp": assistant_time,
                "sources": []
            })
        else:
            # Step B: Semantic Retrieval using Sentence Transformers & FAISS
            status_container.write("🔍 Embedding query and searching FAISS vector space...")
            emb_mgr = get_embedding_manager()
            query_embedding = emb_mgr.encode_query(user_query)

            retrieved_chunks = vector_store.similarity_search(
                query_embedding=query_embedding,
                top_k=top_k,
                threshold=similarity_threshold
            )

            status_container.write(f"📊 Retrieved {len(retrieved_chunks)} relevant passage(s) above threshold {similarity_threshold}.")

            # Step C: Confidence check
            if not retrieved_chunks:
                status_container.update(label="⚠️ Insufficient document evidence found", state="complete", expanded=False)
                fallback_ans = prompt_templates.FALLBACK_MESSAGE
                response_placeholder.markdown(fallback_ans)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": fallback_ans,
                    "timestamp": assistant_time,
                    "sources": []
                })
            else:
                # Step D: Construct Strict RAG Prompt
                status_container.write("📝 Synthesizing grounded context for local LLM...")
                rag_prompt = prompt_templates.build_rag_prompt(user_query, retrieved_chunks)

                # Step E: Query Ollama
                status_container.write(f"🤖 Generating document-grounded response via Ollama ({selected_model})...")

                ollama_cli = get_ollama_client()
                is_ollama_ready, err_msg = ollama_cli.check_connection()

                if not is_ollama_ready:
                    status_container.update(label="⚠️ Ollama service unavailable", state="error", expanded=False)
                    error_ans = (
                        f"**Unable to connect to Ollama** ({err_msg}).\n\n"
                        "To use local LLM generation:\n"
                        "1. Ensure Ollama is installed (download from [ollama.com](https://ollama.com)).\n"
                        "2. Start Ollama with `ollama serve` or launch the desktop application.\n"
                        f"3. Pull your desired model: `ollama pull {selected_model}`.\n\n"
                        "**Retrieved Document Evidence Below:**"
                    )
                    response_placeholder.markdown(error_ans)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": error_ans,
                        "timestamp": assistant_time,
                        "sources": retrieved_chunks
                    })
                else:
                    try:
                        # Stream generation for smooth UI
                        accumulated_response = ""
                        for token in ollama_cli.generate_stream(
                            prompt=rag_prompt,
                            system=prompt_templates.STRICT_SYSTEM_PROMPT,
                            model=selected_model,
                            temperature=0.1
                        ):
                            accumulated_response += token
                            response_placeholder.markdown(accumulated_response + "▌")

                        final_response = accumulated_response.strip()
                        response_placeholder.markdown(final_response)
                        status_container.update(label="✅ Answer generated successfully!", state="complete", expanded=False)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": final_response,
                            "timestamp": assistant_time,
                            "sources": retrieved_chunks
                        })

                    except OllamaClientError as oce:
                        status_container.update(label=f"❌ Ollama Error: {oce}", state="error", expanded=False)
                        error_ans = f"**Ollama Generation Error:** {str(oce)}"
                        response_placeholder.markdown(error_ans)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_ans,
                            "timestamp": assistant_time,
                            "sources": retrieved_chunks
                        })
                    except Exception as gen_err:
                        status_container.update(label="❌ Generation failed", state="error", expanded=False)
                        error_ans = f"An unexpected error occurred: {str(gen_err)}"
                        response_placeholder.markdown(error_ans)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": error_ans,
                            "timestamp": assistant_time,
                            "sources": retrieved_chunks
                        })

    st.rerun()

