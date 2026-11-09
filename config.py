"""
SmartRAG Configuration Module
Defines system defaults, UI styling, and operational parameters.
"""

APP_NAME = "SmartRAG"
APP_SUBTITLE = "Document-Based AI Chatbot"
APP_DESCRIPTION = "Ask questions and get accurate, document-grounded answers with citations."

# Default Ollama configurations
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
SUPPORTED_MODELS = [
    "llama3.2:3b",
    "llama3.2:1b",
    "llama3:8b",
    "llama3:latest",
    "mistral:latest",
    "phi3:mini",
    "gemma2:2b",
    "qwen2.5:3b",
]

# Embedding model configurations
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384

# Document chunking defaults
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150

# Retrieval defaults
DEFAULT_TOP_K = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.35

# Supported file extensions
SUPPORTED_FILE_TYPES = [".pdf", ".txt"]

# Modern SaaS Dark Theme CSS (Navy blue, purple accents, clean cards)
CUSTOM_CSS = """
<style>
/* Main app background and typography */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #0d1117 0%, #07090e 90%);
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #1e293b;
}

/* Sidebar header area */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 0 16px 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 16px;
}

.brand-icon {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    font-size: 20px;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);
}

.brand-title {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 40%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 0.75rem;
    color: #94a3b8;
    margin-top: -2px;
}

/* Status indicator badge */
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-pill.online {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-pill.offline {
    background: rgba(239, 68, 68, 0.15);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}
.status-dot.online {
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
}
.status-dot.offline {
    background-color: #ef4444;
}

/* Document cards in sidebar */
.doc-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.doc-badge {
    font-size: 0.65rem;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 4px;
    text-transform: uppercase;
}
.doc-badge.pdf {
    background: rgba(239, 68, 68, 0.2);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.4);
}
.doc-badge.txt {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.4);
}

/* Chat area header */
.chat-header {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(8px);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Citation card */
.citation-container {
    background: #0f172a;
    border-left: 3px solid #8b5cf6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-top: 10px;
    font-size: 0.82rem;
    color: #cbd5e1;
}

.citation-header {
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 600;
    color: #c4b5fd;
    margin-bottom: 4px;
}

.citation-tag {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 2px 6px;
    font-size: 0.75rem;
    color: #a78bfa;
    margin-right: 6px;
}

/* Welcome Hero Box */
.welcome-hero {
    text-align: center;
    padding: 40px 20px;
    max-width: 650px;
    margin: 0 auto;
}
.welcome-hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
}
.welcome-hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    line-height: 1.6;
}

/* Bottom mascot info card in sidebar */
.assistant-mascot-card {
    background: linear-gradient(145deg, #1e1e2e, #131828);
    border: 1px solid #2d3748;
    border-radius: 12px;
    padding: 14px;
    margin-top: 20px;
    text-align: center;
}
.assistant-mascot-card .title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
}
.assistant-mascot-card .subtitle {
    font-size: 0.72rem;
    color: #8b5cf6;
    margin-top: 2px;
}

/* Smooth button styling */
.stButton button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
</style>
"""

