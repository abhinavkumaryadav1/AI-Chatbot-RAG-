# SmartRAG – Document-Based AI Chatbot

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![UI](https://img.shields.io/badge/Interface-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Vector Store](https://img.shields.io/badge/Vector%20Store-FAISS-00599C.svg)](https://github.com/facebookresearch/faiss)
[![Embeddings](https://img.shields.io/badge/Embeddings-Sentence%20Transformers%20(MiniLM)-green.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Local LLM](https://img.shields.io/badge/LLM-Ollama%20(Llama%203.2)-black.svg)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-Academic%20Practical-purple.svg)]()

> **Practical Assignment Project for Prompt Engineering**  
> A professional, production-grade, 100% local **Retrieval-Augmented Generation (RAG)** chatbot with zero paid API keys, zero cloud dependencies, full page-level source citations, and strict prompt-engineered anti-hallucination guardrails.

---

## 1. Project Overview

**SmartRAG** is an intelligent document-grounded AI conversational agent designed to answer complex user queries exclusively using knowledge extracted from uploaded PDF and TXT files. 

### Key Capabilities:
- **Multi-Document Support**: Simultaneously upload multiple PDF and TXT documents.
- **Accurate Page-Level Citations**: Retains exact 1-based page numbers for PDF sources.
- **100% Local & Free Execution**: Powered locally by **Ollama (`llama3.2:3b` / `llama3.2:1b`)**, **FAISS**, and **Sentence Transformers (`all-MiniLM-L6-v2`)**. No OpenAI API keys, paid credits, or third-party cloud data transmission required.
- **Strict Anti-Hallucination Guardrails**: Robust prompt engineering ensures the chatbot responds with:  
  `"I could not find this information in the uploaded documents."`  
  whenever information is missing or below the similarity confidence threshold.
- **Transparent Source Inspection**: Every response displays source filenames, page numbers, cosine similarity scores, and expandable panels showing the exact retrieved text passages.
- **Dynamic Index Management**: Real-time document removal with automatic, safe index rebuilding.

---

## 2. System Architecture & RAG Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                          SmartRAG Architecture                         │
└────────────────────────────────────────────────────────────────────────┘

 1. INGESTION & PARSING
   [ Upload PDF / TXT ] ──────────────► [ PyMuPDF / Native TXT ]
                                                 │
                                                 ▼
 2. CHUNKING                           [ Page-Preserved Text ]
   [ RecursiveCharacterTextSplitter ] ◄──────────┘
      - Chunk Size: 800 chars
      - Overlap: 150 chars
      - Strictly isolated per page/doc
                                                 │
                                                 ▼
 3. EMBEDDING & INDEXING               [ Text Chunks + Metadata ]
   [ all-MiniLM-L6-v2 ] ─────────────► [ 384-dim Dense Vectors ]
                                                 │
                                                 ▼
                                       [ In-Memory FAISS Index ]
                                       (IndexFlatIP - Cosine Similarity)

 4. RETRIEVAL & GUARDRAILS
   [ User Query ] ──► [ Query Vector ] ──► [ FAISS Similarity Search ]
                                                 │
                                                 ▼
                                        Top-K Chunks (Score >= 0.35)
                                                 │
                                 ┌───────────────┴───────────────┐
                                 ▼                               ▼
                          [ Score >= Threshold ]        [ Below Threshold ]
                                 │                               │
                                 ▼                               ▼
 5. PROMPT SYNTHESIS     [ Strict RAG Prompt ]       [ Return Fallback Message: ]
                          (System Guardrails)        "I could not find this..."
                                 │
                                 ▼
 6. LOCAL LLM INFERENCE
   [ Ollama HTTP API ] (llama3.2:3b / llama3.2:1b)
                                 │
                                 ▼
 7. USER INTERFACE
   [ Streamlit UI ] ◄── Answer + Markdown + Page Citations + Passage Expanders
```

---

## 3. Technology Stack

| Component | Technology | Rationale / Specification |
| :--- | :--- | :--- |
| **Language** | Python 3.11 / 3.12 / 3.13 | High performance, cross-platform compatibility |
| **User Interface** | Streamlit | Modern dark theme, reactive chat bubbles, settings drawer |
| **PDF Extraction** | PyMuPDF (`pymupdf`) | Fast, accurate page-by-page text parsing with metadata |
| **TXT Extraction** | Native Python | Multi-encoding fallback (UTF-8, Latin-1, CP1252) |
| **Text Chunking** | `langchain-text-splitters` | `RecursiveCharacterTextSplitter` with strict page isolation |
| **Embeddings** | `sentence-transformers` | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, normalized) |
| **Vector Database** | FAISS (`faiss-cpu`) | In-memory `IndexFlatIP` calculating exact cosine similarities |
| **Local LLM** | Ollama | Local model runner without API keys |
| **Default Model** | `llama3.2:3b` | Meta's state-of-the-art compact multilingual model (fallback: `llama3.2:1b`) |
| **Testing** | `pytest` | 10 automated unit and integration tests |

---

## 4. Project File Structure

```
RAG Project/
├── app.py                      # Streamlit application with modern UI & chat flow
├── config.py                   # Global constants, hyperparameter defaults & custom CSS
├── document_processor.py       # PDF/TXT extraction, page metadata preservation, error handling
├── text_splitter.py           # Recursive text chunking isolating document & page boundaries
├── embedding_manager.py        # SentenceTransformers wrapper with L2 normalization
├── vector_store.py             # FAISS in-memory vector database with threshold filtering
├── ollama_client.py            # Local Ollama HTTP API client with streaming & health checks
├── prompt_templates.py         # Strict prompt engineering system prompt & context formatting
├── test_rag.py                 # Pytest test suite (10 comprehensive tests)
├── run_evaluation.py           # Standalone verification script for the 5 college test questions
├── generate_sample_docs.py     # Generator script for multi-page test documents
├── sample_documents/
│   ├── Employee_Handbook.pdf   # 3-page corporate policy manual (Leave, WFH, Ethics)
│   └── HR_Guidelines.txt       # Benefits, probation, and bonus policies
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Complete documentation & practical manual
```

---

## 5. Installation & Setup Instructions

### Prerequisites
- Python 3.11 or higher installed on your system.
- 4GB+ RAM available for CPU inference.

### Step 1: Clone or Open the Workspace
```bash
cd "c:\Users\hp\Desktop\RAG Project"
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
```

Activate the virtual environment:
- **Windows (PowerShell)**:
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
- **Windows (Command Prompt)**:
  ```cmd
  .\venv\Scripts\activate.bat
  ```
- **macOS / Linux**:
  ```bash
  source venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Ollama Setup (Local LLM)

SmartRAG utilizes **Ollama** to run models locally without any external API keys or cloud telemetry.

### Step 1: Install Ollama
Download and install Ollama from the official website:
- **Website**: [https://ollama.com/download](https://ollama.com/download)
- Or via Windows Package Manager:
  ```powershell
  winget install --id Ollama.Ollama -e
  ```

### Step 2: Start the Ollama Service
Ensure Ollama is running in the background:
```bash
ollama serve
```
*(On Windows/macOS, launching the Ollama desktop app automatically starts the service).*

### Step 3: Download the Language Model
Pull the default **Llama 3.2 3B** model:
```bash
ollama pull llama3.2:3b
```
*(Optional lightweight alternative for ultra-fast CPU inference: `ollama pull llama3.2:1b`)*

Verify your installed models:
```bash
ollama list
```

---

## 7. Running the Application

Launch the SmartRAG Streamlit web application:

```bash
streamlit run app.py
```

Once running, open your browser to:
👉 **`http://localhost:8501`**

---

## 8. RAG Workflow Explanation

1. **Document Ingestion**:
   - The user uploads PDF or TXT files via the sidebar.
   - For PDFs, `PyMuPDF` reads each page individually, attaching 1-indexed page numbers.
   - For TXT files, encoding detection guarantees error-free text decoding.
2. **Deterministic Chunking**:
   - `ChunkSplitter` partitions each page into segments of 800 characters with 150-character overlaps.
   - Text chunks never span across document or page boundaries.
3. **Normalized Embedding**:
   - `EmbeddingManager` computes 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
   - Embeddings are unit-normalized ($L_2$ norm = 1.0) to enable cosine similarity calculation via inner product.
4. **FAISS Indexing**:
   - Chunks and vectors are indexed in an in-memory `faiss.IndexFlatIP`.
5. **Semantic Retrieval**:
   - User queries are embedded using the same Sentence Transformers model.
   - FAISS searches the vector space for the top-$K$ nearest chunks.
   - Chunks with similarity score $< 0.35$ are filtered out.
6. **Prompt Synthesis & Guardrails**:
   - If no chunks pass the threshold, the chatbot immediately returns the fallback message without invoking the LLM.
   - If valid chunks exist, they are structured into a strict prompt enforcing:
     *"Answer questions using only the retrieved document context. Do not use outside knowledge."*
7. **Document-Grounded Answer & Citation**:
   - Ollama streams the response in real time.
   - The exact document name, page number, similarity score, and excerpt passage are displayed below the answer.

---

## 9. Automated Testing & Verification

SmartRAG includes a complete automated test suite (`pytest`) and an end-to-end evaluation runner.

### Running Automated Tests
```bash
pytest test_rag.py -v
```
All **10 test suites** verify:
1. `test_pdf_extraction`: Multi-page text extraction with page numbers.
2. `test_txt_extraction`: Multi-encoding TXT extraction.
3. `test_empty_and_unsupported_files`: Exception handling for blank and invalid files.
4. `test_chunking_and_metadata_preservation`: Boundary preservation & metadata integrity.
5. `test_embedding_generation`: Vector dimensions (384) & L2 normalization.
6. `test_faiss_indexing_and_retrieval`: Cosine similarity search and score calculation.
7. `test_document_removal_and_rebuilding`: Safe index cleanup when removing files.
8. `test_ollama_unavailable_graceful_handling`: Graceful error reporting when Ollama is offline.
9. `test_prompt_engineering_templates`: Guardrail formatting and strict system prompt.
10. `test_five_college_test_questions`: Complete RAG pipeline verification.

---

## 10. College Assignment Test Plan & Results

The system was evaluated against the 5 mandated assignment test cases using the generated `sample_documents/Employee_Handbook.pdf`:

| # | Question Type | User Question | Retrieved Sources | Grounded Answer Summary |
| :-: | :--- | :--- | :--- | :--- |
| **Q1** | **Directly Answerable** | *What is the annual and sick leave entitlement for full-time employees?* | `Employee_Handbook.pdf` (Page 1) | 20 days annual leave, 10 days sick leave, and 5 days casual leave. |
| **Q2** | **Directly Answerable** | *How many days in advance must leave requests be submitted, and through which portal?* | `Employee_Handbook.pdf` (Page 1) | At least 3 working days in advance through the internal HR portal. |
| **Q3** | **Directly Answerable** | *What is the policy regarding working from home?* | `Employee_Handbook.pdf` (Page 2) | Employees may work from home up to 3 days per week with prior manager approval. Permanent remote work is strictly not permitted for core teams. |
| **Q4** | **Multi-Chunk Reasoning** | *Can an employee work permanently from home while taking casual leave, and what approvals are needed?* | `Employee_Handbook.pdf` (Pages 1 & 2) | Combines Page 2 (permanent remote work restriction) and Page 1 (manager approval and HR portal procedures). |
| **Q5** | **Unanswerable (Anti-Hallucination)** | *What is the company's reimbursement policy for international business travel flights and 5-star hotels?* | *None (All chunks score below 0.35 threshold)* | **"I could not find this information in the uploaded documents."** |

Run the standalone evaluation anytime:
```bash
python run_evaluation.py
```

---

## 11. Troubleshooting Guide

| Issue | Root Cause | Solution |
| :--- | :--- | :--- |
| **"Could not connect to Ollama"** | The Ollama background service is not running. | Open a terminal and run `ollama serve`, or open the Ollama application from the Start menu. |
| **"Model 'llama3.2:3b' not found"** | The model hasn't been pulled yet. | Run `ollama pull llama3.2:3b` in your terminal. You can also switch to `llama3.2:1b` in the sidebar settings. |
| **"No extractable text found in PDF"** | The uploaded PDF consists of scanned images without an embedded text layer. | Use an electronic PDF with text characters, or run OCR software on the file prior to upload. |
| **"Chunk overlap must be less than chunk size"** | Overlap slider set higher than chunk size in settings. | Adjust the **Chunk Overlap** slider in the sidebar to be lower than **Chunk Size**. |
| **Streamlit port conflict** | Port 8501 is occupied by another application. | Run `streamlit run app.py --server.port 8502`. |

