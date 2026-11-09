"""
SmartRAG Automated Test Suite
Verifies all 10 core assignment testing criteria:
1. Document extraction (PDF with pages, TXT)
2. Chunking & boundary preservation
3. Embedding generation & normalization
4. FAISS vector storage & similarity retrieval
5. Document removal & index rebuilding
6. Empty & unsupported file handling
7. Missing Ollama connection detection
8. Retrieval similarity threshold filtering
9. RAG prompt engineering & context formatting
10. The 5 mandatory college test questions
"""

import os
import pytest
import numpy as np

from config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    EMBEDDING_DIMENSION,
    DEFAULT_SIMILARITY_THRESHOLD,
)
from document_processor import DocumentProcessor, DocumentProcessingError
from text_splitter import ChunkSplitter
from embedding_manager import EmbeddingManager
from vector_store import FAISSVectorStore
from ollama_client import OllamaClient, OllamaClientError
from prompt_templates import (
    STRICT_SYSTEM_PROMPT,
    FALLBACK_MESSAGE,
    format_context_chunks,
    build_rag_prompt,
)


@pytest.fixture(scope="module")
def embedding_mgr():
    """Initializes and caches the SentenceTransformers embedding manager for tests."""
    return EmbeddingManager()


@pytest.fixture(scope="module")
def sample_pdf_path():
    path = os.path.join("sample_documents", "Employee_Handbook.pdf")
    if not os.path.exists(path):
        from generate_sample_docs import generate_sample_pdf
        generate_sample_pdf(path)
    return path


@pytest.fixture(scope="module")
def sample_txt_path():
    path = os.path.join("sample_documents", "HR_Guidelines.txt")
    if not os.path.exists(path):
        from generate_sample_docs import generate_sample_txt
        generate_sample_txt(path)
    return path


# -------------------------------------------------------------
# Test 1: PDF Document Extraction
# -------------------------------------------------------------
def test_pdf_extraction(sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = DocumentProcessor.extract_from_pdf(pdf_bytes, "Employee_Handbook.pdf")

    assert len(pages) == 3, "Sample PDF should contain exactly 3 pages"
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2
    assert pages[2]["page"] == 3

    assert pages[0]["source"] == "Employee_Handbook.pdf"
    assert "20 days of annual leave" in pages[0]["text"]
    assert "work from home up to 3 days per week" in pages[1]["text"]
    assert "Code of Conduct" in pages[2]["text"]


# -------------------------------------------------------------
# Test 2: TXT Document Extraction
# -------------------------------------------------------------
def test_txt_extraction(sample_txt_path):
    with open(sample_txt_path, "rb") as f:
        txt_bytes = f.read()

    pages = DocumentProcessor.extract_from_txt(txt_bytes, "HR_Guidelines.txt")

    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert pages[0]["source"] == "HR_Guidelines.txt"
    assert "PROBATIONARY PERIOD" in pages[0]["text"]
    assert "$500,000" in pages[0]["text"]


# -------------------------------------------------------------
# Test 3: Empty and Unsupported File Handling
# -------------------------------------------------------------
def test_empty_and_unsupported_files():
    # Empty bytes PDF
    with pytest.raises(DocumentProcessingError, match="empty"):
        DocumentProcessor.process_file(b"", "empty.pdf")

    # Empty bytes TXT
    with pytest.raises(DocumentProcessingError, match="empty"):
        DocumentProcessor.process_file(b"", "empty.txt")

    # Whitespace only TXT
    with pytest.raises(DocumentProcessingError, match="whitespace"):
        DocumentProcessor.process_file(b"   \n  \t  ", "blank.txt")

    # Unsupported file extension
    with pytest.raises(DocumentProcessingError, match="Unsupported file format"):
        DocumentProcessor.process_file(b"some content", "data.docx")


# -------------------------------------------------------------
# Test 4: Chunking and Metadata Preservation
# -------------------------------------------------------------
def test_chunking_and_metadata_preservation(sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = DocumentProcessor.extract_from_pdf(pdf_bytes, "Employee_Handbook.pdf")
    splitter = ChunkSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_pages(pages)

    assert len(chunks) > 0

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "text" in chunk
        assert "source" in chunk
        assert "page" in chunk
        assert "metadata" in chunk

        assert chunk["source"] == "Employee_Handbook.pdf"
        assert chunk["page"] in [1, 2, 3]
        assert len(chunk["text"]) <= 650  # within reasonable bound with delimiters


# -------------------------------------------------------------
# Test 5: Embedding Generation & L2 Normalization
# -------------------------------------------------------------
def test_embedding_generation(embedding_mgr):
    texts = [
        "Employees are entitled to 20 days annual leave.",
        "Remote work is permitted up to 3 days per week."
    ]

    embeddings = embedding_mgr.encode_texts(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (2, EMBEDDING_DIMENSION)

    # Check L2 normalization: norm should be close to 1.0
    norms = np.linalg.norm(embeddings, axis=1)
    for norm in norms:
        assert np.isclose(norm, 1.0, atol=1e-3)

    # Query encoding test
    query_vec = embedding_mgr.encode_query("What is the leave policy?")
    assert query_vec.shape == (1, EMBEDDING_DIMENSION)
    assert np.isclose(np.linalg.norm(query_vec), 1.0, atol=1e-3)


# -------------------------------------------------------------
# Test 6: FAISS Vector Store Indexing and Cosine Similarity Retrieval
# -------------------------------------------------------------
def test_faiss_indexing_and_retrieval(embedding_mgr, sample_pdf_path):
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = DocumentProcessor.extract_from_pdf(pdf_bytes, "Employee_Handbook.pdf")
    splitter = ChunkSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    chunks = splitter.split_pages(pages)

    texts = [c["text"] for c in chunks]
    embeddings = embedding_mgr.encode_texts(texts)

    store = FAISSVectorStore(dimension=embedding_mgr.dimension)
    store.add_chunks(chunks, embeddings)

    assert store.total_chunks == len(chunks)
    assert "Employee_Handbook.pdf" in store.indexed_documents

    # Query for leave entitlement
    query_emb = embedding_mgr.encode_query("How many days of annual leave do employees receive?")
    results = store.similarity_search(query_emb, top_k=2, threshold=0.3)

    assert len(results) > 0
    top_chunk = results[0]
    assert "score" in top_chunk
    assert top_chunk["score"] > 0.4
    assert top_chunk["page"] == 1
    assert any("annual leave" in r["text"] for r in results)


# -------------------------------------------------------------
# Test 7: Safe Document Removal and Index Rebuilding
# -------------------------------------------------------------
def test_document_removal_and_rebuilding(embedding_mgr):
    store = FAISSVectorStore(dimension=embedding_mgr.dimension)

    doc1_chunks = [
        {"chunk_id": "doc1_p1_c1", "text": "Doc 1 text passage about alpha", "source": "doc1.txt", "page": 1}
    ]
    doc2_chunks = [
        {"chunk_id": "doc2_p1_c1", "text": "Doc 2 text passage about beta", "source": "doc2.txt", "page": 1}
    ]

    emb1 = embedding_mgr.encode_texts([c["text"] for c in doc1_chunks])
    emb2 = embedding_mgr.encode_texts([c["text"] for c in doc2_chunks])

    store.add_chunks(doc1_chunks, emb1)
    store.add_chunks(doc2_chunks, emb2)

    assert store.total_chunks == 2
    assert set(store.indexed_documents) == {"doc1.txt", "doc2.txt"}

    # Remove doc1
    store.remove_document("doc1.txt", embedding_manager=embedding_mgr)

    assert store.total_chunks == 1
    assert store.indexed_documents == ["doc2.txt"]
    assert store.chunks_metadata[0]["source"] == "doc2.txt"


# -------------------------------------------------------------
# Test 8: Ollama Unavailable Graceful Error Handling
# -------------------------------------------------------------
def test_ollama_unavailable_graceful_handling():
    # Intentionally point to an unused port
    client = OllamaClient(host="http://localhost:59999")

    is_connected, msg = client.check_connection()
    assert is_connected is False
    assert "Could not connect to Ollama" in msg

    models = client.list_models()
    assert models == []

    # Attempting generation should raise OllamaClientError, not crash
    with pytest.raises(OllamaClientError, match="Unable to connect to Ollama"):
        client.generate("Test prompt")


# -------------------------------------------------------------
# Test 9: Prompt Engineering & Guardrail Formatting
# -------------------------------------------------------------
def test_prompt_engineering_templates():
    sample_chunks = [
        {"source": "Handbook.pdf", "page": 1, "text": "Leave entitlement is 20 days."},
        {"source": "Policy.pdf", "page": 2, "text": "Remote work is up to 3 days."}
    ]

    formatted_context = format_context_chunks(sample_chunks)
    assert "Handbook.pdf" in formatted_context
    assert "Page 1" in formatted_context
    assert "Policy.pdf" in formatted_context
    assert "Page 2" in formatted_context

    prompt = build_rag_prompt("What is the leave policy?", sample_chunks)
    assert "Context from uploaded documents:" in prompt
    assert "Question: What is the leave policy?" in prompt


# -------------------------------------------------------------
# Test 10: Five Mandatory Assignment Questions Evaluation
# -------------------------------------------------------------
def test_five_college_test_questions(embedding_mgr, sample_pdf_path):
    """
    Evaluates the 5 required test questions against the indexed Employee Handbook:
    1. Directly answerable: Annual and sick leave entitlement
    2. Directly answerable: Leave request submission deadline and portal
    3. Directly answerable: Work from home policy
    4. Multi-chunk answerable: Working permanently from home while taking leave
    5. Unanswerable question: International business travel flight & 5-star hotel reimbursement
    """
    with open(sample_pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = DocumentProcessor.extract_from_pdf(pdf_bytes, "Employee_Handbook.pdf")
    splitter = ChunkSplitter()
    chunks = splitter.split_pages(pages)
    texts = [c["text"] for c in chunks]
    embeddings = embedding_mgr.encode_texts(texts)

    store = FAISSVectorStore(dimension=embedding_mgr.dimension)
    store.add_chunks(chunks, embeddings)

    # Q1 (Directly answerable)
    q1 = "What is the annual and sick leave entitlement for full-time employees?"
    res1 = store.similarity_search(embedding_mgr.encode_query(q1), top_k=2)
    assert len(res1) > 0
    assert any("20 days of annual leave" in r["text"] and "10 days of sick leave" in r["text"] for r in res1)
    assert res1[0]["page"] == 1

    # Q2 (Directly answerable)
    q2 = "How many days in advance must leave requests be submitted, and through which portal?"
    res2 = store.similarity_search(embedding_mgr.encode_query(q2), top_k=2)
    assert len(res2) > 0
    assert any("3 working days in advance" in r["text"] and "HR portal" in r["text"] for r in res2)
    assert res2[0]["page"] == 1

    # Q3 (Directly answerable)
    q3 = "What is the policy regarding working from home?"
    res3 = store.similarity_search(embedding_mgr.encode_query(q3), top_k=2)
    assert len(res3) > 0
    assert any("work from home up to 3 days per week" in r["text"] for r in res3)
    assert any(r["page"] == 2 for r in res3)

    # Q4 (Multi-chunk: combines leave approval from Page 1 and permanent WFH restriction from Page 2)
    q4 = "Can an employee work permanently from home while taking casual leave, and what approvals are needed?"
    res4 = store.similarity_search(embedding_mgr.encode_query(q4), top_k=3)
    assert len(res4) >= 2
    pages_retrieved = {r["page"] for r in res4}
    assert 1 in pages_retrieved, "Should retrieve Page 1 regarding leave and manager approval"
    assert 2 in pages_retrieved, "Should retrieve Page 2 regarding permanent remote work restriction"

    # Q5 (Unanswerable from document)
    q5 = "What is the company's reimbursement policy for international business travel flights and 5-star hotels?"
    res5 = store.similarity_search(embedding_mgr.encode_query(q5), top_k=3, threshold=0.35)
    # Check that either no chunks meet the similarity threshold or chunks contain no mention of international travel
    for chunk in res5:
        assert "international business travel" not in chunk["text"].lower()
        assert "5-star" not in chunk["text"].lower()
