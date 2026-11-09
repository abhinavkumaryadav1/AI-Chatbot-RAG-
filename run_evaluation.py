"""
SmartRAG Evaluation Script
Runs the 5 mandatory college assignment questions against the local RAG pipeline:
1. Directly answerable: Annual and sick leave entitlement
2. Directly answerable: Leave request submission deadline and portal
3. Directly answerable: Work from home policy
4. Multi-chunk answerable: Working permanently from home while taking casual leave
5. Unanswerable question: International business travel flight & 5-star hotel reimbursement
"""

from document_processor import DocumentProcessor
from text_splitter import ChunkSplitter
from embedding_manager import EmbeddingManager
from vector_store import FAISSVectorStore
from ollama_client import OllamaClient
import prompt_templates


def run_evaluation():
    print("=" * 70)
    print(" SmartRAG End-to-End Evaluation: 5 Assignment Test Questions")
    print("=" * 70)

    # 1. Process document
    pdf_path = "sample_documents/Employee_Handbook.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    pages = DocumentProcessor.extract_from_pdf(pdf_bytes, "Employee_Handbook.pdf")
    print(f"[*] Extracted {len(pages)} pages from {pdf_path}")

    # 2. Chunk text
    splitter = ChunkSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_pages(pages)
    print(f"[*] Generated {len(chunks)} text chunks (chunk_size=800, chunk_overlap=150)")

    # 3. Embed & Index
    emb_mgr = EmbeddingManager()
    embeddings = emb_mgr.encode_texts([c["text"] for c in chunks])
    store = FAISSVectorStore(emb_mgr.dimension)
    store.add_chunks(chunks, embeddings)
    print(f"[*] Indexed {store.total_chunks} chunks into FAISS vector store\n")

    # 4. Check Ollama
    cli = OllamaClient()
    models = cli.list_models()
    model_to_use = "llama3.2:3b" if "llama3.2:3b" in models else ("llama3.2:1b" if "llama3.2:1b" in models else models[0])
    print(f"[*] Using local Ollama model: {model_to_use}")

    questions = [
        "What is the annual and sick leave entitlement for full-time employees?",
        "How many days in advance must leave requests be submitted, and through which portal?",
        "What is the policy regarding working from home?",
        "Can an employee work permanently from home while taking casual leave, and what approvals are needed?",
        "What is the company's reimbursement policy for international business travel flights and 5-star hotels?"
    ]

    for idx, q in enumerate(questions, 1):
        print("\n" + "=" * 70)
        print(f"QUESTION {idx}: {q}")
        print("=" * 70)

        q_emb = emb_mgr.encode_query(q)
        hits = store.similarity_search(q_emb, top_k=3, threshold=0.35)

        if not hits:
            print("[RETRIEVAL RESULT]: No chunks exceeded confidence threshold (0.35).")
            print(f"[ANSWER]: {prompt_templates.FALLBACK_MESSAGE}")
        else:
            prompt = prompt_templates.build_rag_prompt(q, hits)
            ans = cli.generate(
                prompt=prompt,
                system=prompt_templates.STRICT_SYSTEM_PROMPT,
                model=model_to_use,
                temperature=0.1
            )
            print(f"[ANSWER]:\n{ans}\n")
            print("[RETRIEVED CITATIONS]:")
            for h in hits:
                src = h["source"]
                pg = h["page"]
                sc = h["score"]
                print(f"  - Document: {src} | Page: {pg} | Cosine Similarity: {sc:.4f}")
                snippet = h["text"].replace("\n", " ")[:120]
                print(f"    Passage: \"{snippet}...\"")


if __name__ == "__main__":
    run_evaluation()
