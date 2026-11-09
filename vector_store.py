"""
Vector Store Module
Manages in-memory FAISS indexing, metadata storage, and semantic retrieval with score thresholds.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from config import EMBEDDING_DIMENSION, DEFAULT_TOP_K, DEFAULT_SIMILARITY_THRESHOLD


class FAISSVectorStore:
    """In-memory FAISS vector store implementing Cosine Similarity via Inner Product (IndexFlatIP)."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION):
        """
        Initializes an empty FAISS index and metadata store.

        Args:
            dimension: Dimensionality of embedding vectors (default: 384).
        """
        self.dimension = dimension
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata: List[Dict[str, Any]] = []
        self._document_registry: Dict[str, List[int]] = {}  # doc_name -> list of chunk indices

    @property
    def total_chunks(self) -> int:
        """Total number of indexed text chunks."""
        return len(self.chunks_metadata)

    @property
    def indexed_documents(self) -> List[str]:
        """List of distinct filenames currently indexed."""
        return list(self._document_registry.keys())

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray) -> None:
        """
        Inserts new text chunks and their embeddings into the FAISS index.

        Args:
            chunks: List of chunk dictionaries containing 'text', 'source', 'page', 'chunk_id'.
            embeddings: 2D numpy array of shape (N, dimension), normalized.
        """
        if not chunks or len(embeddings) == 0:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: received {len(chunks)} chunks but {len(embeddings)} embeddings."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}."
            )

        start_idx = len(self.chunks_metadata)

        # Add vectors to FAISS index
        self.index.add(embeddings.astype(np.float32))

        # Update metadata storage and document registry
        for offset, chunk in enumerate(chunks):
            current_idx = start_idx + offset
            self.chunks_metadata.append(chunk)

            doc_name = chunk.get("source", "unknown")
            if doc_name not in self._document_registry:
                self._document_registry[doc_name] = []
            self._document_registry[doc_name].append(current_idx)

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        Performs semantic similarity retrieval using cosine similarity.
        Filters out any chunks whose similarity score falls below `threshold`.

        Args:
            query_embedding: 2D numpy array of shape (1, dimension).
            top_k: Maximum number of relevant chunks to retrieve.
            threshold: Minimum cosine similarity score (between 0.0 and 1.0).

        Returns:
            List of matching chunk dicts with an added 'score' field, ordered descending by score.
        """
        if self.total_chunks == 0:
            return []

        k = min(top_k, self.total_chunks)
        query_vec = query_embedding.astype(np.float32)

        # FAISS search: D contains cosine similarities (inner product of normalized vectors)
        # I contains indices in chunks_metadata
        distances, indices = self.index.search(query_vec, k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.chunks_metadata):
                continue

            float_score = float(score)

            # Apply similarity confidence check
            if float_score >= threshold:
                chunk_copy = dict(self.chunks_metadata[idx])
                chunk_copy["score"] = round(float_score, 4)
                results.append(chunk_copy)

        return results

    def remove_document(self, document_name: str, embedding_manager=None) -> None:
        """
        Safely removes a document from the index and rebuilds the FAISS store
        with the remaining documents to ensure index consistency.

        Args:
            document_name: Filename of the document to remove.
            embedding_manager: Optional EmbeddingManager instance to re-embed remaining chunks.
        """
        if document_name not in self._document_registry:
            return

        remaining_chunks = [
            chunk for chunk in self.chunks_metadata
            if chunk.get("source") != document_name
        ]

        self.clear()

        if remaining_chunks and embedding_manager is not None:
            texts = [c["text"] for c in remaining_chunks]
            embeddings = embedding_manager.encode_texts(texts)
            self.add_chunks(remaining_chunks, embeddings)

    def clear(self) -> None:
        """Clears all vectors and metadata, resetting to empty state."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []
        self._document_registry = {}

