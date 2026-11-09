"""
Embedding Manager Module
Generates dense vector embeddings using Sentence Transformers with L2 normalization.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_DIMENSION


class EmbeddingManager:
    """Encodes text passages and queries into normalized dense vector embeddings."""

    _instance = None
    _model = None

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        """
        Initializes the sentence transformer embedding model.

        Args:
            model_name: Hugging Face model identifier (default: all-MiniLM-L6-v2).
        """
        self.model_name = model_name
        self._load_model()

    def _load_model(self):
        """Loads the sentence transformer model with local caching."""
        if EmbeddingManager._model is None or EmbeddingManager._instance != self.model_name:
            EmbeddingManager._model = SentenceTransformer(self.model_name)
            EmbeddingManager._instance = self.model_name
        self.model = EmbeddingManager._model

    @property
    def dimension(self) -> int:
        """Returns the embedding vector dimension (384 for all-MiniLM-L6-v2)."""
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension() or EMBEDDING_DIMENSION)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of text strings into normalized numpy vectors.

        Args:
            texts: List of text strings to embed.
            batch_size: Mini-batch size for inference.

        Returns:
            2D numpy array of shape (N, dimension) with float32 dtype and unit norm.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Enables cosine similarity via dot product (Inner Product)
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encodes a single user query into a normalized 2D numpy vector.

        Args:
            query: The user prompt string.

        Returns:
            2D numpy array of shape (1, dimension) with float32 dtype and unit norm.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode(
            [query.strip()],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return query_embedding.astype(np.float32)
