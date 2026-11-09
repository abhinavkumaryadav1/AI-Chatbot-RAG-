"""
Text Splitter Module
Splits document pages into overlapping text chunks with rich metadata using LangChain Text Splitters.
"""

from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


class ChunkSplitter:
    """Handles semantic text chunking while strictly preserving document and page boundaries."""

    def __init__(self, chunk_size: int = DEFAULT_CHUNK_SIZE, chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        """
        Initializes the chunk splitter with specified parameters.

        Args:
            chunk_size: Maximum character length per chunk (default: 800).
            chunk_overlap: Overlap character count between successive chunks (default: 150).
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be strictly less than chunk size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            keep_separator=True
        )

    def split_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits a list of page objects into chunks.
        Guarantees that chunks NEVER mix content across different pages or documents.

        Args:
            pages: List of page dicts with 'text', 'source', 'page'.

        Returns:
            List of chunk dicts:
            {
                'chunk_id': str,
                'text': str,
                'source': str,
                'page': int or str,
                'metadata': dict
            }
        """
        all_chunks: List[Dict[str, Any]] = []

        for page_data in pages:
            source = page_data.get("source", "unknown")
            page_num = page_data.get("page", 1)
            page_text = page_data.get("text", "")

            if not page_text or not page_text.strip():
                continue

            # Split only within the current page boundary
            raw_chunks = self._splitter.split_text(page_text)

            for chunk_idx, chunk_text in enumerate(raw_chunks):
                cleaned_chunk = chunk_text.strip()
                if not cleaned_chunk:
                    continue

                chunk_id = f"{source}_p{page_num}_c{chunk_idx + 1}"

                metadata = {
                    "source": source,
                    "page": page_num,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx + 1,
                    "char_count": len(cleaned_chunk)
                }

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": cleaned_chunk,
                    "source": source,
                    "page": page_num,
                    "metadata": metadata
                })

        return all_chunks

