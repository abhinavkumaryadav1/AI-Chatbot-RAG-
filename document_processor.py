"""
Document Processor Module
Handles text extraction from PDF (using PyMuPDF) and TXT files, preserving page metadata.
"""

from typing import List, Dict, Any
import io
import os
import pymupdf  # Official modern PyMuPDF import


class DocumentProcessingError(Exception):
    """Custom exception raised when document extraction fails."""
    pass


class DocumentProcessor:
    """Extracts text content and metadata from supported document files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    @classmethod
    def extract_from_pdf(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF file page by page using PyMuPDF.
        Preserves page numbers (1-indexed) and source filename.

        Args:
            file_bytes: Raw binary bytes of the PDF.
            filename: Name of the uploaded file.

        Returns:
            List of dicts with keys: 'text', 'source', 'page'.
        """
        if not file_bytes:
            raise DocumentProcessingError(f"The file '{filename}' is empty.")

        pages_data: List[Dict[str, Any]] = []

        try:
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise DocumentProcessingError(f"Failed to open PDF '{filename}': {str(e)}")

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise DocumentProcessingError(f"The PDF '{filename}' contains no pages.")

        has_extractable_text = False

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            try:
                page = doc[page_idx]
                text = page.get_text() or ""
                cleaned_text = text.strip()

                if cleaned_text:
                    has_extractable_text = True
                    pages_data.append({
                        "text": cleaned_text,
                        "source": filename,
                        "page": page_num
                    })
                else:
                    # Page exists but has no extractable text (e.g. image-only or blank)
                    # We log/skip or retain note if necessary, here we continue
                    pass
            except Exception as e:
                # Log or handle specific page corruption gracefully
                continue

        doc.close()

        if not has_extractable_text or not pages_data:
            raise DocumentProcessingError(
                f"No extractable text found in '{filename}'. It may be empty or contain only scanned images."
            )

        return pages_data

    @classmethod
    def extract_from_txt(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Extract text from a plain TXT file with robust encoding detection.

        Args:
            file_bytes: Raw bytes of the TXT file.
            filename: Name of the uploaded file.

        Returns:
            List of dicts with keys: 'text', 'source', 'page'.
        """
        if not file_bytes:
            raise DocumentProcessingError(f"The text file '{filename}' is empty.")

        if not file_bytes.strip():
            raise DocumentProcessingError(f"The text file '{filename}' contains only whitespace.")

        # Attempt decoding with standard encodings
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        decoded_text: str = ""

        for enc in encodings:
            try:
                decoded_text = file_bytes.decode(enc)
                break
            except (UnicodeDecodeError, LookupError):
                continue

        if not decoded_text:
            raise DocumentProcessingError(
                f"Could not decode text file '{filename}' with supported encodings (UTF-8, Latin-1)."
            )

        cleaned_text = decoded_text.strip()
        if not cleaned_text:
            raise DocumentProcessingError(f"The text file '{filename}' contains only whitespace.")

        return [{
            "text": cleaned_text,
            "source": filename,
            "page": 1  # Standard 1 for TXT files
        }]

    @classmethod
    def process_file(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Validates and processes an uploaded file (PDF or TXT).

        Args:
            file_bytes: Raw bytes of the file.
            filename: Name of the file with extension.

        Returns:
            List of extracted page dictionaries.
        """
        _, ext = os.path.splitext(filename.lower())

        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise DocumentProcessingError(
                f"Unsupported file format '{ext}'. SmartRAG supports {', '.join(cls.SUPPORTED_EXTENSIONS)}."
            )

        if ext == ".pdf":
            return cls.extract_from_pdf(file_bytes, filename)
        elif ext == ".txt":
            return cls.extract_from_txt(file_bytes, filename)
        else:
            raise DocumentProcessingError(f"Unsupported file type: {ext}")
