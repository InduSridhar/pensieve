"""
Text chunker using LangChain's RecursiveCharacterTextSplitter.
Splits documents into overlapping chunks with metadata.
"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(
    text: str,
    metadata: dict[str, Any],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Split text into overlapping chunks and attach metadata to each.

    Args:
        text: The raw text to split.
        metadata: Base metadata (source, type, etc.) to attach to each chunk.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of dicts, each with 'text', 'metadata' (including chunk_index), and 'id'.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    splits = splitter.split_text(text)

    chunks = []
    source_name = metadata.get("source", "unknown")

    for i, chunk_text_str in enumerate(splits):
        chunk_id = f"{source_name}::chunk_{i}"
        chunk_metadata = {
            **metadata,
            "chunk_index": i,
            "total_chunks": len(splits),
        }
        chunks.append({
            "id": chunk_id,
            "text": chunk_text_str,
            "metadata": chunk_metadata,
        })

    return chunks
