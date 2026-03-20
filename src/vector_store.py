"""
ChromaDB vector store wrapper.
Handles adding, querying, deleting, and listing documents in the local vector DB.
"""

from __future__ import annotations

from typing import Any

import chromadb

from src.config import CHROMA_DB_DIR

# ── Constants ────────────────────────────────────────────────────────

COLLECTION_NAME = "pensieve_docs"

# ── Client / Collection ─────────────────────────────────────────────

_client: chromadb.PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Lazily initialise ChromaDB and return the main collection."""
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ── Public API ───────────────────────────────────────────────────────

def add_documents(
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict[str, Any]],
) -> None:
    """
    Add document chunks to the vector store.

    Args:
        ids: Unique ID for each chunk.
        texts: The chunk text contents.
        embeddings: Pre-computed embedding vectors.
        metadatas: Metadata dicts for each chunk.
    """
    collection = _get_collection()
    # ChromaDB supports upsert to avoid duplicates
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query(
    query_embedding: list[float],
    n_results: int = 5,
) -> dict[str, Any]:
    """
    Query the vector store for the most similar chunks.

    Args:
        query_embedding: The embedding of the user's question.
        n_results: Number of results to return.

    Returns:
        ChromaDB query results dict with 'ids', 'documents', 'metadatas', 'distances'.
    """
    collection = _get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    return results


def delete_source(source_name: str) -> int:
    """
    Delete all chunks belonging to a specific source.

    Args:
        source_name: The source identifier to delete.

    Returns:
        Number of chunks deleted.
    """
    collection = _get_collection()

    # Find all chunk IDs for this source
    results = collection.get(
        where={"source": source_name},
        include=[],
    )
    chunk_ids = results["ids"]

    if chunk_ids:
        collection.delete(ids=chunk_ids)

    return len(chunk_ids)


def list_sources() -> list[dict[str, Any]]:
    """
    List all unique sources in the vector store.

    Returns:
        List of dicts with 'source', 'type', and 'num_chunks'.
    """
    collection = _get_collection()
    all_data = collection.get(include=["metadatas"])

    sources: dict[str, dict[str, Any]] = {}
    for meta in all_data["metadatas"]:
        source = meta.get("source", "unknown")
        if source not in sources:
            sources[source] = {
                "source": source,
                "type": meta.get("type", "unknown"),
                "num_chunks": 0,
            }
        sources[source]["num_chunks"] += 1

    return list(sources.values())


def get_source_chunks(source_name: str) -> list[dict[str, Any]]:
    """
    Retrieve all chunks for a given source, ordered by chunk index.

    Args:
        source_name: The source identifier.

    Returns:
        List of dicts with 'text' and 'metadata', sorted by chunk_index.
    """
    collection = _get_collection()
    results = collection.get(
        where={"source": source_name},
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        chunks.append({"text": doc, "metadata": meta})

    # Sort by chunk_index for correct ordering
    chunks.sort(key=lambda c: c["metadata"].get("chunk_index", 0))
    return chunks


def count() -> int:
    """Return the total number of chunks in the vector store."""
    collection = _get_collection()
    return collection.count()


def reset_collection() -> None:
    """Delete and recreate the collection. Used for testing."""
    global _collection
    client = _get_collection()  # ensure client is initialised
    if _client is not None:
        _client.delete_collection(COLLECTION_NAME)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
