"""
RAG Pipeline — the core orchestrator.
Handles ingestion (load → chunk → embed → store) and querying (embed → retrieve → generate).
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

from src.document_loader import load_pdf, load_txt, load_youtube
from src.chunker import chunk_text
from src.embeddings import embed_text, embed_batch
from src import vector_store
from src.chat import build_rag_prompt, build_summary_prompt
from src.config import TOP_K


# ── Ingest ───────────────────────────────────────────────────────────

def ingest_pdf(file: BytesIO, filename: str) -> dict[str, Any]:
    """Ingest a PDF file into the vector store and return its summary."""
    doc = load_pdf(file, filename)
    return _ingest_document(doc)


def ingest_txt(file: BytesIO, filename: str) -> dict[str, Any]:
    """Ingest a text file into the vector store and return its summary."""
    doc = load_txt(file, filename)
    return _ingest_document(doc)


def ingest_youtube(url: str) -> dict[str, Any]:
    """Ingest a YouTube video transcript into the vector store and return its summary."""
    doc = load_youtube(url)
    return _ingest_document(doc)


def _ingest_document(doc: dict[str, Any]) -> dict[str, Any]:
    """
    Internal: chunk, embed, store, and auto-summarize a loaded document.

    Args:
        doc: Dict with 'text' and 'metadata' from a document loader.

    Returns:
        Dict with 'source', 'num_chunks', and 'summary'.
    """
    text = doc["text"]
    metadata = doc["metadata"]

    # 1. Chunk the text
    chunks = chunk_text(text, metadata)

    # 2. Embed all chunks
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_batch(chunk_texts)

    # 3. Store in ChromaDB
    ids = [c["id"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    vector_store.add_documents(ids, chunk_texts, embeddings, metadatas)

    # 4. Generate auto-summary
    summary = summarize_source(metadata["source"])

    return {
        "source": metadata["source"],
        "type": metadata["type"],
        "num_chunks": len(chunks),
        "summary": summary,
    }


# ── Query ────────────────────────────────────────────────────────────

def query(question: str, chat_history: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """
    Answer a question using RAG: embed → retrieve → generate.

    Args:
        question: The user's question.
        chat_history: Optional list of previous messages [{"role": ..., "content": ...}].

    Returns:
        Dict with 'answer', 'sources' (retrieved chunks), and 'context_used'.
    """
    # 1. Embed the question
    question_embedding = embed_text(question)

    # 2. Retrieve top-K relevant chunks
    results = vector_store.query(question_embedding, n_results=TOP_K)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    distances = results["distances"][0] if results["distances"] else []

    # 3. Build context from retrieved chunks
    context_chunks = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        context_chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "distance": dist,
        })

    # 4. Build the prompt and generate answer
    prompt = build_rag_prompt(question, context_chunks, chat_history)
    answer = _generate(prompt)

    return {
        "answer": answer,
        "sources": context_chunks,
        "context_used": len(context_chunks),
    }


# ── Summarize ────────────────────────────────────────────────────────

def summarize_source(source_name: str) -> str:
    """
    Generate a summary for a source by retrieving its chunks and prompting Gemini.

    Args:
        source_name: The source identifier.

    Returns:
        A summary string.
    """
    chunks = vector_store.get_source_chunks(source_name)

    if not chunks:
        return "No content found for this source."

    # Use first N chunks (up to ~8000 chars) for summarisation
    combined_text = ""
    for chunk in chunks:
        if len(combined_text) + len(chunk["text"]) > 8000:
            break
        combined_text += chunk["text"] + "\n\n"

    prompt = build_summary_prompt(combined_text, source_name)
    return _generate(prompt)


# ── Gemini Generation ────────────────────────────────────────────────

def _generate(prompt: str) -> str:
    """
    Call Gemini to generate text from a prompt.

    Args:
        prompt: The full prompt string.

    Returns:
        The model's response text.
    """
    from google import genai
    from src.config import GOOGLE_API_KEY, CHAT_MODEL

    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
    )
    return response.text
