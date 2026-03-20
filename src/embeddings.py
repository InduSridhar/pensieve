"""
Gemini embedding wrapper.
Wraps the google-genai SDK to embed text into 768-dim vectors.
"""

from __future__ import annotations

from google import genai

from src.config import GOOGLE_API_KEY, EMBEDDING_MODEL

# ── Client ───────────────────────────────────────────────────────────

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily initialise and return the Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


# ── Public API ───────────────────────────────────────────────────────

def embed_text(text: str) -> list[float]:
    """
    Embed a single text string using Gemini's embedding model.

    Args:
        text: The text to embed.

    Returns:
        A list of floats (768-dimensional vector).
    """
    client = _get_client()
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return result.embeddings[0].values


def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """
    Embed a list of texts in batches.

    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts per API call.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
        )
        all_embeddings.extend([e.values for e in result.embeddings])

    return all_embeddings
