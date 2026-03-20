"""
Configuration for Pensieve.
Loads environment variables and defines constants used across the app.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── API Key ──────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# ── Model Names ──────────────────────────────────────────────────────
CHAT_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "gemini-embedding-001"

# ── Chunking Settings ────────────────────────────────────────────────
CHUNK_SIZE: int = 1000          # characters per chunk
CHUNK_OVERLAP: int = 200        # overlap between consecutive chunks

# ── Retrieval Settings ───────────────────────────────────────────────
TOP_K: int = 5                  # number of chunks to retrieve per query

# ── Embedding Dimensions ────────────────────────────────────────────
EMBEDDING_DIM: int = 3072

# ── Paths ────────────────────────────────────────────────────────────
CHROMA_DB_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
UPLOADS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")

# ── Validation ───────────────────────────────────────────────────────
def validate_api_key() -> bool:
    """Return True if an API key is configured."""
    return bool(GOOGLE_API_KEY and GOOGLE_API_KEY != "your-gemini-api-key-here")
