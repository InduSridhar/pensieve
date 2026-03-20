"""
Document loaders for PDF, TXT, and YouTube transcripts.
Each loader returns a dict with 'text' (raw string) and 'metadata' (source info).
"""

from __future__ import annotations

import re
from io import BytesIO
from typing import Any

from PyPDF2 import PdfReader
from youtube_transcript_api import YouTubeTranscriptApi, FetchedTranscript


# ── Helpers ──────────────────────────────────────────────────────────

def _extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from various URL formats."""
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/)([A-Za-z0-9_-]{11})",
        r"(?:embed\/)([A-Za-z0-9_-]{11})",
        r"(?:shorts\/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


# ── Loaders ──────────────────────────────────────────────────────────

def load_pdf(file: BytesIO, filename: str = "document.pdf") -> dict[str, Any]:
    """
    Extract text from a PDF file.

    Args:
        file: A file-like object (BytesIO) containing the PDF.
        filename: Display name for the source.

    Returns:
        Dict with 'text' (full extracted text) and 'metadata'.
    """
    reader = PdfReader(file)
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)

    full_text = "\n\n".join(pages)
    if not full_text.strip():
        raise ValueError(f"No extractable text found in PDF: {filename}")

    return {
        "text": full_text,
        "metadata": {
            "source": filename,
            "type": "pdf",
            "num_pages": len(reader.pages),
        },
    }


def load_txt(file: BytesIO, filename: str = "document.txt") -> dict[str, Any]:
    """
    Read text from a plain text file.

    Args:
        file: A file-like object containing text.
        filename: Display name for the source.

    Returns:
        Dict with 'text' and 'metadata'.
    """
    content = file.read()
    if isinstance(content, bytes):
        text = content.decode("utf-8", errors="replace")
    else:
        text = content

    if not text.strip():
        raise ValueError(f"Text file is empty: {filename}")

    return {
        "text": text,
        "metadata": {
            "source": filename,
            "type": "txt",
        },
    }


def load_youtube(url: str) -> dict[str, Any]:
    """
    Fetch the transcript for a YouTube video.

    Args:
        url: A YouTube video URL.

    Returns:
        Dict with 'text' (full transcript) and 'metadata'.
    """
    video_id = _extract_video_id(url)

    try:
        api = YouTubeTranscriptApi()
        transcript: FetchedTranscript = api.fetch(video_id)
    except Exception as exc:
        raise ValueError(
            f"Could not fetch transcript for video {video_id}. "
            "The video may not have captions enabled."
        ) from exc

    # Join all transcript snippets into a single string
    text = " ".join(snippet.text for snippet in transcript)

    if not text.strip():
        raise ValueError(f"Transcript is empty for video: {video_id}")

    return {
        "text": text,
        "metadata": {
            "source": url,
            "type": "youtube",
            "video_id": video_id,
        },
    }
