"""Tests for the text chunker."""

import pytest
from src.chunker import chunk_text


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_basic_chunking(self):
        """Text should be split into chunks with metadata."""
        text = "Hello world. " * 200  # ~2600 chars
        metadata = {"source": "test.txt", "type": "txt"}

        chunks = chunk_text(text, metadata, chunk_size=1000, chunk_overlap=200)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "id" in chunk
            assert len(chunk["text"]) <= 1000 + 50  # allow small overflow from splitter

    def test_metadata_preserved(self):
        """Each chunk should carry the base metadata plus chunk_index."""
        text = "Some content. " * 100
        metadata = {"source": "report.pdf", "type": "pdf", "num_pages": 5}

        chunks = chunk_text(text, metadata)

        for chunk in chunks:
            assert chunk["metadata"]["source"] == "report.pdf"
            assert chunk["metadata"]["type"] == "pdf"
            assert "chunk_index" in chunk["metadata"]
            assert "total_chunks" in chunk["metadata"]

    def test_chunk_ids_unique(self):
        """Each chunk should have a unique ID."""
        text = "Test content here. " * 200
        metadata = {"source": "doc.txt", "type": "txt"}

        chunks = chunk_text(text, metadata)
        ids = [c["id"] for c in chunks]

        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_chunk_id_format(self):
        """Chunk IDs should follow the source::chunk_N format."""
        text = "Some text. " * 200
        metadata = {"source": "myfile.pdf", "type": "pdf"}

        chunks = chunk_text(text, metadata)

        for i, chunk in enumerate(chunks):
            assert chunk["id"] == f"myfile.pdf::chunk_{i}"

    def test_short_text_single_chunk(self):
        """Short text should produce a single chunk."""
        text = "A short document."
        metadata = {"source": "short.txt", "type": "txt"}

        chunks = chunk_text(text, metadata)

        assert len(chunks) == 1
        assert chunks[0]["text"] == "A short document."
        assert chunks[0]["metadata"]["chunk_index"] == 0
        assert chunks[0]["metadata"]["total_chunks"] == 1

    def test_empty_text_produces_empty_list(self):
        """Empty text should produce no chunks."""
        text = ""
        metadata = {"source": "empty.txt", "type": "txt"}

        chunks = chunk_text(text, metadata)

        assert chunks == []

    def test_custom_chunk_size(self):
        """Custom chunk sizes should be respected."""
        text = "Word " * 500  # ~2500 chars
        metadata = {"source": "test.txt", "type": "txt"}

        small_chunks = chunk_text(text, metadata, chunk_size=500, chunk_overlap=50)
        large_chunks = chunk_text(text, metadata, chunk_size=2000, chunk_overlap=200)

        assert len(small_chunks) > len(large_chunks)

    def test_chunk_index_sequential(self):
        """Chunk indices should be sequential starting from 0."""
        text = "Content here. " * 200
        metadata = {"source": "test.txt", "type": "txt"}

        chunks = chunk_text(text, metadata)
        indices = [c["metadata"]["chunk_index"] for c in chunks]

        assert indices == list(range(len(chunks)))
