"""Tests for the RAG pipeline and chat module (with mocked Gemini & ChromaDB)."""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

from src.chat import build_rag_prompt, build_summary_prompt


class TestBuildRagPrompt:
    """Tests for RAG prompt construction."""

    def test_basic_prompt(self):
        """Prompt should include the question and context."""
        question = "What is machine learning?"
        chunks = [
            {"text": "ML is a subset of AI.", "source": "ml_guide.pdf", "chunk_index": 0, "distance": 0.1},
        ]

        prompt = build_rag_prompt(question, chunks)

        assert "What is machine learning?" in prompt
        assert "ML is a subset of AI." in prompt
        assert "ml_guide.pdf" in prompt

    def test_multiple_chunks(self):
        """All context chunks should appear in the prompt."""
        chunks = [
            {"text": "Chunk one content.", "source": "doc1.pdf", "chunk_index": 0, "distance": 0.1},
            {"text": "Chunk two content.", "source": "doc2.txt", "chunk_index": 1, "distance": 0.2},
        ]

        prompt = build_rag_prompt("test question", chunks)

        assert "Chunk one content." in prompt
        assert "Chunk two content." in prompt

    def test_chat_history_included(self):
        """Chat history should be included when provided."""
        question = "Tell me more"
        chunks = [{"text": "Some text.", "source": "doc.txt", "chunk_index": 0, "distance": 0.1}]
        history = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is artificial intelligence."},
        ]

        prompt = build_rag_prompt(question, chunks, history)

        assert "What is AI?" in prompt
        assert "AI is artificial intelligence." in prompt

    def test_system_prompt_present(self):
        """The system prompt with grounding rules should be included."""
        prompt = build_rag_prompt("test", [])

        assert "Pensieve" in prompt
        assert "context" in prompt.lower()

    def test_empty_context(self):
        """Prompt should still work with no context chunks."""
        prompt = build_rag_prompt("test question", [])

        assert "test question" in prompt


class TestBuildSummaryPrompt:
    """Tests for the summary prompt builder."""

    def test_source_name_included(self):
        prompt = build_summary_prompt("Some content here.", "report.pdf")

        assert "report.pdf" in prompt
        assert "Some content here." in prompt

    def test_summary_instructions(self):
        prompt = build_summary_prompt("Content", "test.txt")

        assert "summary" in prompt.lower()
        assert "key points" in prompt.lower()


class TestRagPipeline:
    """Integration-style tests for the RAG pipeline (with mocks)."""

    @patch("src.rag_pipeline._generate")
    @patch("src.rag_pipeline.vector_store")
    @patch("src.rag_pipeline.embed_text")
    def test_query_returns_answer(self, mock_embed, mock_vs, mock_generate):
        """query() should return an answer with sources."""
        from src.rag_pipeline import query

        mock_embed.return_value = [0.1] * 768
        mock_vs.query.return_value = {
            "documents": [["Chunk text about AI."]],
            "metadatas": [[{"source": "doc.pdf", "chunk_index": 0}]],
            "distances": [[0.15]],
        }
        mock_generate.return_value = "AI is artificial intelligence."

        result = query("What is AI?")

        assert result["answer"] == "AI is artificial intelligence."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["source"] == "doc.pdf"

    @patch("src.rag_pipeline._generate")
    @patch("src.rag_pipeline.vector_store")
    @patch("src.rag_pipeline.embed_batch")
    @patch("src.document_loader.PdfReader")
    def test_ingest_pdf_flow(self, mock_reader_class, mock_embed_batch, mock_vs, mock_generate):
        """ingest_pdf() should chunk, embed, store, and summarize."""
        from src.rag_pipeline import ingest_pdf

        # Mock PDF reading
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Test content. " * 100
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_class.return_value = mock_reader

        # Mock embeddings - return correct number of embeddings
        mock_embed_batch.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

        # Mock vector store
        mock_vs.add_documents.return_value = None
        mock_vs.get_source_chunks.return_value = [
            {"text": "Test content.", "metadata": {"source": "test.pdf", "chunk_index": 0}},
        ]

        # Mock generation
        mock_generate.return_value = "This is a summary of test content."

        file = BytesIO(b"fake pdf")
        result = ingest_pdf(file, "test.pdf")

        assert result["source"] == "test.pdf"
        assert result["num_chunks"] > 0
        assert result["summary"] == "This is a summary of test content."
        mock_vs.add_documents.assert_called_once()
