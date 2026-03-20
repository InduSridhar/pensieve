"""Tests for the document loaders."""

import pytest
from io import BytesIO
from unittest.mock import patch, MagicMock

from src.document_loader import load_pdf, load_txt, load_youtube, _extract_video_id


class TestExtractVideoId:
    """Tests for YouTube video ID extraction."""

    def test_standard_url(self):
        assert _extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_short_url(self):
        assert _extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url(self):
        assert _extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        assert _extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self):
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120&list=PLtest"
        assert _extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        with pytest.raises(ValueError, match="Could not extract video ID"):
            _extract_video_id("https://www.google.com")


class TestLoadTxt:
    """Tests for the TXT loader."""

    def test_basic_load(self):
        content = b"Hello, this is a test document."
        file = BytesIO(content)

        result = load_txt(file, "test.txt")

        assert result["text"] == "Hello, this is a test document."
        assert result["metadata"]["source"] == "test.txt"
        assert result["metadata"]["type"] == "txt"

    def test_empty_file_raises(self):
        file = BytesIO(b"")
        with pytest.raises(ValueError, match="empty"):
            load_txt(file, "empty.txt")

    def test_whitespace_only_raises(self):
        file = BytesIO(b"   \n\n  ")
        with pytest.raises(ValueError, match="empty"):
            load_txt(file, "whitespace.txt")

    def test_utf8_content(self):
        content = "Héllo wörld — café résumé".encode("utf-8")
        file = BytesIO(content)

        result = load_txt(file, "unicode.txt")
        assert "café" in result["text"]


class TestLoadPdf:
    """Tests for the PDF loader (using mocked PyPDF2)."""

    @patch("src.document_loader.PdfReader")
    def test_basic_pdf(self, mock_reader_class):
        """A PDF with text should load correctly."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Page 1 content here."

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_class.return_value = mock_reader

        file = BytesIO(b"fake pdf bytes")
        result = load_pdf(file, "test.pdf")

        assert "Page 1 content" in result["text"]
        assert result["metadata"]["source"] == "test.pdf"
        assert result["metadata"]["type"] == "pdf"
        assert result["metadata"]["num_pages"] == 1

    @patch("src.document_loader.PdfReader")
    def test_empty_pdf_raises(self, mock_reader_class):
        """A PDF with no extractable text should raise ValueError."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""

        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        mock_reader_class.return_value = mock_reader

        file = BytesIO(b"fake pdf bytes")
        with pytest.raises(ValueError, match="No extractable text"):
            load_pdf(file, "empty.pdf")


class TestLoadYoutube:
    """Tests for the YouTube loader (using mocked youtube-transcript-api)."""

    @patch("src.document_loader.YouTubeTranscriptApi")
    def test_basic_transcript(self, mock_api_class):
        """A video with captions should load correctly."""
        # Create mock snippets with .text attribute
        snippet1 = MagicMock()
        snippet1.text = "Hello everyone"
        snippet2 = MagicMock()
        snippet2.text = "welcome to the video"

        mock_transcript = MagicMock()
        mock_transcript.__iter__ = MagicMock(return_value=iter([snippet1, snippet2]))

        mock_api = MagicMock()
        mock_api.fetch.return_value = mock_transcript
        mock_api_class.return_value = mock_api

        result = load_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

        assert "Hello everyone" in result["text"]
        assert "welcome to the video" in result["text"]
        assert result["metadata"]["type"] == "youtube"
        assert result["metadata"]["video_id"] == "dQw4w9WgXcQ"

    @patch("src.document_loader.YouTubeTranscriptApi")
    def test_no_captions_raises(self, mock_api_class):
        """A video without captions should raise ValueError."""
        mock_api = MagicMock()
        mock_api.fetch.side_effect = Exception("No transcript available")
        mock_api_class.return_value = mock_api

        with pytest.raises(ValueError, match="Could not fetch transcript"):
            load_youtube("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
