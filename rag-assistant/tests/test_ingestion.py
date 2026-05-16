"""
tests/test_ingestion.py

Tests for the document ingestion pipeline — chunking logic in particular.
"""

from unittest.mock import patch, MagicMock

import pytest

from app.services.ingestion import _clean, _make_chunks


class TestClean:
    def test_collapses_blank_lines(self):
        text   = "Para one\n\n\n\n\nPara two"
        result = _clean(text)
        assert "\n\n\n" not in result
        assert "Para one" in result
        assert "Para two" in result

    def test_strips_leading_trailing_whitespace(self):
        result = _clean("   hello world   ")
        assert result == "hello world"


class TestMakeChunks:
    def test_produces_chunks(self):
        long_text = "word " * 600   # ~3000 chars → several chunks at size=800
        chunks    = _make_chunks(long_text, source_label="test.md", chunk_size=800, chunk_overlap=150)
        assert len(chunks) > 1

    def test_chunk_metadata(self):
        text   = "Short document."
        chunks = _make_chunks(text, source_label="my_file.md", chunk_size=800, chunk_overlap=50)
        assert chunks[0]["source"] == "my_file.md"
        assert chunks[0]["chunk_idx"] == 0

    def test_overlap_means_no_info_loss(self):
        # A word near a chunk boundary should appear in at least one chunk
        repeated_word = "BOUNDARY_MARKER"
        text = ("padding " * 100) + repeated_word + (" padding" * 100)
        chunks = _make_chunks(text, source_label="x", chunk_size=400, chunk_overlap=100)
        found = any(repeated_word in c["text"] for c in chunks)
        assert found, "Boundary word should appear in at least one chunk"


class TestIngestFileBytesIntegration:
    @patch("app.services.ingestion.vs.add_chunks", return_value=5)
    def test_markdown_file_ingested(self, mock_add):
        from app.services.ingestion import ingest_file_bytes
        content = b"# Hello\n\nThis is a markdown document with enough content to chunk."
        result  = ingest_file_bytes(content, filename="test.md")
        assert result["success"] is True
        assert result["documents_processed"] == 1
        assert result["indexed_chunks"] == 5

    @patch("app.services.ingestion.vs.add_chunks", return_value=3)
    def test_html_tags_stripped(self, mock_add):
        from app.services.ingestion import ingest_file_bytes
        html = b"<html><body><h1>Title</h1><p>Content</p><script>bad()</script></body></html>"
        result = ingest_file_bytes(html, filename="page.html")
        # Verify add_chunks was called (stripping happened without errors)
        mock_add.assert_called_once()
        chunks_arg = mock_add.call_args[0][0]
        for chunk in chunks_arg:
            assert "<script>" not in chunk["text"]
