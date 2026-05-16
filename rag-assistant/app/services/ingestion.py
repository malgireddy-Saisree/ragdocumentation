"""
Document ingestion pipeline.

Responsibilities
----------------
1. Load raw text from local files (Markdown / plain-text) or remote URLs.
2. Split into semantically reasonable chunks that fit inside the LLM context
   without losing too much surrounding information.
3. Hand the chunks to the vector store for embedding + storage.

Chunking strategy
-----------------
We use LangChain's RecursiveCharacterTextSplitter because technical docs tend
to have a mix of prose, code blocks, and lists.  The recursive splitter tries
larger separators first (double newline → single newline → space → char) so it
prefers splitting on paragraph boundaries rather than mid-sentence.

chunk_size  = 800 tokens-equivalent characters  — large enough to hold a full
              code example but not so large that retrieval returns walls of text.
chunk_overlap = 150 chars — keeps a bit of the previous chunk's tail so that
              sentences that straddle a boundary are still present in at least
              one chunk.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from app.config import get_settings
from app.services import vector_store as vs


def _fetch_url(url: str) -> str:
    """Download a URL and return clean text (strips HTML tags if needed)."""
    headers = {"User-Agent": "RAG-Doc-Fetcher/1.0"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type:
        soup = BeautifulSoup(resp.text, "html.parser")
        # Remove noisy elements
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    return resp.text


def _load_file(path: str | Path) -> str:
    """Read a local text / Markdown file."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def _clean(text: str) -> str:
    """Light normalisation: collapse excessive blank lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _make_chunks(
    text: str,
    source_label: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    return [
        {"text": chunk, "source": source_label, "chunk_idx": idx}
        for idx, chunk in enumerate(raw_chunks)
    ]


# ── Public functions ──────────────────────────────────────────────────────────

def ingest_urls(
    urls: list[str],
    collection_name: Optional[str] = None,
) -> dict:
    """
    Fetch, chunk, and index a list of URLs.
    Returns a dict with counts for the API response.
    """
    cfg = get_settings()
    all_chunks: list[dict] = []
    processed  = 0

    for url in urls:
        try:
            logger.info("Fetching URL: {}", url)
            raw  = _fetch_url(url)
            text = _clean(raw)
            chunks = _make_chunks(
                text,
                source_label=url,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
            all_chunks.extend(chunks)
            processed += 1
            logger.info("  → {} chunks from {}", len(chunks), url)
        except Exception as exc:
            logger.warning("Failed to fetch {}: {}", url, exc)

    stored = vs.add_chunks(all_chunks, collection_name=collection_name)
    return {
        "success": processed > 0,
        "indexed_chunks": stored,
        "documents_processed": processed,
    }


def ingest_file_bytes(
    content: bytes,
    filename: str,
    collection_name: Optional[str] = None,
) -> dict:
    """
    Chunk and index a file uploaded via the API.
    Handles .txt and .md; HTML content is stripped automatically.
    """
    cfg = get_settings()
    text = content.decode("utf-8", errors="replace")

    if filename.endswith(".html") or filename.endswith(".htm"):
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)

    text   = _clean(text)
    chunks = _make_chunks(
        text,
        source_label=filename,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    stored = vs.add_chunks(chunks, collection_name=collection_name)
    return {
        "success": True,
        "indexed_chunks": stored,
        "documents_processed": 1,
    }
