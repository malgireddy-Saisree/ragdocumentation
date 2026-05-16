#!/usr/bin/env python3
"""
scripts/ingest_sample_docs.py

One-shot script to ingest the bundled sample documentation into the vector
store so the API is ready to answer questions immediately after setup.

Usage
-----
    # From the project root directory:
    python scripts/ingest_sample_docs.py

    # Ingest remote URLs instead:
    python scripts/ingest_sample_docs.py --urls \
        https://docs.langchain.com/... \
        https://langchain-ai.github.io/langgraph/...

Requirements
------------
Copy .env.example → .env and fill in your Azure OpenAI credentials before
running this script.
"""

import argparse
import sys
from pathlib import Path

# Make the project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from loguru import logger
from app.services.ingestion import ingest_file_bytes, ingest_urls


LOCAL_DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

SAMPLE_URLS = [
    # Feel free to replace with your own documentation URLs
    "https://python.langchain.com/docs/introduction/",
    "https://langchain-ai.github.io/langgraph/",
]


def ingest_local_docs(docs_dir: Path) -> None:
    md_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))
    if not md_files:
        logger.warning("No .md or .txt files found in {}", docs_dir)
        return

    total_chunks = 0
    for filepath in md_files:
        logger.info("Ingesting: {}", filepath.name)
        content = filepath.read_bytes()
        result  = ingest_file_bytes(content=content, filename=filepath.name)
        total_chunks += result["indexed_chunks"]
        logger.info(
            "  ✓ {} → {} chunks indexed",
            filepath.name, result["indexed_chunks"]
        )

    logger.info("Local ingestion complete — {} total chunks stored.", total_chunks)


def ingest_remote_urls(urls: list[str]) -> None:
    logger.info("Fetching {} URL(s)...", len(urls))
    result = ingest_urls(urls)
    logger.info(
        "Remote ingestion complete — {} chunks from {} document(s).",
        result["indexed_chunks"], result["documents_processed"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest documentation into the RAG vector store."
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        help="Remote URLs to fetch and index (optional; uses sample URLs if not set).",
    )
    parser.add_argument(
        "--remote-only",
        action="store_true",
        help="Skip local docs and only ingest URLs.",
    )
    args = parser.parse_args()

    if not args.remote_only:
        ingest_local_docs(LOCAL_DOCS_DIR)

    if args.urls or args.remote_only:
        urls = args.urls or SAMPLE_URLS
        ingest_remote_urls(urls)

    logger.info("All done! Start the API server with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
