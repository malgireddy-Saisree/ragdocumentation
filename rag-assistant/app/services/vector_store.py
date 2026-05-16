"""
Vector store service wrapping ChromaDB.

Design decisions
----------------
* One ChromaDB collection per logical corpus (configurable via env).
* Embeddings are generated via Azure OpenAI's text-embedding-ada-002 so that
  the same model family is used end-to-end — avoids mismatch between indexing
  and query-time embeddings.
* The client is kept as a module-level singleton so that the HTTP connection
  pool is reused across requests.
"""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import AzureOpenAIEmbeddings
from loguru import logger

from app.config import get_settings


_chroma_client: chromadb.ClientAPI | None = None
_embedder: AzureOpenAIEmbeddings | None = None


def _get_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        cfg = get_settings()
        _chroma_client = chromadb.PersistentClient(
            path=cfg.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info("ChromaDB client initialised (persist_dir={})", cfg.chroma_persist_dir)
    return _chroma_client


def _get_embedder() -> AzureOpenAIEmbeddings:
    global _embedder
    if _embedder is None:
        cfg = get_settings()
        _embedder = AzureOpenAIEmbeddings(
            azure_deployment=cfg.azure_openai_embedding_deployment,
            azure_endpoint=cfg.azure_openai_endpoint,
            api_key=cfg.azure_openai_api_key,
            api_version=cfg.azure_openai_api_version,
        )
        logger.info(
            "Azure embedding model ready (deployment={})",
            cfg.azure_openai_embedding_deployment,
        )
    return _embedder


def _get_collection(name: str | None = None) -> chromadb.Collection:
    cfg = get_settings()
    col_name = name or cfg.collection_name
    client = _get_client()
    return client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )


# ── Public API ────────────────────────────────────────────────────────────────

def add_chunks(
    chunks: list[dict[str, Any]],
    collection_name: str | None = None,
) -> int:
    """
    Embed and store a list of chunk dicts.

    Expected chunk format::

        {
            "text":     "...",
            "source":   "path/to/file.md",
            "chunk_idx": 3,
        }

    Returns the number of chunks successfully stored.
    """
    if not chunks:
        return 0

    texts     = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_idx": c["chunk_idx"]} for c in chunks]
    ids       = [str(uuid.uuid4()) for _ in chunks]

    embedder   = _get_embedder()
    embeddings = embedder.embed_documents(texts)

    collection = _get_collection(collection_name)
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    logger.info("Stored {} chunks into collection '{}'", len(chunks), collection.name)
    return len(chunks)


def similarity_search(
    query: str,
    top_k: int = 5,
    collection_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Embed the query and return the top-k most similar chunks.

    Each result dict contains: text, source, chunk_idx, distance.
    """
    embedder    = _get_embedder()
    query_vec   = embedder.embed_query(query)
    collection  = _get_collection(collection_name)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":      doc,
            "source":    meta.get("source", "unknown"),
            "chunk_idx": meta.get("chunk_idx", 0),
            "distance":  round(dist, 4),
        })

    logger.debug("Similarity search returned {} hits for query: '{:.60}'", len(hits), query)
    return hits


def list_documents(collection_name: str | None = None) -> list[dict[str, Any]]:
    """
    Return a summary of every unique source document in the collection.
    """
    collection = _get_collection(collection_name)
    total      = collection.count()

    if total == 0:
        return []

    # Fetch all entries (fine for small corpora; a paginated version would be
    # needed for millions of chunks)
    result = collection.get(include=["documents", "metadatas"])

    source_map: dict[str, list] = {}
    for doc, meta in zip(result["documents"], result["metadatas"]):
        src = meta.get("source", "unknown")
        source_map.setdefault(src, []).append(doc)

    summary = []
    for src, texts in source_map.items():
        summary.append({
            "source":       src,
            "total_chunks": len(texts),
            "sample_text":  texts[0][:200] + "…" if len(texts[0]) > 200 else texts[0],
        })

    return summary


def get_total_chunks(collection_name: str | None = None) -> int:
    return _get_collection(collection_name).count()
