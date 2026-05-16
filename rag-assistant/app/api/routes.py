"""
FastAPI route handlers.

Every endpoint validates its input via Pydantic, delegates the heavy lifting
to service/core modules, and returns a typed Pydantic response.

Endpoints
---------
POST /query      – run the LangGraph RAG pipeline
POST /ingest     – index files (multipart upload) or URLs (JSON body)
GET  /documents  – list what's currently in the vector store
POST /feedback   – record thumbs-up / thumbs-down on an answer
GET  /health     – lightweight liveness probe
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from loguru import logger

from app.core.graph import run_graph
from app.models.schemas import (
    DocumentListResponse, DocumentInfo,
    FeedbackRequest, FeedbackResponse,
    IngestResponse, IngestURLRequest,
    QueryRequest, QueryResponse, SourceDocument,
)
from app.services import feedback as fb_svc
from app.services import ingestion, vector_store as vs

router = APIRouter()


# ── Health check ──────────────────────────────────────────────────────────────

@router.get("/health", tags=["meta"])
async def health() -> dict:
    return {"status": "ok"}


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question",
    tags=["rag"],
)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """
    Submit a natural-language question.
    The system retrieves relevant documentation chunks, grades them,
    and generates a grounded answer with source citations.
    """
    if vs.get_total_chunks() == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No documents have been indexed yet.  "
                   "Please POST to /ingest first.",
        )

    try:
        final_state = run_graph(
            question=req.question,
            session_id=req.session_id,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("Graph execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {exc}",
        )

    return QueryResponse(
        answer=final_state.answer,
        sources=final_state.sources,
        query_type=final_state.query_type,
        rewritten_query=final_state.rewritten_query or None,
        retry_count=final_state.retry_count,
        fallback_used=final_state.fallback_used,
    )


# ── Ingest – URL list ─────────────────────────────────────────────────────────

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents from URLs or file uploads",
    tags=["ingest"],
)
async def ingest_urls_endpoint(req: IngestURLRequest) -> IngestResponse:
    """
    Accepts a JSON body with a list of URLs to fetch, chunk, embed and index.
    """
    try:
        result = ingestion.ingest_urls(
            urls=req.urls,
            collection_name=req.collection_name,
        )
    except Exception as exc:
        logger.exception("Ingestion (URLs) failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return IngestResponse(
        success=result["success"],
        indexed_chunks=result["indexed_chunks"],
        documents_processed=result["documents_processed"],
        message=(
            f"Indexed {result['indexed_chunks']} chunks from "
            f"{result['documents_processed']} document(s)."
        ),
    )


# ── Ingest – file upload ──────────────────────────────────────────────────────

@router.post(
    "/ingest/upload",
    response_model=IngestResponse,
    summary="Upload a file to ingest",
    tags=["ingest"],
)
async def ingest_file_endpoint(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
) -> IngestResponse:
    """
    Accepts a multipart file upload (.txt, .md, .html).
    The file is chunked and added to the vector store.
    """
    allowed = {".txt", ".md", ".html", ".htm"}
    suffix  = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in allowed:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}",
        )

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:   # 10 MB guard
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File exceeds 10 MB limit.",
        )

    try:
        result = ingestion.ingest_file_bytes(
            content=content,
            filename=file.filename,
            collection_name=collection_name,
        )
    except Exception as exc:
        logger.exception("Ingestion (file upload) failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return IngestResponse(
        success=result["success"],
        indexed_chunks=result["indexed_chunks"],
        documents_processed=result["documents_processed"],
        message=f"Indexed {result['indexed_chunks']} chunks from '{file.filename}'.",
    )


# ── List documents ────────────────────────────────────────────────────────────

@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List indexed documents",
    tags=["ingest"],
)
async def list_documents() -> DocumentListResponse:
    """
    Returns a summary of every unique source document in the vector store,
    including how many chunks each one contributed.
    """
    docs  = vs.list_documents()
    total = vs.get_total_chunks()

    return DocumentListResponse(
        total_documents=len(docs),
        total_chunks=total,
        documents=[
            DocumentInfo(
                source=d["source"],
                total_chunks=d["total_chunks"],
                sample_text=d["sample_text"],
            )
            for d in docs
        ],
    )


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback on an answer",
    tags=["feedback"],
)
async def feedback_endpoint(req: FeedbackRequest) -> FeedbackResponse:
    """
    Record a thumbs-up or thumbs-down rating with an optional comment.
    This data can later be used to evaluate and improve the pipeline.
    """
    ok = fb_svc.record_feedback(req)
    return FeedbackResponse(
        recorded=ok,
        message="Thank you for your feedback!" if ok else "Feedback could not be stored.",
    )
