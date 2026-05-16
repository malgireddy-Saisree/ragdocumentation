"""
Pydantic schemas for every API request and response in the system.
Keeping these in one file makes it easy to track the data contracts.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, HttpUrl


# ── Enums ─────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    conceptual   = "conceptual"
    how_to       = "how_to"
    troubleshoot = "troubleshoot"
    api_ref      = "api_ref"
    unknown      = "unknown"


class GradeLabel(str, Enum):
    relevant   = "relevant"
    irrelevant = "irrelevant"


class FeedbackType(str, Enum):
    thumbs_up   = "thumbs_up"
    thumbs_down = "thumbs_down"


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    urls: list[str] = Field(..., min_length=1, description="List of URLs to fetch and index")
    collection_name: Optional[str] = Field(None, description="Override the default collection")


class IngestResponse(BaseModel):
    success: bool
    indexed_chunks: int
    documents_processed: int
    message: str


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural-language question to answer")
    session_id: Optional[str] = Field(None, description="Session ID for conversation memory")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Override default top-k retrieval")


class SourceDocument(BaseModel):
    source: str
    chunk_index: int
    excerpt: str          # first ~200 chars of the chunk


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceDocument]
    query_type: QueryType
    rewritten_query: Optional[str] = None
    retry_count: int = 0
    fallback_used: bool = False


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    source: str
    total_chunks: int
    sample_text: str


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list[DocumentInfo]


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: FeedbackType
    comment: Optional[str] = Field(None, max_length=1000)


class FeedbackResponse(BaseModel):
    recorded: bool
    message: str


# ── Internal graph state ──────────────────────────────────────────────────────

class GraphState(BaseModel):
    """
    Shared state object that flows through every node in the LangGraph workflow.
    Each node reads what it needs and writes back its outputs.
    """
    # Input
    original_question: str
    session_id: Optional[str] = None
    top_k: int = 5

    # After query analysis
    rewritten_query: str = ""
    query_type: QueryType = QueryType.unknown

    # After retrieval
    retrieved_docs: list[dict[str, Any]] = Field(default_factory=list)

    # After grading
    relevant_docs: list[dict[str, Any]] = Field(default_factory=list)
    all_irrelevant: bool = False

    # Routing / retry bookkeeping
    retry_count: int = 0
    fallback_used: bool = False

    # Final output
    answer: str = ""
    sources: list[SourceDocument] = Field(default_factory=list)

    # Conversation history (optional memory)
    chat_history: list[dict[str, str]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True
