"""
tests/test_api.py

Integration-style tests for the FastAPI endpoints.
The LLM and vector store are mocked so no live credentials are needed.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    # Patch settings so the app starts without a real .env
    with patch("app.config.Settings._env_file", None):
        with patch.dict(
            "os.environ",
            {
                "AZURE_OPENAI_API_KEY":           "fake-key",
                "AZURE_OPENAI_ENDPOINT":          "https://fake.openai.azure.com/",
                "AZURE_OPENAI_API_VERSION":       "2024-02-01",
                "AZURE_OPENAI_CHAT_DEPLOYMENT":   "gpt-4o-mini",
                "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-ada-002",
            },
        ):
            from main import app
            yield TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── /documents ────────────────────────────────────────────────────────────────

@patch("app.api.routes.vs.list_documents", return_value=[])
@patch("app.api.routes.vs.get_total_chunks", return_value=0)
def test_list_documents_empty(mock_count, mock_list, client):
    resp = client.get("/documents")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_chunks"] == 0
    assert data["documents"] == []


# ── /query ────────────────────────────────────────────────────────────────────

@patch("app.api.routes.vs.get_total_chunks", return_value=0)
def test_query_without_docs_returns_422(mock_count, client):
    resp = client.post("/query", json={"question": "What is LangGraph?"})
    assert resp.status_code == 422


@patch("app.api.routes.run_graph")
@patch("app.api.routes.vs.get_total_chunks", return_value=10)
def test_query_success(mock_count, mock_run, client):
    from app.models.schemas import GraphState, QueryType, SourceDocument

    mock_run.return_value = GraphState(
        original_question="What is LangGraph?",
        answer="LangGraph is a graph-based agent framework.",
        sources=[
            SourceDocument(source="langgraph.md", chunk_index=0, excerpt="LangGraph is…")
        ],
        query_type=QueryType.conceptual,
        rewritten_query="Explain LangGraph",
        retry_count=0,
        fallback_used=False,
    )

    resp = client.post("/query", json={"question": "What is LangGraph?"})
    assert resp.status_code == 200
    data = resp.json()
    assert "LangGraph" in data["answer"]
    assert len(data["sources"]) == 1


# ── /ingest ───────────────────────────────────────────────────────────────────

@patch("app.api.routes.ingestion.ingest_urls")
def test_ingest_urls(mock_ingest, client):
    mock_ingest.return_value = {
        "success": True,
        "indexed_chunks": 42,
        "documents_processed": 2,
    }
    resp = client.post(
        "/ingest",
        json={"urls": ["https://example.com/doc1", "https://example.com/doc2"]},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["indexed_chunks"] == 42
    assert data["documents_processed"] == 2


# ── /feedback ─────────────────────────────────────────────────────────────────

@patch("app.api.routes.fb_svc.record_feedback", return_value=True)
def test_feedback_thumbs_up(mock_record, client):
    resp = client.post("/feedback", json={
        "question": "What is FastAPI?",
        "answer":   "FastAPI is a web framework.",
        "rating":   "thumbs_up",
        "comment":  "Very helpful!",
    })
    assert resp.status_code == 200
    assert resp.json()["recorded"] is True


@patch("app.api.routes.fb_svc.record_feedback", return_value=True)
def test_feedback_thumbs_down(mock_record, client):
    resp = client.post("/feedback", json={
        "question": "What is FastAPI?",
        "answer":   "FastAPI is a web framework.",
        "rating":   "thumbs_down",
    })
    assert resp.status_code == 200
