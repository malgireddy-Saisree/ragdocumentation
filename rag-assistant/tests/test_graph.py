"""
tests/test_graph.py

Unit tests for the LangGraph workflow nodes.  We mock the LLM and vector
store so these tests run without any live Azure credentials.
"""

from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import GraphState, QueryType


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_state(**kwargs) -> GraphState:
    defaults = {
        "original_question": "What is LangGraph?",
        "top_k": 3,
    }
    defaults.update(kwargs)
    return GraphState(**defaults)


# ── query_analysis_node ────────────────────────────────────────────────────────

class TestQueryAnalysisNode:
    @patch("app.core.nodes.get_llm")
    def test_returns_rewritten_query_and_type(self, mock_get_llm):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(
            content='{"rewritten_query": "How does LangGraph work?", "query_type": "conceptual"}'
        )
        mock_get_llm.return_value = llm

        from app.core.nodes import query_analysis_node
        result = query_analysis_node(_make_state())

        assert result["rewritten_query"] == "How does LangGraph work?"
        assert result["query_type"] == QueryType.conceptual

    @patch("app.core.nodes.get_llm")
    def test_falls_back_on_bad_json(self, mock_get_llm):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="not json at all")
        mock_get_llm.return_value = llm

        from app.core.nodes import query_analysis_node
        state  = _make_state()
        result = query_analysis_node(state)

        assert result["rewritten_query"] == state.original_question
        assert result["query_type"] == QueryType.unknown


# ── retrieval_node ────────────────────────────────────────────────────────────

class TestRetrievalNode:
    @patch("app.core.nodes.vs")
    def test_calls_similarity_search(self, mock_vs):
        mock_vs.similarity_search.return_value = [
            {"text": "LangGraph is…", "source": "langgraph.md", "chunk_idx": 0, "distance": 0.1}
        ]

        from app.core.nodes import retrieval_node
        state  = _make_state(rewritten_query="How does LangGraph work?")
        result = retrieval_node(state)

        mock_vs.similarity_search.assert_called_once_with("How does LangGraph work?", top_k=3)
        assert len(result["retrieved_docs"]) == 1


# ── document_grading_node ─────────────────────────────────────────────────────

class TestDocumentGradingNode:
    @patch("app.core.nodes.get_llm")
    def test_marks_relevant_chunk(self, mock_get_llm):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"label": "relevant"}')
        mock_get_llm.return_value = llm

        from app.core.nodes import document_grading_node
        docs  = [{"text": "LangGraph enables stateful agents.", "source": "lg.md", "chunk_idx": 0}]
        state = _make_state(retrieved_docs=docs)
        result = document_grading_node(state)

        assert len(result["relevant_docs"]) == 1
        assert result["all_irrelevant"] is False

    @patch("app.core.nodes.get_llm")
    def test_marks_all_irrelevant(self, mock_get_llm):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='{"label": "irrelevant"}')
        mock_get_llm.return_value = llm

        from app.core.nodes import document_grading_node
        docs  = [{"text": "Unrelated topic.", "source": "other.md", "chunk_idx": 0}]
        state = _make_state(retrieved_docs=docs)
        result = document_grading_node(state)

        assert result["relevant_docs"] == []
        assert result["all_irrelevant"] is True


# ── routing logic ─────────────────────────────────────────────────────────────

class TestRouting:
    def test_routes_to_generate_when_relevant(self):
        from app.core.graph import _route_after_grading
        state = _make_state(all_irrelevant=False, retry_count=0)
        assert _route_after_grading(state) == "generate"

    def test_routes_to_rewrite_when_retries_left(self):
        from app.core.graph import _route_after_grading
        state = _make_state(all_irrelevant=True, retry_count=0)
        assert _route_after_grading(state) == "rewrite"

    def test_routes_to_give_up_when_retries_exhausted(self):
        from app.core.graph import _route_after_grading
        state = _make_state(all_irrelevant=True, retry_count=10)
        assert _route_after_grading(state) == "give_up"


# ── generation_node ───────────────────────────────────────────────────────────

class TestGenerationNode:
    @patch("app.core.nodes.get_llm")
    def test_generates_answer_with_sources(self, mock_get_llm):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="LangGraph is a graph-based agent framework.")
        mock_get_llm.return_value = llm

        from app.core.nodes import generation_node
        docs  = [{"text": "LangGraph overview…", "source": "lg.md", "chunk_idx": 0}]
        state = _make_state(relevant_docs=docs)
        result = generation_node(state)

        assert "LangGraph" in result["answer"]
        assert len(result["sources"]) == 1
        assert result["sources"][0].source == "lg.md"


# ── fallback_node ─────────────────────────────────────────────────────────────

class TestFallbackNode:
    def test_returns_fallback_message(self):
        from app.core.nodes import fallback_node
        state  = _make_state(retry_count=2)
        result = fallback_node(state)

        assert "wasn't able to find" in result["answer"]
        assert result["sources"] == []
        assert result["fallback_used"] is True
