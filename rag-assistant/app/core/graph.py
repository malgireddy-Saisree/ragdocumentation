"""
LangGraph workflow definition.

Graph layout
------------

    [query_analysis]
          │
     [retrieval]
          │
   [document_grading]
          │
    ┌─────┴──────────────────────┐
  relevant              all irrelevant
    │                       │
[generation]      retry_count < MAX?
    │               yes ──── no
  END          [query_rewrite]  [fallback]
                    │               │
               [retrieval]        END
                    │
              [document_grading]
                  ...

The conditional edge after document_grading decides between:
  - "generate"    → at least one relevant chunk found
  - "rewrite"     → no relevant chunks AND retries left
  - "give_up"     → no relevant chunks AND retries exhausted

We compile the graph once at module import time and cache it; every API
request just calls .invoke() on the compiled graph.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, StateGraph

from app.config import get_settings
from app.core.nodes import (
    document_grading_node,
    fallback_node,
    generation_node,
    query_analysis_node,
    query_rewrite_node,
    retrieval_node,
)
from app.models.schemas import GraphState


# ── Routing logic ─────────────────────────────────────────────────────────────

def _route_after_grading(state: GraphState) -> str:
    """
    Decide what to do once document grading has run.

    Returns one of the string node names recognised by the StateGraph.
    """
    cfg = get_settings()

    if not state.all_irrelevant:
        return "generate"

    if state.retry_count < cfg.max_retry_attempts:
        return "rewrite"

    return "give_up"


# ── Graph builder ─────────────────────────────────────────────────────────────

@lru_cache()
def build_graph():
    """
    Build and compile the LangGraph StateGraph.  Cached so the graph object
    is created only once regardless of how many requests hit the server.
    """
    # LangGraph needs to know the TypedDict / Pydantic schema for the state.
    # We pass GraphState directly; LangGraph uses its field annotations.
    workflow = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("query_analysis",    query_analysis_node)
    workflow.add_node("retrieval",         retrieval_node)
    workflow.add_node("document_grading",  document_grading_node)
    workflow.add_node("generation",        generation_node)
    workflow.add_node("query_rewrite",     query_rewrite_node)
    workflow.add_node("fallback",          fallback_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    workflow.set_entry_point("query_analysis")

    # ── Deterministic edges ───────────────────────────────────────────────────
    workflow.add_edge("query_analysis",   "retrieval")
    workflow.add_edge("retrieval",        "document_grading")
    workflow.add_edge("query_rewrite",    "retrieval")      # retry loop back
    workflow.add_edge("generation",       END)
    workflow.add_edge("fallback",         END)

    # ── Conditional edge (the self-corrective part) ───────────────────────────
    workflow.add_conditional_edges(
        "document_grading",
        _route_after_grading,
        {
            "generate": "generation",
            "rewrite":  "query_rewrite",
            "give_up":  "fallback",
        },
    )

    return workflow.compile()


def run_graph(
    question: str,
    session_id: str | None = None,
    top_k: int | None = None,
    chat_history: list[dict] | None = None,
) -> GraphState:
    """
    Execute the compiled graph for a single question and return the final state.
    """
    cfg = get_settings()
    graph = build_graph()

    initial_state = GraphState(
        original_question=question,
        session_id=session_id,
        top_k=top_k or cfg.top_k_retrieval,
        chat_history=chat_history or [],
    )

    # LangGraph returns the final state as a dict; we coerce it back to our model.
    result = graph.invoke(initial_state)
    return GraphState(**result)
