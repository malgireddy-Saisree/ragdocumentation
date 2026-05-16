"""
LangGraph node implementations.

Each node is a plain Python function that accepts the current GraphState and
returns a *partial* dict of fields to update.  LangGraph merges the returned
dict into the state automatically.

Node order
----------
  query_analysis  →  retrieval  →  document_grading
                                         │
                        ┌────────────────┴───────────────┐
                    relevant                       all irrelevant
                        │                               │
                   generation                   (retry or give up)
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from app.config import get_settings
from app.core.llm import get_llm
from app.models.schemas import GradeLabel, GraphState, QueryType, SourceDocument
from app.services import vector_store as vs


# ── Node 1: Query Analysis ────────────────────────────────────────────────────

_ANALYSIS_SYSTEM = """You are a query analysis assistant for a technical documentation chatbot.
Given the user's raw question you must do two things and reply ONLY with valid JSON — no markdown fences:

{
  "rewritten_query": "<improved version of the question, adding synonyms or clarifying intent>",
  "query_type": "<one of: conceptual | how_to | troubleshoot | api_ref | unknown>"
}

Rules:
- Expand abbreviations (e.g. "LLM" → "large language model").
- For how-to questions, rephrase so the key action verb appears near the start.
- Keep the rewrite concise — do NOT add fabricated details."""


def query_analysis_node(state: GraphState) -> dict[str, Any]:
    """
    Rewrite the user's question for better retrieval and classify its type.
    """
    llm  = get_llm(temperature=0.0)
    msgs = [
        SystemMessage(content=_ANALYSIS_SYSTEM),
        HumanMessage(content=state.original_question),
    ]
    response = llm.invoke(msgs)

    try:
        parsed = json.loads(response.content)
        rewritten  = parsed.get("rewritten_query", state.original_question)
        q_type_raw = parsed.get("query_type", "unknown")
        q_type     = QueryType(q_type_raw) if q_type_raw in QueryType.__members__ else QueryType.unknown
    except (json.JSONDecodeError, ValueError):
        logger.warning("query_analysis: failed to parse LLM JSON, using original question")
        rewritten = state.original_question
        q_type    = QueryType.unknown

    logger.info("Query analysis → type={} rewrite='{:.80}'", q_type, rewritten)
    return {"rewritten_query": rewritten, "query_type": q_type}


# ── Node 2: Retrieval ─────────────────────────────────────────────────────────

def retrieval_node(state: GraphState) -> dict[str, Any]:
    """
    Hit the vector store with the (possibly rewritten) query.
    """
    query = state.rewritten_query or state.original_question
    docs  = vs.similarity_search(query, top_k=state.top_k)
    logger.info("Retrieval: {} docs returned for query '{:.60}'", len(docs), query)
    return {"retrieved_docs": docs}


# ── Node 3: Document Grading ──────────────────────────────────────────────────

_GRADE_SYSTEM = """You are a relevance grader for a RAG system.
Given a question and a retrieved document chunk, decide whether the chunk
is useful for answering the question.

Reply ONLY with valid JSON (no markdown):
{"label": "relevant"}   or   {"label": "irrelevant"}

Be strict: if the chunk is about a completely different topic, label it irrelevant."""


def document_grading_node(state: GraphState) -> dict[str, Any]:
    """
    Grade each retrieved chunk as relevant or irrelevant and filter the list.
    """
    llm      = get_llm(temperature=0.0)
    question = state.original_question
    relevant = []

    for doc in state.retrieved_docs:
        prompt = f"Question: {question}\n\nChunk:\n{doc['text'][:1200]}"
        msgs   = [SystemMessage(content=_GRADE_SYSTEM), HumanMessage(content=prompt)]
        resp   = llm.invoke(msgs)

        try:
            label = json.loads(resp.content).get("label", "irrelevant")
        except (json.JSONDecodeError, AttributeError):
            label = "irrelevant"

        if label == GradeLabel.relevant:
            relevant.append(doc)
        else:
            logger.debug("Grader: chunk from '{}' marked irrelevant", doc.get("source"))

    all_irrelevant = len(relevant) == 0
    logger.info(
        "Grading: {}/{} chunks relevant (all_irrelevant={})",
        len(relevant), len(state.retrieved_docs), all_irrelevant,
    )
    return {"relevant_docs": relevant, "all_irrelevant": all_irrelevant}


# ── Node 4: Generation ────────────────────────────────────────────────────────

_GENERATION_SYSTEM = """You are a helpful technical documentation assistant.
Use ONLY the provided context to answer the question.
At the end of your answer cite which sources you used, like [source: <filename or URL>].
If the context doesn't contain enough information say so clearly — do not guess."""


def generation_node(state: GraphState) -> dict[str, Any]:
    """
    Generate a grounded answer from the relevant chunks.
    """
    llm      = get_llm(temperature=0.2)
    question = state.original_question

    # Build the context block
    context_parts = []
    for i, doc in enumerate(state.relevant_docs, 1):
        context_parts.append(f"[{i}] (source: {doc['source']})\n{doc['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Optionally prepend chat history (conversation memory)
    history_block = ""
    if state.chat_history:
        lines = []
        for turn in state.chat_history[-6:]:  # last 3 exchanges
            lines.append(f"User: {turn['user']}")
            lines.append(f"Assistant: {turn['assistant']}")
        history_block = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

    user_prompt = (
        f"{history_block}"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    msgs = [
        SystemMessage(content=_GENERATION_SYSTEM),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(msgs)
    answer   = response.content.strip()

    # Build structured source list for the API response
    sources = [
        SourceDocument(
            source=doc["source"],
            chunk_index=doc["chunk_idx"],
            excerpt=doc["text"][:200],
        )
        for doc in state.relevant_docs
    ]

    logger.info("Generation complete ({} chars, {} sources)", len(answer), len(sources))
    return {"answer": answer, "sources": sources}


# ── Fallback node (no relevant docs) ─────────────────────────────────────────

def fallback_node(state: GraphState) -> dict[str, Any]:
    """
    Called when all retrieved chunks were graded irrelevant and we've
    exhausted retries.  Returns a polite "I don't know" answer.
    """
    answer = (
        "I wasn't able to find information in the indexed documents that "
        "directly answers your question.  Please try rephrasing your query, "
        "or make sure the relevant documentation has been ingested."
    )
    logger.info("Fallback node triggered after {} retries", state.retry_count)
    return {"answer": answer, "sources": [], "fallback_used": True}


# ── Query rewrite node (for retry loop) ───────────────────────────────────────

_REWRITE_SYSTEM = """The previous retrieval attempt returned no relevant results.
Rewrite the query below in a different way to improve the chance of finding
relevant documentation.  Reply with ONLY the rewritten query — no explanation."""


def query_rewrite_node(state: GraphState) -> dict[str, Any]:
    """
    Alternative to fallback: rewrite the query and increment the retry counter.
    """
    llm  = get_llm(temperature=0.3)
    msgs = [
        SystemMessage(content=_REWRITE_SYSTEM),
        HumanMessage(content=state.rewritten_query or state.original_question),
    ]
    response  = llm.invoke(msgs)
    new_query = response.content.strip()

    logger.info(
        "Query rewrite (attempt {}): '{:.80}'",
        state.retry_count + 1, new_query,
    )
    return {
        "rewritten_query": new_query,
        "retry_count":     state.retry_count + 1,
        # Reset so the next grading pass starts fresh
        "retrieved_docs":  [],
        "relevant_docs":   [],
        "all_irrelevant":  False,
    }
