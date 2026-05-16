"""
ui/streamlit_app.py

Streamlit frontend for the RAG Documentation Assistant.

Run with:
    streamlit run ui/streamlit_app.py

The FastAPI backend must be running separately:
    uvicorn main:app --reload
"""

import requests
import streamlit as st
from pathlib import Path

API_BASE = "http://localhost:8000"
TIMEOUT  = 60   # seconds


# ── Helpers ────────────────────────────────────────────────────────────────────

def _api_get(path: str):
    """
    GET request. Returns (data_dict | None, error_str | None).
    Never raises — all connection problems come back as an error string.
    """
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=TIMEOUT)
        if resp.ok:
            return resp.json(), None
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return None, f"API returned {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, (
            "Cannot reach the API server at `localhost:8000`.\n\n"
            "**Start it first in a separate terminal:**\n"
            "```\nuvicorn main:app --reload\n```"
        )
    except requests.exceptions.Timeout:
        return None, "Request timed out. The server may be overloaded."
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


def _api_post(path: str, *, json=None, files=None):
    """
    POST request. Returns (data_dict | None, error_str | None).
    Never raises.
    """
    try:
        resp = requests.post(
            f"{API_BASE}{path}",
            json=json,
            files=files,
            timeout=TIMEOUT,
        )
        if resp.ok:
            return resp.json(), None
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return None, f"API returned {resp.status_code}: {detail}"
    except requests.exceptions.ConnectionError:
        return None, (
            "Cannot reach the API server at `localhost:8000`.\n\n"
            "**Start it first in a separate terminal:**\n"
            "```\nuvicorn main:app --reload\n```"
        )
    except requests.exceptions.Timeout:
        return None, "Request timed out — the model may be taking too long."
    except Exception as exc:
        return None, f"Unexpected error: {exc}"


# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Doc Assistant",
    page_icon="📚",
    layout="wide",
)

# ── Session state defaults ─────────────────────────────────────────────────────

for _key, _default in {
    "messages":      [],
    "last_answer":   None,
    "last_question": None,
    "backend_ok":    None,   # None = never checked
}.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 RAG Doc Assistant")

    # ── Backend status ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔌 Backend status")

    if st.button("Check connection"):
        with st.spinner("Pinging server…"):
            data, err = _api_get("/health")
        if err:
            st.session_state.backend_ok = False
        else:
            st.session_state.backend_ok = True

    if st.session_state.backend_ok is True:
        st.success("Connected ✅")
    elif st.session_state.backend_ok is False:
        st.error(
            "Offline ❌\n\n"
            "Run this in a separate terminal:\n"
            "```\nuvicorn main:app --reload\n```"
        )
    else:
        st.info("Click **Check connection** to verify the server is running.")

    # ── Settings ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("⚙️ Settings")
    top_k      = st.slider("Top-K retrieval", min_value=1, max_value=10, value=5)
    session_id = st.text_input("Session ID (for memory)", value="default-session")

    # ── Ingest ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📥 Ingest Documents")
    ingest_mode = st.radio("Source", ["Upload file", "URL"], horizontal=True)

    if ingest_mode == "Upload file":
        uploaded = st.file_uploader(
            "Upload .md, .txt, or .html",
            type=["md", "txt", "html", "htm"],
        )
        if st.button("Ingest File", disabled=(uploaded is None)):
            with st.spinner("Indexing…"):
                data, err = _api_post(
                    "/ingest/upload",
                    files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")},
                )
            if err:
                st.error(err)
            else:
                st.success(f"✅ {data['message']}")

    else:
        url_input = st.text_area("Enter URLs (one per line)", height=100)
        if st.button("Ingest URLs", disabled=(not url_input.strip())):
            urls = [u.strip() for u in url_input.strip().splitlines() if u.strip()]
            with st.spinner(f"Fetching {len(urls)} URL(s)…"):
                data, err = _api_post("/ingest", json={"urls": urls})
            if err:
                st.error(err)
            else:
                st.success(f"✅ {data['message']}")

    # ── Indexed documents ───────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📄 Indexed Documents")

    # Explicitly button-gated — never fires automatically on page load
    if st.button("🔄 Refresh list"):
        with st.spinner("Loading…"):
            data, err = _api_get("/documents")
        if err:
            st.error(err)
        else:
            st.metric("Total chunks", data["total_chunks"])
            if data["total_documents"] == 0:
                st.info("No documents indexed yet.")
            else:
                for doc in data["documents"]:
                    with st.expander(Path(doc["source"]).name):
                        st.write(f"**Chunks:** {doc['total_chunks']}")
                        st.caption(doc["sample_text"])


# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("💬 Ask a Question")
st.caption("Powered by Azure OpenAI + LangGraph self-corrective RAG")

# Render previous conversation turns
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{Path(src['source']).name}** — chunk {src['chunk_index']}"
                    )
                    st.code(src["excerpt"], language="text")

# Chat input — only triggers when the user submits
if question := st.chat_input("Ask about your documentation…"):

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            data, err = _api_post(
                "/query",
                json={
                    "question":   question,
                    "session_id": session_id,
                    "top_k":      top_k,
                },
            )

        if err:
            st.error(err)
            st.session_state.messages.append(
                {"role": "assistant", "content": f"_(error)_", "sources": []}
            )

        else:
            answer      = data["answer"]
            sources     = data.get("sources", [])
            retry_count = data.get("retry_count", 0)
            fallback    = data.get("fallback_used", False)
            rewritten   = data.get("rewritten_query", "")

            st.markdown(answer)

            if rewritten and rewritten != question:
                st.caption(f"🔍 Query rewritten to: *{rewritten}*")
            if retry_count > 0:
                st.caption(f"🔄 Retried retrieval {retry_count} time(s).")
            if fallback:
                st.warning(
                    "⚠️ No relevant documents found. "
                    "Try ingesting more docs or rephrasing your question."
                )

            if sources:
                with st.expander("📎 Sources"):
                    for src in sources:
                        st.markdown(
                            f"**{Path(src['source']).name}** — chunk {src['chunk_index']}"
                        )
                        st.code(src["excerpt"], language="text")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
            st.session_state.last_answer   = answer
            st.session_state.last_question = question


# ── Feedback panel ─────────────────────────────────────────────────────────────

if st.session_state.last_answer:
    st.markdown("---")
    st.subheader("Was this answer helpful?")

    feedback_comment = st.text_input(
        "comment", label_visibility="collapsed",
        placeholder="Optional comment…", key="fb_comment"
    )
    col1, col2, _ = st.columns([1, 1, 4])

    with col1:
        if st.button("👍 Yes"):
            _, err = _api_post("/feedback", json={
                "question": st.session_state.last_question,
                "answer":   st.session_state.last_answer,
                "rating":   "thumbs_up",
                "comment":  feedback_comment or None,
            })
            st.error(err) if err else st.success("Thanks! 🙏")

    with col2:
        if st.button("👎 No"):
            _, err = _api_post("/feedback", json={
                "question": st.session_state.last_question,
                "answer":   st.session_state.last_answer,
                "rating":   "thumbs_down",
                "comment":  feedback_comment or None,
            })
            st.error(err) if err else st.info("Feedback recorded.")
