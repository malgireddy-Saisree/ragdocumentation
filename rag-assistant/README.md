# RAG Technical Documentation Assistant

A self-corrective Retrieval-Augmented Generation (RAG) system that answers questions about technical documentation. Built with LangGraph, FastAPI, ChromaDB, and Azure OpenAI.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Project Structure](#3-project-structure)
4. [Setup Instructions](#4-setup-instructions)
5. [Running the Application](#5-running-the-application)
6. [API Reference](#6-api-reference)
7. [Example Requests & Responses](#7-example-requests--responses)
8. [Streamlit UI](#8-streamlit-ui)
9. [Running Tests](#9-running-tests)
10. [Design Decisions & Tradeoffs](#10-design-decisions--tradeoffs)
11. [What I Would Improve with More Time](#11-what-i-would-improve-with-more-time)
12. [Assumptions Made](#12-assumptions-made)

---

## 1. Project Overview

This system lets you index any technical documentation (Markdown, plain text, or HTML pages) and then ask natural-language questions against it. A LangGraph workflow handles the full pipeline:

- **Query analysis** — rewrites the user's question to improve retrieval and classifies its type (conceptual / how-to / troubleshooting / API reference).
- **Retrieval** — performs cosine-similarity search in a ChromaDB vector store using Azure OpenAI embeddings.
- **Document grading** — an LLM evaluates each retrieved chunk and filters out off-topic ones. This is the *self-corrective* step.
- **Generation** — the LLM produces a grounded answer with source citations using only the relevant chunks.
- **Retry loop** — if all chunks are graded irrelevant, the query is automatically rewritten and retrieval is retried (up to a configurable limit) before a friendly fallback response is returned.

---

## 2. Architecture

### LangGraph Workflow

```
┌─────────────────┐
│  query_analysis │  ← rewrites query, classifies intent
└────────┬────────┘
         │
┌────────▼────────┐
│    retrieval    │  ← cosine similarity search in ChromaDB
└────────┬────────┘
         │
┌────────▼────────┐
│document_grading │  ← LLM grades each chunk as relevant / irrelevant
└────────┬────────┘
         │
   ┌─────┴──────────────────────────────────────┐
   │ relevant?                                   │ all irrelevant?
   ▼                                             ▼
┌──────────┐                     retries left?       retries exhausted?
│generation│                          │                     │
└──────────┘                  ┌───────▼───────┐    ┌───────▼───────┐
     │                        │ query_rewrite │    │   fallback    │
    END                       └───────┬───────┘    └───────────────┘
                                      │                   END
                               back to retrieval
```

### Component Map

| Layer | Technology | Role |
|---|---|---|
| Workflow orchestration | LangGraph `StateGraph` | Defines nodes, edges, and routing logic |
| LLM | Azure OpenAI (GPT-4o-mini) | Query rewriting, grading, generation |
| Embeddings | Azure OpenAI (text-embedding-ada-002) | Converting text to vectors |
| Vector store | ChromaDB (persistent) | Storing and querying document embeddings |
| API server | FastAPI + Uvicorn | Exposing endpoints to clients |
| Frontend (bonus) | Streamlit | Interactive chat UI |

### State Schema

The `GraphState` Pydantic model is the single source of truth that flows through every node:

```python
class GraphState(BaseModel):
    original_question: str      # raw user input — never mutated
    rewritten_query:   str      # query_analysis output
    query_type:        QueryType
    retrieved_docs:    list     # raw retrieval results
    relevant_docs:     list     # after grading
    all_irrelevant:    bool     # grading verdict
    retry_count:       int      # controls the retry loop
    fallback_used:     bool
    answer:            str      # final answer text
    sources:           list     # structured source citations
    chat_history:      list     # optional conversation memory
```

---

## 3. Project Structure

```
rag-assistant/
├── main.py                        # FastAPI app entry point
├── requirements.txt
├── .env.example                   # copy to .env and fill in credentials
├── .gitignore
├── conftest.py                    # pytest session fixtures
│
├── app/
│   ├── config.py                  # pydantic-settings; all config from env vars
│   │
│   ├── api/
│   │   └── routes.py              # all FastAPI route handlers
│   │
│   ├── core/
│   │   ├── llm.py                 # Azure OpenAI client factory
│   │   ├── nodes.py               # the four LangGraph node functions
│   │   └── graph.py               # StateGraph definition + run_graph()
│   │
│   ├── models/
│   │   └── schemas.py             # Pydantic request/response + GraphState schemas
│   │
│   └── services/
│       ├── vector_store.py        # ChromaDB wrapper (add, search, list)
│       ├── ingestion.py           # document loading, cleaning, chunking
│       └── feedback.py            # feedback recording (JSONL log)
│
├── docs/                          # sample documentation corpus
│   ├── langchain_overview.md
│   ├── langgraph_reference.md
│   ├── fastapi_reference.md
│   └── chromadb_guide.md
│
├── scripts/
│   └── ingest_sample_docs.py      # one-shot ingestion helper
│
├── tests/
│   ├── test_graph.py              # node unit tests (LLM mocked)
│   ├── test_api.py                # endpoint integration tests
│   └── test_ingestion.py          # chunking / ingestion tests
│
└── ui/
    └── streamlit_app.py           # bonus Streamlit chat frontend
```

---

## 4. Setup Instructions

### Prerequisites

- Python 3.10 or higher
- An **Azure OpenAI** resource with two deployments:
  - A chat deployment (e.g. `gpt-4o-mini`)
  - An embedding deployment (e.g. `text-embedding-ada-002`)

### Step 1 — Clone and create a virtual environment

```bash
git clone https://github.com/your-username/rag-assistant.git
cd rag-assistant

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your Azure OpenAI details:

```env
AZURE_OPENAI_API_KEY=your-actual-key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

The remaining settings have sensible defaults and rarely need changing.

### Step 4 — Ingest the sample documentation

```bash
python scripts/ingest_sample_docs.py
```

This indexes the four bundled Markdown files under `docs/`. You should see output like:

```
INFO | Ingesting: langchain_overview.md
INFO |   ✓ langchain_overview.md → 8 chunks indexed
INFO | Ingesting: langgraph_reference.md
...
INFO | All done! Start the API server with: uvicorn main:app --reload
```

---

## 5. Running the Application

### FastAPI server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The interactive API docs are automatically available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Streamlit UI (bonus)

In a second terminal (with the same virtual environment active):

```bash
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

---

## 6. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `POST` | `/query` | Ask a question, get an answer with sources |
| `POST` | `/ingest` | Index documents from a list of URLs |
| `POST` | `/ingest/upload` | Index an uploaded `.md`, `.txt`, or `.html` file |
| `GET` | `/documents` | List all indexed documents and their chunk counts |
| `POST` | `/feedback` | Record thumbs-up / thumbs-down on an answer |

Full request/response schemas are visible in the Swagger UI.

---

## 7. Example Requests & Responses

### POST /query

**Request:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I add a conditional edge in LangGraph?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "In LangGraph you add a conditional edge with `workflow.add_conditional_edges()`. You pass it the source node name, a routing function that returns a string, and a dict that maps each possible return value to a destination node name. For example:\n\n```python\nworkflow.add_conditional_edges(\n    \"document_grading\",\n    route_after_grading,\n    {\"generate\": \"generation\", \"rewrite\": \"query_rewrite\"}\n)\n```\n[source: langgraph_reference.md]",
  "sources": [
    {
      "source": "langgraph_reference.md",
      "chunk_index": 2,
      "excerpt": "A conditional edge chooses the next node based on a routing function..."
    }
  ],
  "query_type": "how_to",
  "rewritten_query": "How to add conditional edges in LangGraph StateGraph?",
  "retry_count": 0,
  "fallback_used": false
}
```

---

### POST /ingest (URLs)

**Request:**
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://python.langchain.com/docs/introduction/",
      "https://fastapi.tiangolo.com/tutorial/"
    ]
  }'
```

**Response:**
```json
{
  "success": true,
  "indexed_chunks": 74,
  "documents_processed": 2,
  "message": "Indexed 74 chunks from 2 document(s)."
}
```

---

### POST /ingest/upload (file)

```bash
curl -X POST http://localhost:8000/ingest/upload \
  -F "file=@my_api_docs.md"
```

**Response:**
```json
{
  "success": true,
  "indexed_chunks": 12,
  "documents_processed": 1,
  "message": "Indexed 12 chunks from 'my_api_docs.md'."
}
```

---

### GET /documents

```bash
curl http://localhost:8000/documents
```

**Response:**
```json
{
  "total_documents": 4,
  "total_chunks": 38,
  "documents": [
    {
      "source": "langchain_overview.md",
      "total_chunks": 8,
      "sample_text": "LangChain is a framework for developing applications powered by large language models..."
    },
    {
      "source": "langgraph_reference.md",
      "total_chunks": 11,
      "sample_text": "LangGraph is a library built on top of LangChain for building stateful, multi-actor applications..."
    }
  ]
}
```

---

### POST /feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is LangGraph?",
    "answer": "LangGraph is a graph-based agent framework...",
    "rating": "thumbs_up",
    "comment": "Clear and accurate!"
  }'
```

**Response:**
```json
{
  "recorded": true,
  "message": "Thank you for your feedback!"
}
```

---

## 8. Streamlit UI

The Streamlit frontend provides:

- A **chat interface** with full conversation history displayed
- Per-answer **source citations** in a collapsible expander
- Indicator when the query was **rewritten** or a **retry** occurred
- A **thumbs-up / thumbs-down feedback** panel after every answer
- A sidebar for **file upload**, **URL ingestion**, and **viewing indexed documents**

---

## 9. Running Tests

```bash
pytest tests/ -v
```

All tests mock the LLM and vector store — no Azure credentials are required to run the test suite.

```
tests/test_graph.py::TestQueryAnalysisNode::test_returns_rewritten_query_and_type PASSED
tests/test_graph.py::TestQueryAnalysisNode::test_falls_back_on_bad_json PASSED
tests/test_graph.py::TestRetrievalNode::test_calls_similarity_search PASSED
tests/test_graph.py::TestDocumentGradingNode::test_marks_relevant_chunk PASSED
tests/test_graph.py::TestDocumentGradingNode::test_marks_all_irrelevant PASSED
tests/test_graph.py::TestRouting::test_routes_to_generate_when_relevant PASSED
tests/test_graph.py::TestRouting::test_routes_to_rewrite_when_retries_left PASSED
tests/test_graph.py::TestRouting::test_routes_to_give_up_when_retries_exhausted PASSED
tests/test_graph.py::TestGenerationNode::test_generates_answer_with_sources PASSED
tests/test_graph.py::TestFallbackNode::test_returns_fallback_message PASSED
tests/test_api.py::test_health PASSED
tests/test_api.py::test_list_documents_empty PASSED
...
```

---

## 10. Design Decisions & Tradeoffs

### Why LangGraph instead of a simple sequential chain?

Sequential chains cannot branch. The self-corrective grading step needs to route to either generation or a retry loop depending on what the grader finds. LangGraph's `add_conditional_edges` makes this explicit and easy to reason about. The graph topology is also inspectable (`graph.get_graph().draw_mermaid()`), which helps with debugging.

### Why Pydantic for GraphState?

Alternatives like `TypedDict` are more lightweight, but Pydantic gives us field-level validation, default values, and IDE autocompletion across all node functions for free. The performance overhead is negligible for this workload.

### Chunking strategy — RecursiveCharacterTextSplitter at 800/150

Technical documentation mixes paragraphs, code blocks, and bullet lists. The recursive splitter tries to split on paragraph boundaries (`\n\n`) first, then single newlines, then spaces — so it respects document structure rather than cutting at arbitrary character counts.

- **chunk_size = 800**: Large enough to hold a full code example plus surrounding prose, but small enough that similarity search returns focused results rather than entire sections.
- **chunk_overlap = 150**: Ensures sentences that fall near a chunk boundary appear fully in at least one chunk. Without overlap a sentence split in half would be partially absent from both adjacent chunks.

A semantic chunker (splitting on sentence embeddings rather than characters) would be more accurate but requires an extra embedding call per document — not worth it for a prototype.

### Why ChromaDB over FAISS?

Both are free and local. ChromaDB persists to disk automatically and supports metadata filtering. FAISS is faster for very large corpora but requires manual serialisation. For a prototype with a few hundred chunks, ChromaDB is the simpler choice.

### Azure OpenAI for both LLM and embeddings

Using the same provider for both avoids the token-mismatch problem that can occur when you index with one embedding model and later swap to a different one. It also simplifies credential management.

### In-memory feedback store

Feedback is written to a `feedback_log.jsonl` file and kept in a module-level list. For production this would go into a proper database (PostgreSQL or similar) so feedback survives restarts and can be queried for analysis.

### Tradeoffs

| Decision | Benefit | Tradeoff |
|---|---|---|
| LangGraph `StateGraph` | Explicit, inspectable control flow | More boilerplate than a simple chain |
| Pydantic `GraphState` | Type safety + IDE support | Slightly more overhead than `TypedDict` |
| ChromaDB | Zero-config persistence | Slower than FAISS for large corpora |
| Grading every chunk individually | High precision | 1 LLM call per retrieved chunk (latency) |
| JSONL feedback file | Simple, no DB dependency | Not queryable without parsing |

---

## 11. What I Would Improve with More Time

1. **Hallucination checker** — A post-generation node that asks the LLM whether the answer is fully supported by the retrieved context (Self-RAG style). If not, it could trigger a regeneration or flag the answer as uncertain.

2. **Web search fallback** — If the vector store has nothing relevant after all retries, fall back to Tavily or Serper before giving up. The answer would still cite its sources.

3. **Proper conversation memory with LangGraph checkpointing** — Right now chat history is passed in the request body. Using LangGraph's `MemorySaver` with a `thread_id` would make session memory server-side and durable.

4. **Streaming responses** — LangGraph supports `.stream()` which emits node-level events. Exposing this via a Server-Sent Events endpoint would make the UI feel much more responsive for long answers.

5. **Better chunking for code** — The recursive splitter can cut code blocks in awkward places. A custom splitter that treats triple-backtick blocks as atomic units would improve code-heavy documentation.

6. **Evaluation pipeline** — A script that runs a set of question/expected-answer pairs through the pipeline and scores precision/recall of retrieved documents and factual accuracy of generated answers.

7. **PostgreSQL for persistence** — Replace the JSONL feedback file and move ChromaDB metadata to Postgres so everything is queryable and survives container restarts.

---

## 12. Assumptions Made

- The corpus is small enough (hundreds to low thousands of chunks) that ChromaDB's in-process mode is sufficient. A hosted vector database would be needed for millions of chunks.
- Documents are in English. The prompts and chunking strategy are optimised for English text.
- Azure OpenAI deployments are already created and the API key has quota available. The system does not handle quota-exceeded errors beyond a generic 500 response.
- The `MAX_RETRY_ATTEMPTS=2` default is a reasonable balance between thoroughness and latency. Each retry adds roughly one LLM round-trip (rewrite) plus retrieval + grading.
- File uploads are limited to 10 MB. Larger documents should be split offline first.


These are the output images:
<img width="1896" height="952" alt="Screenshot 2026-05-16 111319" src="https://github.com/user-attachments/assets/c4670672-5123-4a95-850a-9312266d388e" />
<img width="1898" height="827" alt="Screenshot 2026-05-16 111333" src="https://github.com/user-attachments/assets/6f443438-bca8-4216-b330-4f1243ba45bd" />
