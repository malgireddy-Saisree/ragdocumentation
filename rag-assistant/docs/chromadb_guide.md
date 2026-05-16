# ChromaDB Vector Store Guide

ChromaDB is an open-source embedding database designed for building AI applications. It stores embeddings alongside their associated text and metadata, enabling fast similarity search.

## Installation

```bash
pip install chromadb
```

## Basic Usage

```python
import chromadb

# In-memory client (no persistence)
client = chromadb.Client()

# Persistent client (data survives restarts)
client = chromadb.PersistentClient(path="./chroma_db")
```

## Collections

A collection is analogous to a table in a relational database. Each collection stores embeddings with their documents and metadata.

```python
# Create or get an existing collection
collection = client.get_or_create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)
```

## Adding Documents

```python
collection.add(
    ids=["doc1", "doc2", "doc3"],
    documents=["Python is great", "FastAPI is fast", "LangChain simplifies LLMs"],
    metadatas=[
        {"source": "intro.md", "chunk_idx": 0},
        {"source": "web.md",   "chunk_idx": 0},
        {"source": "ai.md",    "chunk_idx": 0},
    ]
)
```

If you supply pre-computed embeddings, pass them as the `embeddings` parameter. Otherwise ChromaDB uses its default embedding function.

## Querying

```python
results = collection.query(
    query_texts=["How do I build an API?"],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)

# results["documents"][0]  → list of matching document texts
# results["metadatas"][0]  → list of metadata dicts
# results["distances"][0]  → list of distance scores (lower = more similar for L2)
```

With pre-computed query embeddings:
```python
results = collection.query(
    query_embeddings=[my_vector],
    n_results=5
)
```

## Distance Metrics

ChromaDB supports three distance functions, set at collection creation time:
- `"l2"` (default) – Euclidean distance, lower = more similar
- `"cosine"` – Cosine distance, lower = more similar (range 0–2)
- `"ip"` – Inner product, higher = more similar

For normalised embeddings (like OpenAI's), cosine and L2 produce equivalent rankings.

## Updating and Deleting

```python
# Update existing entries by ID
collection.update(
    ids=["doc1"],
    documents=["Updated Python content"],
)

# Delete by ID
collection.delete(ids=["doc2"])
```

## Listing and Counting

```python
# Total number of entries
count = collection.count()

# Retrieve all entries (use carefully on large collections)
all_data = collection.get(include=["documents", "metadatas"])
```

## Filtering with Metadata

ChromaDB supports metadata filtering alongside similarity search.

```python
results = collection.query(
    query_texts=["authentication"],
    n_results=5,
    where={"source": "fastapi_reference.md"}  # filter by metadata field
)
```

## Integration with LangChain

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    azure_deployment="text-embedding-ada-002",
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_version="2024-02-01"
)

vectorstore = Chroma(
    collection_name="my_docs",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

# Add texts
vectorstore.add_texts(["Document content here..."])

# Similarity search
docs = vectorstore.similarity_search("my query", k=5)
```

## Performance Tips

- Use `PersistentClient` so embeddings survive restarts — recomputing them is expensive.
- For large corpora (> 100k chunks), consider FAISS or a hosted vector database like Pinecone.
- Choose `cosine` similarity when working with OpenAI or sentence-transformer embeddings, as these models produce unit-normalised vectors by default.
- Batch your `add()` calls (e.g. 100 items at a time) rather than adding one document at a time.
