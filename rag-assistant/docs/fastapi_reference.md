# FastAPI Reference Guide

FastAPI is a modern, high-performance Python web framework for building APIs. It is built on Starlette (ASGI) and uses Pydantic for data validation.

## Installation

```bash
pip install fastapi uvicorn[standard]
```

## Creating an Application

```python
from fastapi import FastAPI

app = FastAPI(title="My API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

Run it with:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Path Operations

FastAPI uses decorators to define routes. Common HTTP methods:

```python
@app.get("/items/{item_id}")
async def get_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}

@app.post("/items/")
async def create_item(item: Item):
    return item

@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, **item.model_dump()}

@app.delete("/items/{item_id}")
async def delete_item(item_id: int):
    return {"deleted": item_id}
```

## Request and Response Models

Use Pydantic BaseModel for automatic validation and serialisation.

```python
from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(None, ge=1, le=20)

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    # process req.question
    return QueryResponse(answer="...", sources=[])
```

## File Uploads

Use `UploadFile` and `File` for multipart form data.

```python
from fastapi import File, UploadFile

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return {"filename": file.filename, "size": len(content)}
```

## HTTP Exceptions

```python
from fastapi import HTTPException, status

@app.get("/items/{item_id}")
async def get_item(item_id: int):
    if item_id not in db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item {item_id} not found"
        )
    return db[item_id]
```

## Dependency Injection

FastAPI has a powerful DI system via `Depends`.

```python
from fastapi import Depends

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/users/")
async def list_users(db=Depends(get_db)):
    return db.query(User).all()
```

## Middleware

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Lifespan Events

Use the `lifespan` context manager for startup and shutdown logic.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: load models, connect to DB
    print("Starting up")
    yield
    # shutdown: release resources
    print("Shutting down")

app = FastAPI(lifespan=lifespan)
```

## Automatic Documentation

FastAPI auto-generates interactive docs:
- Swagger UI at `http://localhost:8000/docs`
- ReDoc at `http://localhost:8000/redoc`
- OpenAPI JSON at `http://localhost:8000/openapi.json`

## Status Codes

Common HTTP status codes available via `fastapi.status`:
- `200 OK` – successful GET / PUT
- `201 Created` – successful POST that creates a resource
- `204 No Content` – successful DELETE
- `400 Bad Request` – malformed input
- `404 Not Found` – resource does not exist
- `422 Unprocessable Entity` – validation failure (FastAPI default for bad body)
- `500 Internal Server Error` – unexpected server failure
