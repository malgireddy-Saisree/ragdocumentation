"""
Central configuration using pydantic-settings.
All values come from environment variables or the .env file.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Azure OpenAI ──────────────────────────────────────────────────────────
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_api_version: str = "2024-02-01"

    # Deployment names you created inside your Azure OpenAI resource
    azure_openai_chat_deployment: str = "gpt-4o"
    azure_openai_embedding_deployment: str = "text-embedding-3-small"

    # ── Vector store ──────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "tech_docs"

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 5
    max_retry_attempts: int = 2

    # ── App ───────────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
