"""
Application entry point.

Run with:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config import get_settings
from app.core.graph import build_graph


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-compile the LangGraph workflow so the first request
    doesn't pay the compilation cost.
    Shutdown: nothing special needed here.
    """
    logger.info("Starting RAG assistant — compiling graph...")
    build_graph()
    logger.info("Graph compiled and ready.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="RAG Technical Documentation Assistant",
        description=(
            "A self-corrective Retrieval-Augmented Generation system built with "
            "LangGraph and Azure OpenAI.  Index your documentation, then ask questions."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS — adjust origins for production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    cfg = get_settings()
    uvicorn.run(
        "main:app",
        host=cfg.app_host,
        port=cfg.app_port,
        reload=True,
        log_level=cfg.log_level.lower(),
    )
