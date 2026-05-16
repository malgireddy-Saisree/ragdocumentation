"""
Thin factory that returns a configured AzureChatOpenAI instance.

Keeping this in one place means if we ever need to swap models or add
retry logic / callbacks we only touch one file.
"""

from functools import lru_cache

from langchain_openai import AzureChatOpenAI

from app.config import get_settings


@lru_cache()
def get_llm(temperature: float = 0.0) -> AzureChatOpenAI:
    """
    Return a cached LLM client.

    temperature=0 for grading/classification (deterministic),
    temperature=0.2 for generation (slightly more varied answers).
    """
    cfg = get_settings()
    return AzureChatOpenAI(
        azure_deployment=cfg.azure_openai_chat_deployment,
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
        temperature=temperature,
    )
