"""
conftest.py

Shared pytest fixtures and env-var mocking so tests can import app code
without needing a real .env file with Azure credentials.
"""

import os
import pytest


# Apply before any test module is imported
@pytest.fixture(autouse=True, scope="session")
def mock_env_vars():
    env_patch = {
        "AZURE_OPENAI_API_KEY":              "test-key",
        "AZURE_OPENAI_ENDPOINT":             "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_VERSION":          "2024-02-01",
        "AZURE_OPENAI_CHAT_DEPLOYMENT":      "gpt-4o",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
        "CHROMA_PERSIST_DIR":                "/tmp/test_chroma",
        "MAX_RETRY_ATTEMPTS":                "2",
    }
    old_vals = {k: os.environ.get(k) for k in env_patch}
    os.environ.update(env_patch)
    yield
    for k, v in old_vals.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
