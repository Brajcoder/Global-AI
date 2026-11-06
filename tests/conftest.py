"""Test setup: isolated temp database and cache, offline Tavily mode.

Environment variables must be set before any project module is imported, because settings and the
DB engine are created at import time. Env vars override values from a developer's .env file.
"""

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="chronos_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["TAVILY_CACHE_DB_PATH"] = f"{_TMP}/tavily_cache.sqlite"
os.environ["TAVILY_API_KEY"] = ""

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from apps.api.main import app  # noqa: E402

AGENT_IDS = ["parametric_baseline", "unbounded_deep_research", "disciplined_verifier"]


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def registered_agents(client):
    for agent_id in AGENT_IDS:
        res = client.post(
            "/api/v1/agents/register",
            json={
                "agent_id": agent_id,
                "name": agent_id,
                "framework": "custom",
                "model_config": {"provider": "groq", "model_name": "llama-3.3-70b-versatile"},
                "prompts": {"system_prompt": "s", "instruction_template": "{{prompt}}"},
            },
        )
        assert res.status_code == 200, res.text
    return AGENT_IDS
