"""Test setup: isolated temp database and cache, offline Tavily mode, no real LLM.

Environment variables must be set before any project module is imported, because settings and the
DB engine are created at import time. Env vars override values from a developer's .env file, so
every key is blanked here: the suite must never make a real network call or spend credits.
"""

import json
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="chronos_tests_")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP}/test.db"
os.environ["TAVILY_CACHE_DB_PATH"] = f"{_TMP}/tavily_cache.sqlite"
for _name in (
    "TAVILY_API_KEY",
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "BIFROST_API_KEY",
    "API_KEY",
):
    os.environ[_name] = ""
os.environ["DEFAULT_LLM_PROVIDER"] = "groq"
os.environ["LLM_MAX_RETRIES"] = "2"
os.environ["CORS_ALLOW_ORIGINS"] = "*"

import httpx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from agents import runtime as agent_runtime  # noqa: E402
from apps.api.main import app  # noqa: E402
from integrations.llm import client as llm_client  # noqa: E402
from packages.evidence_engine import llm_judge  # noqa: E402
from tests.helpers import FAKE_BASE_URL, FAKE_KEY  # noqa: E402

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


class FakeLLM:
    """An in-process OpenAI-compatible /chat/completions endpoint with scriptable behaviour."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.urls: list[str] = []
        self.answer = "Pydantic V2 uses model_dump() for serialization and @field_validator for validation."
        self.search_query = "pydantic v2 model_dump field_validator"
        self.always_search = False  # keep asking for the same search (a looping agent)
        self.supports_tools = True
        self.report_usage = True
        self.fail_statuses: list[int] = []  # statuses returned (in order) before succeeding
        self.fail_always: int | None = None
        self.judge_reply: dict | str | None = None
        self.reject_param: str | None = None  # e.g. "max_tokens": 400 until the client stops sending it

    # -- helpers -------------------------------------------------------------------------------
    @property
    def agent_requests(self) -> list[dict]:
        return [r for r in self.requests if "response_format" not in r]

    @property
    def judge_requests(self) -> list[dict]:
        return [r for r in self.requests if "response_format" in r]

    def _reply(self, body: dict, message: dict) -> httpx.Response:
        payload = {"id": "cmpl-test", "model": body["model"], "choices": [{"index": 0, "message": message}]}
        if self.report_usage:
            payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
        return httpx.Response(200, json=payload)

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        self.urls.append(str(request.url))

        if self.fail_always is not None:
            return httpx.Response(self.fail_always, json={"error": {"message": "scripted failure"}})
        if self.fail_statuses:
            return httpx.Response(self.fail_statuses.pop(0), json={"error": {"message": "scripted failure"}})
        if self.reject_param and self.reject_param in body:
            hint = "max_completion_tokens" if self.reject_param == "max_tokens" else self.reject_param
            return httpx.Response(400, json={"error": {"message": f"Unsupported parameter: use '{hint}' instead."}})

        if "response_format" in body:  # the stance judge
            reply = self.judge_reply
            if reply is None:
                reply = {"judgements": [{"index": 1, "stance": "supports", "confidence": 90}]}
            return self._reply(body, {"role": "assistant", "content": reply if isinstance(reply, str) else json.dumps(reply)})

        if body.get("tools"):
            if not self.supports_tools:
                return httpx.Response(400, json={"error": {"message": "tools is not supported by this model"}})
            searched = any(m.get("role") == "tool" for m in body["messages"])
            if body.get("tool_choice") != "none" and (self.always_search or not searched):
                call = {
                    "id": f"call_{len(self.requests)}",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"query": self.search_query})},
                }
                return self._reply(body, {"role": "assistant", "content": None, "tool_calls": [call]})
        return self._reply(body, {"role": "assistant", "content": self.answer})


@pytest.fixture
def fake_llm():
    fake = FakeLLM()
    llm_client.set_transport(httpx.MockTransport(fake.handler))
    llm_judge.clear_cache()
    yield fake
    llm_client.set_transport(None)
    llm_judge.clear_cache()


@pytest.fixture
def mock_provider(client, fake_llm):
    """A registered provider named "mock" that points at the fake LLM."""
    res = client.post(
        "/api/v1/providers",
        json={"name": "mock", "base_url": FAKE_BASE_URL, "api_key": FAKE_KEY, "default_model": "fake-model"},
    )
    assert res.status_code == 200, res.text
    yield fake_llm
    client.delete("/api/v1/providers/mock")


@pytest.fixture
def http_agent():
    """Install a fake external agent endpoint; yields a dict the test can script."""
    state = {"response": httpx.Response(200, json={"output": "ok"}), "requests": []}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(json.loads(request.content))
        return state["response"]

    agent_runtime.set_http_transport(httpx.MockTransport(handler))
    yield state
    agent_runtime.set_http_transport(None)
