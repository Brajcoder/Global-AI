"""Benchmarks with real (mocked) LLM agents, external agents and background runs."""

import json
import time

import httpx

from apps.api.routers.benchmarks import focus_coverage
from tests.helpers import FAKE_BASE_URL, FAKE_KEY


def register(client, agent_id, profile=None, **model_config):
    body = {
        "agent_id": agent_id,
        "name": agent_id,
        "model_config": {"provider": "mock", "model_name": "agent-model", **model_config},
        "prompts": {"system_prompt": "Be precise.", "instruction_template": "Task: {{prompt}}"},
    }
    if profile:
        body["profile"] = profile
    res = client.post("/api/v1/agents/register", json=body)
    assert res.status_code == 200, res.text
    return res.json()["agent"]


def run_benchmark(client, agent_ids, limit=1, **extra):
    res = client.post("/api/v1/benchmarks/run", json={"agent_ids": agent_ids, "limit_tasks": limit, **extra})
    assert res.status_code == 200, res.text
    body = res.json()
    detail = client.get(f"/api/v1/benchmarks/{body['benchmark_id']}").json()
    return body, detail


def test_baseline_agent_calls_the_configured_model(client, mock_provider):
    agent = register(client, "llm_baseline", profile="parametric_baseline", temperature=0.3, max_tokens=512)
    assert agent["execution"] == "llm" and agent["profile"] == "parametric_baseline"

    body, detail = run_benchmark(client, ["llm_baseline"], judge_llm={"provider": "mock"})
    row = body["leaderboard"][0]
    assert body["status"] == "completed" and body["telemetry_source"] == "provider_usage"
    assert (row["model_name"], row["tokens_used"], row["tasks_failed"]) == ("agent-model", 120, 0)

    request = mock_provider.agent_requests[0]
    assert request["model"] == "agent-model"
    assert (request["temperature"], request["max_tokens"]) == (0.3, 512)
    assert "tools" not in request
    assert "Be precise." in request["messages"][0]["content"]
    assert request["messages"][1]["content"].startswith("Task: Explain the correct serialization")
    assert mock_provider.headers[0]["authorization"] == f"Bearer {FAKE_KEY}"

    traj = detail["trajectories"][0]
    assert traj["raw_output"] == mock_provider.answer
    assert (traj["step_count"], traj["prompt_tokens"], traj["completion_tokens"]) == (1, 100, 20)
    assert traj["telemetry_source"] == "provider_usage" and traj["error"] == ""
    assert traj["truth_score"] == row["truth_score"] > 0
    assert traj["focus_coverage"] == 100.0  # pydantic, model_dump, field_validator all mentioned
    # The agent's answer was judged by the LLM judge as well.
    assert len(mock_provider.judge_requests) == 1


def test_verifier_agent_searches_then_answers(client, mock_provider):
    register(client, "llm_verifier", profile="disciplined_verifier")
    _, detail = run_benchmark(client, ["llm_verifier"])

    traj = detail["trajectories"][0]
    calls = json.loads(traj["invoked_tool_calls_json"])
    assert calls == [{"tool": "web_search", "query": mock_provider.search_query, "repeated": False, "results": 1}]
    assert (traj["step_count"], traj["loop_count"]) == (2, 0)
    assert traj["prompt_tokens"] == 200

    first, second = mock_provider.agent_requests
    assert first["tools"][0]["function"]["name"] == "web_search" and first["tool_choice"] == "auto"
    tool_message = second["messages"][-1]
    assert tool_message["role"] == "tool" and "Pydantic Migration Guide" in tool_message["content"]
    assert "untrusted" in tool_message["content"]


def test_looping_research_agent_is_capped_and_penalised(client, mock_provider):
    mock_provider.always_search = True  # asks for the same search forever
    register(client, "llm_looper", profile="unbounded_deep_research")
    register(client, "llm_baseline", profile="parametric_baseline")
    body, detail = run_benchmark(client, ["llm_looper", "llm_baseline"])

    traj = next(t for t in detail["trajectories"] if t["agent_id"] == "llm_looper")
    assert traj["step_count"] == 10  # MAX_AGENT_STEPS_PER_TASK
    assert traj["loop_count"] == 8  # 9 search requests, 8 of them repeats
    assert traj["raw_output"] == mock_provider.answer  # forced to answer on the last step
    assert mock_provider.agent_requests[9]["tool_choice"] == "none"

    scores = {r["agent_id"]: r for r in body["leaderboard"]}
    assert scores["llm_looper"]["eff_score"] < scores["llm_baseline"]["eff_score"]
    assert scores["llm_looper"]["tokens_used"] == 10 * 120


def test_model_without_tool_support_falls_back_to_search_first(client, mock_provider):
    mock_provider.supports_tools = False
    register(client, "llm_verifier", profile="disciplined_verifier")
    _, detail = run_benchmark(client, ["llm_verifier"])

    traj = detail["trajectories"][0]
    assert traj["raw_output"] == mock_provider.answer and traj["error"] == ""
    assert json.loads(traj["invoked_tool_calls_json"])[0]["tool"] == "web_search"
    final = mock_provider.agent_requests[-1]
    assert "tools" not in final and "Search results" in final["messages"][1]["content"]


def test_llm_failure_fails_the_task_not_the_benchmark(client, mock_provider):
    mock_provider.fail_always = 401
    register(client, "llm_baseline", profile="parametric_baseline")
    body, detail = run_benchmark(client, ["llm_baseline"], limit=2)

    row = body["leaderboard"][0]
    assert body["status"] == "completed"
    assert (row["tasks_failed"], row["truth_score"]) == (2, 0.0)
    assert all("401" in t["error"] and t["raw_output"] == "" for t in detail["trajectories"])


def test_estimated_telemetry_is_labelled_when_usage_is_missing(client, mock_provider):
    mock_provider.report_usage = False
    register(client, "llm_baseline", profile="parametric_baseline")
    body, _ = run_benchmark(client, ["llm_baseline"])
    assert body["leaderboard"][0]["telemetry_source"] == "estimated"
    assert body["leaderboard"][0]["tokens_used"] > 0


def test_agent_with_inline_base_url_and_key(client, fake_llm):
    agent = register(client, "inline_agent", provider="whatever", model_name="inline-model", base_url=FAKE_BASE_URL, api_key="inline-key")
    assert agent["execution"] == "llm"
    run_benchmark(client, ["inline_agent"])
    assert fake_llm.agent_requests[0]["model"] == "inline-model"
    assert fake_llm.headers[0]["authorization"] == "Bearer inline-key"


def test_agent_without_llm_runs_offline(client, registered_agents):
    body, detail = run_benchmark(client, ["disciplined_verifier"])
    assert body["telemetry_source"] == "estimated"
    assert detail["trajectories"][0]["observed_model"] == "offline-stub"
    assert client.get("/api/v1/agents/disciplined_verifier").json()["execution"] == "offline"


def test_external_http_agent(client, http_agent):
    http_agent["response"] = httpx.Response(
        200,
        json={
            "output": "Pydantic V2 uses model_dump() for serialization.",
            "usage": {"prompt_tokens": 30, "completion_tokens": 12},
            "steps": 4,
            "model": "their-model",
        },
    )
    res = client.post(
        "/api/v1/agents/register",
        json={"agent_id": "ext_agent", "name": "External", "framework": "langgraph", "endpoint_or_command": "http://agent.test/run", "model_config": {"model_name": "their-model"}},
    )
    assert res.json()["agent"]["execution"] == "external"

    body, detail = run_benchmark(client, ["ext_agent"])
    sent = http_agent["requests"][0]
    assert sent["task_id"] == "task_software_001" and sent["prompt"].startswith("Explain the correct")
    traj = detail["trajectories"][0]
    assert (traj["step_count"], traj["prompt_tokens"], traj["completion_tokens"]) == (4, 30, 12)
    assert traj["observed_model"] == "their-model"
    assert body["leaderboard"][0]["truth_score"] > 0

    http_agent["response"] = httpx.Response(500, text="boom")
    body, detail = run_benchmark(client, ["ext_agent"])
    assert body["leaderboard"][0]["tasks_failed"] == 1
    assert "External agent call failed" in detail["trajectories"][0]["error"]


def test_background_benchmark_reports_progress_and_persists_leaderboard(client, registered_agents):
    res = client.post("/api/v1/benchmarks/run", json={"agent_ids": registered_agents, "limit_tasks": 2, "background": True})
    assert res.status_code == 202, res.text
    started = res.json()
    assert started["status"] == "running" and started["tasks_total"] == 6

    deadline = time.time() + 20
    state = started
    while state["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
        state = client.get(f"/api/v1/benchmarks/{started['benchmark_id']}").json()

    assert state["status"] == "completed", state
    assert state["tasks_done"] == 6
    assert [r["agent_id"] for r in state["leaderboard"]] == sorted(
        registered_agents, key=lambda a: -next(r["chronos_score"] for r in state["leaderboard"] if r["agent_id"] == a)
    )
    listed = client.get("/api/v1/benchmarks").json()
    assert any(r["benchmark_id"] == started["benchmark_id"] and len(r["leaderboard"]) == 3 for r in listed)


def test_duplicate_agent_ids_run_once(client, registered_agents):
    body, detail = run_benchmark(client, ["disciplined_verifier", "disciplined_verifier"])
    assert len(body["leaderboard"]) == 1 and len(detail["trajectories"]) == 1


def test_unknown_judge_provider_is_rejected_before_the_run_starts(client, registered_agents):
    before = len(client.get("/api/v1/benchmarks?limit=100").json())
    res = client.post("/api/v1/benchmarks/run", json={"agent_ids": registered_agents[:1], "judge_llm": {"provider": "nope", "model": "m"}})
    assert res.status_code == 400
    assert len(client.get("/api/v1/benchmarks?limit=100").json()) == before


def test_suites_endpoint(client):
    suite = client.get("/api/v1/benchmarks/suites").json()[0]
    assert suite["suite_id"] == "temporal_traps_v1"
    assert suite["tasks"] == sum(suite["cohorts"].values()) >= 15


def test_focus_coverage():
    assert focus_coverage("Use model_dump() and @field_validator.", ["model_dump", "field_validator"]) == 100.0
    assert focus_coverage("Status is needs review.", ["needs_review", "contradicted"]) == 50.0
    assert focus_coverage("anything", []) == 0.0
