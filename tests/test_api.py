STALE_TEXT = "To serialize in Pydantic, use .dict() and @validator."


def test_health(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"


def test_verify_can_be_called_repeatedly(client):
    # Regression: claim IDs were clm_001.. per run and collided on the primary key on the 2nd call.
    ids = set()
    for _ in range(3):
        res = client.post("/api/v1/verify", json={"text": STALE_TEXT})
        assert res.status_code == 200, res.text
        ids.update(c["claim_id"] for c in res.json()["claims"])
    assert len(ids) == 3


def test_stale_pydantic_claim_is_flagged_with_concrete_patch(client):
    body = client.post("/api/v1/verify", json={"text": STALE_TEXT, "options": {"persist": False}}).json()
    claim = body["claims"][0]
    assert claim["status"] == "STALE"
    assert "model_dump" in claim["suggested_patch"]
    assert "field_validator" in claim["suggested_patch"]
    assert body["overall_decision"] == "stale_detected"
    assert body["usage"]["offline_mode"] is True


def test_modern_pydantic_claim_is_not_stale(client):
    text = "Pydantic V2 uses model_dump() for serialization and @field_validator for validation."
    claim = client.post("/api/v1/verify", json={"text": text, "options": {"persist": False}}).json()["claims"][0]
    assert claim["status"] == "VERIFIED"


def test_unrelated_claim_gets_no_offline_pydantic_evidence(client):
    text = "The latest API version of FastAPI is 0.116."
    claim = client.post("/api/v1/verify", json={"text": text, "options": {"persist": False}}).json()["claims"][0]
    assert claim["evidence"] == []
    assert claim["status"] == "UNVERIFIED"


def test_strict_mode_blocks_stale_answers(client):
    body = client.post(
        "/api/v1/verify", json={"text": STALE_TEXT, "mode": "strict", "options": {"persist": False}}
    ).json()
    assert body["policy"]["recommended_action"] == "block"


def test_empty_text_is_not_reported_clean(client):
    body = client.post("/api/v1/verify", json={"text": "", "options": {"persist": False}}).json()
    assert body["overall_decision"] == "no_claims"


def test_get_verification_returns_persisted_claims(client):
    created = client.post("/api/v1/verify", json={"text": STALE_TEXT}).json()
    res = client.get(f"/api/v1/verification/{created['verification_id']}")
    assert res.status_code == 200
    assert len(res.json()["claims"]) == len(created["claims"])


def test_reverify_and_history(client):
    created = client.post("/api/v1/verify", json={"text": STALE_TEXT}).json()
    claim_id = created["claims"][0]["claim_id"]
    res = client.post(f"/api/v1/claims/{claim_id}/reverify")
    assert res.status_code == 200, res.text
    assert res.json()["previous_status"] == "STALE"
    assert client.get(f"/api/v1/claims/{claim_id}/history").status_code == 200


def test_benchmark_runs_all_tasks_for_all_agents(client, registered_agents):
    # Regression: the benchmark crashed on the second task because of the claim-ID collision.
    res = client.post("/api/v1/benchmarks/run", json={"agent_ids": registered_agents})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["status"] == "completed"
    scores = {row["agent_id"]: row for row in body["leaderboard"]}
    assert set(scores) == set(registered_agents)
    # The stale-API baseline must not beat the agent that uses current APIs.
    assert scores["disciplined_verifier"]["truth_score"] > scores["parametric_baseline"]["truth_score"]

    detail = client.get(f"/api/v1/benchmarks/{body['benchmark_id']}").json()
    assert detail["run"]["status"] == "completed"
    assert len(detail["trajectories"]) == body["tasks_total"] == 15 * len(registered_agents)


def test_benchmark_rejects_unknown_agent(client):
    res = client.post("/api/v1/benchmarks/run", json={"agent_ids": ["does_not_exist"]})
    assert res.status_code == 404


def test_benchmark_rejects_unknown_suite(client, registered_agents):
    res = client.post("/api/v1/benchmarks/run", json={"agent_ids": registered_agents[:1], "suite_id": "nope"})
    assert res.status_code == 400


def test_reregistering_agent_updates_it(client, registered_agents):
    res = client.post(
        "/api/v1/agents/register",
        json={
            "agent_id": registered_agents[0],
            "name": registered_agents[0],
            "framework": "custom",
            "model_config": {"provider": "groq", "model_name": "m2"},
            "prompts": {"system_prompt": "s", "instruction_template": "t"},
        },
    )
    assert res.json()["status"] == "updated"
    assert res.json()["agent"]["model_name"] == "m2"
