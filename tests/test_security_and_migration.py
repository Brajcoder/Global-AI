import pytest
from sqlalchemy import inspect, text
from sqlmodel import create_engine

from packages.contracts.database import _add_missing_columns
from packages.contracts.settings import get_settings

VERIFY_BODY = {"text": "Binary search runs in O(log n) time.", "options": {"persist": False}}


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "api_key", "s3cret")
    return "s3cret"


def test_no_api_key_configured_means_open_access(client):
    assert client.post("/api/v1/verify", json=VERIFY_BODY).status_code == 200
    assert client.get("/api/v1/health").json()["auth_required"] is False


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("post", "/api/v1/verify", VERIFY_BODY),
        ("post", "/api/v1/agents/register", {"agent_id": "a1", "name": "a", "model_config": {"model_name": "m"}}),
        ("delete", "/api/v1/agents/a1", None),
        ("post", "/api/v1/benchmarks/run", {"agent_ids": ["a1"]}),
        ("post", "/api/v1/providers", {"name": "p1", "base_url": "http://x.test/v1"}),
        ("delete", "/api/v1/providers/p1", None),
        ("post", "/api/v1/providers/p1/test", None),
        ("post", "/api/v1/claims/clm_x/reverify", None),
    ],
)
def test_write_endpoints_require_the_key(client, api_key, method, path, body):
    kwargs = {"json": body} if body is not None else {}
    assert getattr(client, method)(path, **kwargs).status_code == 401
    assert getattr(client, method)(path, headers={"X-API-Key": "wrong"}, **kwargs).status_code == 401


def test_correct_key_is_accepted_and_reads_stay_open(client, api_key):
    assert client.post("/api/v1/verify", json=VERIFY_BODY, headers={"X-API-Key": api_key}).status_code == 200
    for path in ("/api/v1/health", "/api/v1/agents", "/api/v1/providers", "/api/v1/benchmarks", "/api/v1/verifications"):
        assert client.get(path).status_code == 200, path
    assert client.get("/api/v1/health").json()["auth_required"] is True


@pytest.mark.parametrize("agent_id", ["a", "has space", "../etc/passwd", "x" * 65, "<script>"])
def test_invalid_agent_ids_are_rejected(client, agent_id):
    res = client.post("/api/v1/agents/register", json={"agent_id": agent_id, "name": "n", "model_config": {"model_name": "m"}})
    assert res.status_code == 422


def test_agent_register_delete_roundtrip(client):
    body = {"agent_id": "temp-agent.v1", "name": "Temp", "model_config": {"model_name": "m"}}
    created = client.post("/api/v1/agents/register", json=body).json()
    assert created["status"] == "registered" and created["agent"]["profile"] == "parametric_baseline"
    assert client.delete("/api/v1/agents/temp-agent.v1").status_code == 200
    assert client.get("/api/v1/agents/temp-agent.v1").status_code == 404
    assert client.delete("/api/v1/agents/temp-agent.v1").status_code == 404


def test_missing_columns_are_added_to_an_old_database(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'old.db').as_posix()}")
    with engine.begin() as conn:
        # Schema as created by the first release, before profile/base_url/api_key and run progress existed.
        conn.execute(text("CREATE TABLE agentmanifestdb (id VARCHAR PRIMARY KEY, name VARCHAR, model_name VARCHAR)"))
        conn.execute(text("CREATE TABLE benchmarkrundb (id VARCHAR PRIMARY KEY, status VARCHAR)"))
        conn.execute(text("INSERT INTO agentmanifestdb VALUES ('old', 'Old agent', 'm')"))
        conn.execute(text("INSERT INTO benchmarkrundb VALUES ('bm_old', 'completed')"))

    _add_missing_columns(engine)
    _add_missing_columns(engine)  # idempotent

    columns = {c["name"] for c in inspect(engine).get_columns("agentmanifestdb")}
    assert {"profile", "base_url", "api_key"} <= columns
    with engine.connect() as conn:
        assert conn.execute(text("SELECT profile, api_key FROM agentmanifestdb")).one() == ("", "")
        assert conn.execute(text("SELECT tasks_total, telemetry_source FROM benchmarkrundb")).one() == (0, "estimated")
    engine.dispose()


def test_verification_detail_includes_evidence_and_scores(client):
    created = client.post("/api/v1/verify", json={"text": "To serialize in Pydantic, use .dict() and @validator."}).json()
    detail = client.get(f"/api/v1/verification/{created['verification_id']}").json()
    claim = detail["claims"][0]
    assert claim["evidence"][0]["url"] == "https://docs.pydantic.dev/latest/migration/"
    assert claim["scores"]["source_quality"] == 100.0
    assert client.get("/api/v1/verifications?limit=1").json()[0]["id"] == created["verification_id"]
