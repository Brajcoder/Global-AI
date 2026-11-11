from integrations.llm.providers import default_llm, resolve_llm
from packages.contracts.database import get_session
from packages.contracts.settings import get_settings
from tests.helpers import FAKE_BASE_URL, FAKE_KEY


def test_register_list_and_delete_provider(client, fake_llm):
    res = client.post(
        "/api/v1/providers",
        json={"name": "My-Proxy", "base_url": FAKE_BASE_URL + "/", "api_key": FAKE_KEY, "default_model": "m1"},
    )
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "registered"
    assert res.json()["provider"]["name"] == "my-proxy"
    assert res.json()["provider"]["base_url"] == FAKE_BASE_URL

    listed = client.get("/api/v1/providers").json()["providers"]
    mine = next(p for p in listed if p["name"] == "my-proxy")
    assert mine["kind"] == "registered" and mine["default_model"] == "m1"
    # Presets are listed too, unconfigured because the test environment has no keys.
    assert next(p for p in listed if p["name"] == "openai")["configured"] is False

    assert client.delete("/api/v1/providers/my-proxy").json()["status"] == "deleted"
    assert client.delete("/api/v1/providers/my-proxy").status_code == 404


def test_api_key_is_never_returned(client, mock_provider):
    assert FAKE_KEY not in client.get("/api/v1/providers").text
    res = client.post(
        "/api/v1/agents/register",
        json={
            "agent_id": "secret_agent",
            "name": "Secret",
            "model_config": {"model_name": "m", "base_url": FAKE_BASE_URL, "api_key": FAKE_KEY},
        },
    )
    assert res.status_code == 200, res.text
    assert FAKE_KEY not in res.text
    assert FAKE_KEY not in client.get("/api/v1/agents/secret_agent").text
    assert FAKE_KEY not in client.get("/api/v1/agents").text
    assert client.get("/api/v1/agents/secret_agent").json()["execution"] == "llm"
    client.delete("/api/v1/agents/secret_agent")


def test_updating_provider_without_key_keeps_stored_key(client, mock_provider):
    res = client.post("/api/v1/providers", json={"name": "mock", "base_url": FAKE_BASE_URL, "default_model": "m2"})
    assert res.json()["status"] == "updated"
    assert client.post("/api/v1/providers/mock/test", json={}).json()["ok"] is True
    assert mock_provider.headers[-1]["authorization"] == f"Bearer {FAKE_KEY}"
    assert mock_provider.requests[-1]["model"] == "m2"


def test_provider_validation(client):
    bad_url = client.post("/api/v1/providers", json={"name": "x1", "base_url": "ftp://nope"})
    assert bad_url.status_code == 422
    bad_name = client.post("/api/v1/providers", json={"name": "has space", "base_url": FAKE_BASE_URL})
    assert bad_name.status_code == 422


def test_provider_test_endpoint(client, mock_provider):
    ok = client.post("/api/v1/providers/mock/test", json={"model": "other-model"}).json()
    assert ok["ok"] is True and ok["model"] == "other-model"

    mock_provider.fail_always = 401
    failed = client.post("/api/v1/providers/mock/test").json()
    assert failed["ok"] is False and "401" in failed["error"]

    assert client.post("/api/v1/providers/unknown-provider/test").status_code == 400


def test_resolve_inline_connection():
    config = resolve_llm(None, "anything", "my-model", "http://host:1234/v1", "k", temperature=0.7, max_tokens=50)
    assert (config.base_url, config.api_key, config.model) == ("http://host:1234/v1", "k", "my-model")
    assert (config.temperature, config.max_tokens) == (0.7, 50)


def test_resolve_registered_provider(client, mock_provider):
    with get_session() as session:
        config = resolve_llm(session, "MOCK", "")
        assert (config.base_url, config.api_key, config.model) == (FAKE_BASE_URL, FAKE_KEY, "fake-model")
        assert resolve_llm(session, "mock", "explicit").model == "explicit"


def test_unconfigured_provider_resolves_to_none(client):
    with get_session() as session:
        assert resolve_llm(session, "groq", "llama-3.3-70b-versatile") is None
        assert resolve_llm(session, "no-such-provider", "m") is None
        assert default_llm(session) is None


def test_preset_with_inline_key():
    config = resolve_llm(None, "openai", "gpt-test", api_key="sk-inline")
    assert (config.base_url, config.api_key) == ("https://api.openai.com/v1", "sk-inline")


def test_default_llm_from_env_and_fallback_rules(client, monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "llm_base_url", "http://default.test/v1")
    monkeypatch.setattr(settings, "llm_api_key", "dk")
    monkeypatch.setattr(settings, "llm_model", "default-model")
    with get_session() as session:
        assert default_llm(session).model == "default-model"
        # "default" provider: the caller's model name is honoured on the default connection.
        assert resolve_llm(session, "default", "other").model == "other"
        # Unknown provider: falls back to the default connection AND the default model.
        fallback = resolve_llm(session, "no-such-provider", "their-model")
        assert (fallback.base_url, fallback.model) == ("http://default.test/v1", "default-model")
    assert client.get("/api/v1/health").json()["default_llm"] == {"provider": "default", "model": "default-model"}


def test_preset_key_from_env(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "groq_api_key", "gsk-env")
    config = resolve_llm(None, "groq", "llama-x")
    assert (config.base_url, config.api_key, config.model) == ("https://api.groq.com/openai/v1", "gsk-env", "llama-x")
    # With DEFAULT_LLM_PROVIDER=groq this also becomes the default LLM.
    assert default_llm().model == settings.default_llm_model
