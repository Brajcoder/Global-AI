import asyncio

from integrations.llm.client import LLMConfig
from packages.evidence_engine.llm_judge import judge_claim
from tests.helpers import FAKE_BASE_URL

STALE_TEXT = "To serialize in Pydantic, use .dict() and @validator."
CONFIG = LLMConfig(provider="mock", base_url=FAKE_BASE_URL, api_key="k", model="judge-model", max_retries=0)
ROWS = [
    {"url": "https://docs.example.dev/a", "title": "A", "excerpt": "Widget 2 removed frobnicate()."},
    {"url": "https://blog.example.com/b", "title": "B", "excerpt": "Totally unrelated cooking recipe."},
]


def judge(claim="Call frobnicate() on the widget.", rows=ROWS, config=CONFIG):
    return asyncio.run(judge_claim(config, claim, rows))


def test_no_llm_uses_heuristics():
    outcome = judge(config=None)
    assert outcome.llm_calls == 0
    assert all(j.source == "heuristic" for j in outcome.judgements)


def test_llm_judgements_are_parsed_and_clamped(fake_llm):
    fake_llm.judge_reply = {
        "judgements": [
            {"index": 1, "stance": "contradicts", "confidence": 250, "outdated_term": "frobnicate()", "replacement": "`frob()`", "correction": "Use frob()."},
            {"index": 2, "stance": "irrelevant", "confidence": 80},
        ]
    }
    outcome = judge()
    first, second = outcome.judgements
    assert (first.stance, first.confidence, first.source) == ("contradicts", 100.0, "llm")
    assert (first.outdated[0].old, first.outdated[0].new) == ("frobnicate", "frob")
    assert first.correction == "Use frob()."
    assert (second.stance, second.confidence) == ("irrelevant", 0.0)
    assert outcome.llm_calls == 1 and outcome.llm_tokens == 120
    assert fake_llm.requests[0]["temperature"] == 0.0


def test_excerpts_the_model_skips_keep_heuristic_judgement(fake_llm):
    fake_llm.judge_reply = {"judgements": [{"index": 2, "stance": "contextual", "confidence": 90}, {"index": 9, "stance": "supports"}, {"index": 1, "stance": "bogus"}]}
    first, second = judge().judgements
    assert first.source == "heuristic"
    assert (second.source, second.stance, second.confidence) == ("llm", "contextual", 50.0)


def test_bad_json_and_http_errors_fall_back(fake_llm):
    fake_llm.judge_reply = "I cannot comply."
    outcome = judge()
    assert outcome.llm_error and all(j.source == "heuristic" for j in outcome.judgements)

    fake_llm.fail_always = 500
    outcome = judge(claim="Another claim about frobnicate().")
    assert "500" in outcome.llm_error and all(j.source == "heuristic" for j in outcome.judgements)


def test_identical_judgements_are_cached(fake_llm):
    judge()
    second = judge()
    assert len(fake_llm.requests) == 1
    assert second.llm_calls == 0 and second.judgements[0].source == "llm"


def test_prompt_marks_excerpts_as_untrusted(fake_llm):
    judge()
    system, user = (m["content"] for m in fake_llm.requests[0]["messages"])
    assert "untrusted" in system
    assert "<excerpt>" in user and "Call frobnicate() on the widget." in user


# ---- through the API ---------------------------------------------------------------------------


def verify(client, **options):
    res = client.post("/api/v1/verify", json={"text": STALE_TEXT, "options": {"persist": False, **options}})
    assert res.status_code == 200, res.text
    return res.json()


def test_verify_uses_llm_judge_when_requested(client, mock_provider):
    mock_provider.judge_reply = {
        "judgements": [{"index": 1, "stance": "contradicts", "confidence": 95, "outdated_term": ".dict()", "replacement": "model_dump()"}]
    }
    body = verify(client, llm={"provider": "mock"})
    claim = body["claims"][0]
    assert claim["status"] == "STALE"
    assert claim["suggested_patch"] == "Per current docs: replace `dict` with `model_dump`."
    assert claim["support_confidence"] == 95
    assert body["usage"]["stance_judge"] == "llm"
    assert body["usage"]["judge_model"] == "fake-model"
    assert body["usage"]["llm_calls"] == 1 and body["usage"]["llm_tokens"] == 120


def test_llm_verdict_overrides_the_heuristic(client, mock_provider):
    # Heuristics call this STALE; the judge says "supports", which proves the LLM verdict is what counts.
    mock_provider.judge_reply = {"judgements": [{"index": 1, "stance": "supports", "confidence": 80}]}
    assert verify(client, llm={"provider": "mock"})["claims"][0]["status"] == "VERIFIED"


def test_contradiction_without_outdated_term_uses_correction(client, mock_provider):
    mock_provider.judge_reply = {
        "judgements": [{"index": 1, "stance": "contradicts", "confidence": 90, "correction": "Pydantic V2 uses model_dump()."}]
    }
    claim = verify(client, llm={"provider": "mock"})["claims"][0]
    assert claim["status"] == "CONTRADICTED"
    assert claim["suggested_patch"] == "Per current sources: Pydantic V2 uses model_dump()."


def test_inline_base_url_key_and_model_on_the_request(client, fake_llm):
    body = verify(client, llm={"base_url": FAKE_BASE_URL, "api_key": "inline-key", "model": "inline-model"})
    assert body["usage"]["judge_model"] == "inline-model"
    assert fake_llm.headers[0]["authorization"] == "Bearer inline-key"
    assert fake_llm.requests[0]["model"] == "inline-model"


def test_judge_failure_falls_back_to_heuristics(client, mock_provider):
    mock_provider.fail_always = 500
    body = verify(client, llm={"provider": "mock"})
    assert body["claims"][0]["status"] == "STALE"  # the heuristic verdict
    assert body["usage"]["stance_judge"] == "heuristic"
    assert body["usage"]["llm_errors"] == 1


def test_llm_judge_can_be_switched_off(client, mock_provider):
    body = verify(client, llm={"provider": "mock"}, llm_judge=False)
    assert body["usage"]["stance_judge"] == "heuristic"
    assert mock_provider.requests == []


def test_no_llm_configured_means_heuristic_judge(client):
    assert verify(client)["usage"]["stance_judge"] == "heuristic"


def test_unknown_provider_override_is_rejected(client):
    res = client.post("/api/v1/verify", json={"text": STALE_TEXT, "options": {"llm": {"provider": "nope", "model": "m"}}})
    assert res.status_code == 400
    assert "not configured" in res.json()["detail"]


def test_base_url_without_model_is_rejected(client):
    res = client.post("/api/v1/verify", json={"text": STALE_TEXT, "options": {"llm": {"base_url": FAKE_BASE_URL}}})
    assert res.status_code == 422
