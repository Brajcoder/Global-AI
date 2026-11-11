import asyncio

import pytest

from integrations.llm import client as llm_client
from integrations.llm.client import LLMConfig, LLMError, parse_json_object
from tests.helpers import FAKE_BASE_URL, FAKE_KEY

CONFIG = LLMConfig(provider="mock", base_url=FAKE_BASE_URL, api_key=FAKE_KEY, model="fake-model", max_retries=2)
MESSAGES = [{"role": "user", "content": "hi"}]


def chat(config=CONFIG, **kwargs):
    return asyncio.run(llm_client.chat(config, MESSAGES, **kwargs))


def test_chat_sends_model_key_and_base_url(fake_llm):
    result = chat()
    assert result.content == fake_llm.answer
    assert fake_llm.urls == [f"{FAKE_BASE_URL}/chat/completions"]
    assert fake_llm.headers[0]["authorization"] == f"Bearer {FAKE_KEY}"
    assert fake_llm.requests[0]["model"] == "fake-model"
    assert (result.prompt_tokens, result.completion_tokens, result.usage_reported) == (100, 20, True)


def test_base_url_with_trailing_slash_or_full_path(fake_llm):
    chat(LLMConfig(provider="p", base_url=FAKE_BASE_URL + "/", api_key="", model="m"))
    chat(LLMConfig(provider="p", base_url=FAKE_BASE_URL + "/chat/completions", api_key="", model="m"))
    assert fake_llm.urls == [f"{FAKE_BASE_URL}/chat/completions"] * 2
    # No key (e.g. a local Ollama server) means no Authorization header at all.
    assert "authorization" not in fake_llm.headers[0]


def test_retries_transient_errors_then_succeeds(fake_llm):
    fake_llm.fail_statuses = [500, 429]
    assert chat().content == fake_llm.answer
    assert len(fake_llm.requests) == 3


def test_gives_up_after_max_retries(fake_llm):
    fake_llm.fail_always = 503
    with pytest.raises(LLMError) as err:
        chat()
    assert err.value.status_code == 503
    assert len(fake_llm.requests) == 3  # 1 try + 2 retries


def test_auth_error_is_not_retried(fake_llm):
    fake_llm.fail_always = 401
    with pytest.raises(LLMError) as err:
        chat()
    assert err.value.status_code == 401
    assert "scripted failure" in str(err.value)
    assert len(fake_llm.requests) == 1


def test_adapts_to_models_that_reject_max_tokens(fake_llm):
    fake_llm.reject_param = "max_tokens"
    assert chat().content == fake_llm.answer
    assert "max_tokens" not in fake_llm.requests[-1]
    assert fake_llm.requests[-1]["max_completion_tokens"] == CONFIG.max_tokens


def test_adapts_to_models_that_reject_temperature(fake_llm):
    fake_llm.reject_param = "temperature"
    assert chat().content == fake_llm.answer
    assert "temperature" not in fake_llm.requests[-1]


def test_missing_usage_is_estimated_and_flagged(fake_llm):
    fake_llm.report_usage = False
    result = chat()
    assert result.usage_reported is False
    assert result.prompt_tokens > 0 and result.completion_tokens > 0


def test_tool_calls_are_parsed(fake_llm):
    tool = {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}}
    result = chat(tools=[tool])
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": fake_llm.search_query}


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": 1} Hope that helps.') == {"a": 1}
    assert parse_json_object("not json") is None
    assert parse_json_object("[1, 2]") is None
