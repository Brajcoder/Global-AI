"""Minimal async client for any OpenAI-compatible ``/chat/completions`` endpoint.

Works with OpenAI, Groq, Gemini (OpenAI mode), OpenRouter, Together, DeepSeek, Mistral, Ollama,
LM Studio, vLLM, LiteLLM, Bifrost and anything else that speaks the same wire format: all it needs
is a base URL, an API key and a model name.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Tests install an httpx.MockTransport here so no real network call is ever made.
_transport: httpx.AsyncBaseTransport | None = None


def set_transport(transport: httpx.AsyncBaseTransport | None) -> None:
    global _transport
    _transport = transport


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.1
    top_p: float = 1.0
    max_tokens: int = 1024
    timeout_seconds: float = 60.0
    max_retries: int = 2

    @property
    def chat_url(self) -> str:
        base = self.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass
class LLMResult:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    model: str = ""
    # False when the provider did not report token usage and the counts were estimated.
    usage_reported: bool = False
    raw_message: dict[str, Any] = field(default_factory=dict)


class LLMError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token) for providers that omit usage."""
    return max(1, len(text) // 4) if text else 0


async def chat(
    config: LLMConfig,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str = "auto",
    json_mode: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> LLMResult:
    """Send one chat completion request and return the parsed reply. Raises LLMError on failure."""
    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature if temperature is None else temperature,
        "top_p": config.top_p,
        "max_tokens": config.max_tokens if max_tokens is None else max_tokens,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    started = time.perf_counter()
    data = await _post_adapting(config, headers, body)
    latency_ms = int((time.perf_counter() - started) * 1000)

    try:
        message = data["choices"][0]["message"] or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError(f"{config.provider}: response has no choices[0].message") from exc

    content = _text_content(message.get("content"))
    tool_calls = _parse_tool_calls(message.get("tool_calls"))
    usage = data.get("usage") or {}
    reported = "prompt_tokens" in usage or "completion_tokens" in usage
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    if not reported:
        prompt_tokens = estimate_tokens(json.dumps(messages))
        completion_tokens = estimate_tokens(content) + sum(estimate_tokens(t.raw_arguments) for t in tool_calls)

    return LLMResult(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        model=str(data.get("model") or config.model),
        usage_reported=reported,
        raw_message=message,
    )


async def _post_adapting(config: LLMConfig, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    """Post the request, dropping optional parameters a provider rejects.

    "OpenAI-compatible" endpoints differ at the edges: some models want max_completion_tokens
    instead of max_tokens, some refuse temperature/top_p, some do not know response_format.
    """
    body = dict(body)
    for _ in range(4):
        try:
            return await _post_with_retries(config, headers, body)
        except LLMError as exc:
            detail = str(exc).lower()
            if exc.status_code not in (400, 422):
                raise
            if "max_completion_tokens" in detail and "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens")
            elif "response_format" in detail and "response_format" in body:
                body.pop("response_format")
            elif "top_p" in detail and "top_p" in body:
                body.pop("top_p")
            elif "temperature" in detail and "temperature" in body:
                body.pop("temperature")
            else:
                raise
    return await _post_with_retries(config, headers, body)


async def _post_with_retries(config: LLMConfig, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    last_error: LLMError | None = None
    async with httpx.AsyncClient(timeout=config.timeout_seconds, transport=_transport) as client:
        for attempt in range(config.max_retries + 1):
            if attempt:
                await asyncio.sleep(min(8.0, 0.5 * 2**attempt) if _transport is None else 0)
            try:
                res = await client.post(config.chat_url, headers=headers, json=body)
            except httpx.HTTPError as exc:
                last_error = LLMError(f"{config.provider}: request failed ({type(exc).__name__}: {exc})")
                continue

            if res.status_code == 200:
                try:
                    data = res.json()
                except ValueError as exc:
                    raise LLMError(f"{config.provider}: response was not JSON") from exc
                if not isinstance(data, dict):
                    raise LLMError(f"{config.provider}: unexpected response shape")
                return data

            last_error = LLMError(
                f"{config.provider}: HTTP {res.status_code} from {config.chat_url}: {_error_detail(res)}",
                status_code=res.status_code,
            )
            if res.status_code not in RETRYABLE_STATUS:
                break
    assert last_error is not None
    raise last_error


def _error_detail(res: httpx.Response) -> str:
    try:
        payload = res.json()
    except ValueError:
        return res.text[:300]
    if isinstance(payload, dict):
        err = payload.get("error", payload)
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        return str(err)[:300]
    return str(payload)[:300]


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # some providers return content parts
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _parse_tool_calls(raw: Any) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for i, item in enumerate(raw or []):
        fn = (item or {}).get("function") or {}
        raw_args = fn.get("arguments") or "{}"
        if isinstance(raw_args, dict):
            args, raw_args = raw_args, json.dumps(raw_args)
        else:
            try:
                args = json.loads(raw_args)
            except ValueError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        calls.append(ToolCall(id=str(item.get("id") or f"call_{i}"), name=str(fn.get("name") or ""), arguments=args, raw_arguments=str(raw_args)))
    return calls


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a model reply, tolerating code fences and surrounding prose."""
    if not text:
        return None
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate[4:] if candidate.lower().startswith("json") else candidate
    for attempt in (candidate, candidate[candidate.find("{") : candidate.rfind("}") + 1]):
        try:
            value = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None
