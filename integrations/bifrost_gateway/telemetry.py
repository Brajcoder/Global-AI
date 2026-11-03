from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WireTelemetry:
    model_name: str
    temperature: float
    top_p: float
    max_tokens: int
    prompt_tokens: int
    completion_tokens: int
    step_count: int
    latency_ms: int
    loop_count: int


def estimate_telemetry(output_text: str, model_name: str, temperature: float, top_p: float, max_tokens: int) -> WireTelemetry:
    # Lightweight local estimator used when no live Bifrost interception is wired.
    word_count = max(1, len(output_text.split()))
    prompt_tokens = int(word_count * 1.2)
    completion_tokens = int(word_count * 1.6)
    step_count = max(1, min(6, word_count // 40 + 1))
    latency_ms = 800 + step_count * 350
    loop_count = 1 if "retry" in output_text.lower() else 0

    return WireTelemetry(
        model_name=model_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        step_count=step_count,
        latency_ms=latency_ms,
        loop_count=loop_count,
    )
