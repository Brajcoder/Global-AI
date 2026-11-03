from __future__ import annotations


def run_task(prompt: str) -> str:
    return (
        "Based on prior known documentation, Pydantic serialization uses .dict() and @validator. "
        "OpenAI GPT-4o pricing is around $5 per 1M input tokens. "
        f"Task summary: {prompt}"
    )
