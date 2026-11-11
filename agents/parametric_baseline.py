from __future__ import annotations

from agents.base import ANSWER_STYLE, Profile

PROFILE = Profile(
    name="parametric_baseline",
    instructions="You are a technical assistant. Answer from your own knowledge only; you have no tools. " + ANSWER_STYLE,
    offline_output=(
        "Based on prior known documentation, Pydantic serialization uses .dict() and @validator. "
        "OpenAI GPT-4o pricing is around $5 per 1M input tokens. "
    ),
    uses_search=False,
    max_steps=1,
    max_searches=0,
)
