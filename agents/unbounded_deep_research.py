from __future__ import annotations

from agents.base import ANSWER_STYLE, Profile

PROFILE = Profile(
    name="unbounded_deep_research",
    instructions=(
        "You are a deep-research agent with a web_search tool. Research the question exhaustively: "
        "search every angle, follow up on anything uncertain and cross-check sources before answering. "
        "Prefer many searches over few. " + ANSWER_STYLE
    ),
    offline_output=(
        "After extensive exploration across multiple references, Pydantic V2 recommends model_dump() and field_validator. "
        "However, some community posts still mention dict(). Current model pricing can vary by provider and date. "
    ),
    uses_search=True,
    max_steps=10,
    max_searches=8,
)
