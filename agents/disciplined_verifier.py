from __future__ import annotations


def run_task(prompt: str) -> str:
    # Stub agent with a canned answer. The prompt is not echoed: echoed task text was being
    # scored as the agent's own claims. Replace with a real LLM call (see NEXT_STEPS.md).
    del prompt
    return (
        "Using official docs, Pydantic V2 uses model_dump() for serialization and @field_validator for validation hooks. "
        "For live pricing and leadership details, verify against official vendor pricing pages and company filings before finalizing answers. "
    )
