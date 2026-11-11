from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from integrations.llm.client import LLMConfig

SearchFn = Callable[[str], Awaitable[dict[str, Any]]]

ANSWER_STYLE = (
    "Write the final answer as plain prose in at most 8 short, self-contained factual sentences. "
    "Name concrete APIs, versions, prices and dates where relevant. No markdown, lists or code blocks."
)


@dataclass(frozen=True)
class Profile:
    """A reference agent behaviour. The model, prompts and sampling come from the agent manifest."""

    name: str
    instructions: str
    # Answer returned when no LLM is configured, so the platform still runs end to end offline.
    offline_output: str
    uses_search: bool = False
    # Hard caps; the step cap is also limited by MAX_AGENT_STEPS_PER_TASK.
    max_steps: int = 1
    max_searches: int = 0


@dataclass
class AgentContext:
    llm: LLMConfig | None
    search: SearchFn
    system_prompt: str = ""
    instruction_template: str = "{{prompt}}"
    max_steps: int = 10


@dataclass
class AgentRun:
    output: str
    steps: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    loop_count: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    model: str = ""
    # "provider_usage": token counts came from the LLM API. "estimated": derived from text length.
    telemetry_source: str = "estimated"
    error: str = ""

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
