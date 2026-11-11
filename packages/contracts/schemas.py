from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class MCPServer(BaseModel):
    name: str
    transport: str = "stdio"
    tools_exposed: list[str] = Field(default_factory=list)


AGENT_PROFILES = ("parametric_baseline", "unbounded_deep_research", "disciplined_verifier")
AgentProfile = Literal["parametric_baseline", "unbounded_deep_research", "disciplined_verifier"]
NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.\-]{1,63}$"


class LLMOverride(BaseModel):
    """Pick an OpenAI-compatible model for one request.

    Give a registered/built-in ``provider`` name, or spell the connection out with ``base_url`` +
    ``api_key``. Anything left empty falls back to the provider's (or the server's) defaults.
    """

    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""

    @model_validator(mode="after")
    def _base_url_needs_model(self) -> "LLMOverride":
        if self.base_url and not self.model:
            raise ValueError("model is required when base_url is given")
        if self.base_url and not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return self


class ProviderRegisterRequest(BaseModel):
    name: str = Field(pattern=NAME_PATTERN)
    base_url: str = Field(min_length=8, max_length=500, pattern=r"^https?://")
    api_key: str = ""
    default_model: str = ""


class ProviderTestRequest(BaseModel):
    model: str = ""


class ModelConfig(BaseModel):
    # Name of a registered or built-in provider ("openai", "groq", "my-proxy", ...) or "default".
    provider: str = "default"
    model_name: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(default=2048, ge=1, le=32000)
    # Optional inline OpenAI-compatible connection, used instead of a named provider.
    base_url: str = ""
    api_key: str = ""


class PromptsConfig(BaseModel):
    system_prompt: str = ""
    # "{{prompt}}" is replaced with the task text.
    instruction_template: str = "{{prompt}}"


class CapabilitiesConfig(BaseModel):
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    native_tools: list[str] = Field(default_factory=list)


class AgentRegisterRequest(BaseModel):
    agent_id: str = Field(pattern=NAME_PATTERN)
    name: str
    framework: str = "custom"
    framework_version: str = ""
    # An http(s) URL makes this an external agent: each task is POSTed there. Leave empty to run
    # one of the built-in reference profiles with the model configured below.
    endpoint_or_command: str = ""
    # Defaults to agent_id when that is a profile name, otherwise parametric_baseline.
    profile: AgentProfile | None = None
    model_cfg: ModelConfig = Field(alias="model_config")
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    capabilities: CapabilitiesConfig = Field(default_factory=CapabilitiesConfig)


class AgentSummary(BaseModel):
    agent_id: str
    name: str
    framework: str
    model_name: str
    temperature: float
    top_p: float
    max_tokens: int
    prompt_hash: str


class VerifyContext(BaseModel):
    domain: str = "software"
    locale: str = "en"
    run_id: str | None = None
    agent_id: str | None = None
    model_used: str | None = None


class VerifyOptions(BaseModel):
    # Also capped server-side by MAX_CLAIMS_PER_REQUEST to protect Tavily credits.
    max_claims: int = Field(default=10, ge=1, le=50)
    include_evidence: bool = True
    generate_patch: bool = True
    persist: bool = True
    # Judge stance with an LLM when one is configured (heuristics otherwise).
    llm_judge: bool = True
    # Model to use for the judge on this request; defaults to the server's default LLM.
    llm: LLMOverride | None = None


class VerifyRequest(BaseModel):
    text: str = Field(max_length=20000)
    mode: Literal["advisory", "strict"] = "advisory"
    context: VerifyContext = Field(default_factory=VerifyContext)
    options: VerifyOptions = Field(default_factory=VerifyOptions)


class Evidence(BaseModel):
    url: str
    title: str
    publisher: str
    published_at: datetime | None = None
    retrieved_at: datetime
    stance: Literal["supports", "contradicts", "contextual", "irrelevant"]
    excerpt: str


class ClaimResult(BaseModel):
    claim_id: str
    text: str
    offsets: list[int]
    category: str
    volatility: str
    time_sensitive: bool
    status: Literal[
        "VERIFIED",
        "STALE",
        "CONTRADICTED",
        "UNVERIFIED",
        "STABLE",
        "NEEDS_REVIEW",
    ]
    freshness_score: float
    source_quality: float
    support_confidence: float
    conflict_level: str
    reason: str
    suggested_patch: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class VerifyResponse(BaseModel):
    verification_id: str
    status: str = "completed"
    overall_decision: str
    truth_score: float
    created_at: datetime
    claims: list[ClaimResult]
    policy: dict[str, Any]
    usage: dict[str, Any]


class BenchmarkRunRequest(BaseModel):
    agent_ids: list[str] = Field(min_length=1, max_length=20)
    suite_id: str = "temporal_traps_v1"
    limit_tasks: int | None = Field(default=None, ge=1)
    # True: return 202 immediately and poll GET /benchmarks/{id}. False: wait for the result.
    background: bool = False
    # Model used to judge the agents' claims; defaults to the server's default LLM.
    judge_llm: LLMOverride | None = None


class BenchmarkAgentScore(BaseModel):
    agent_id: str
    model_name: str = ""
    truth_score: float
    eff_score: float
    chronos_score: float
    # Share of the task's expected focus terms the answer mentions. Informational: not part of ChronosScore.
    focus_coverage: float = 0.0
    tokens_used: int
    latency_ms: int
    tasks_failed: int = 0
    # "provider_usage": token counts reported by the LLM API. "estimated": derived from output length.
    telemetry_source: str = "estimated"


class BenchmarkRunResponse(BaseModel):
    benchmark_id: str
    status: str
    suite_id: str = ""
    tasks_total: int = 0
    tasks_done: int = 0
    leaderboard: list[BenchmarkAgentScore] = Field(default_factory=list)
    # "provider_usage" | "estimated" | "mixed" across the agents in this run.
    telemetry_source: str = "estimated"
    error: str = ""
