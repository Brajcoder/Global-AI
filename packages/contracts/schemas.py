from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MCPServer(BaseModel):
    name: str
    transport: str = "stdio"
    tools_exposed: list[str] = Field(default_factory=list)


class ModelConfig(BaseModel):
    provider: str
    model_name: str
    temperature: float = 0.1
    top_p: float = 0.95
    max_tokens: int = 2048


class PromptsConfig(BaseModel):
    system_prompt: str
    instruction_template: str


class CapabilitiesConfig(BaseModel):
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    native_tools: list[str] = Field(default_factory=list)


class AgentRegisterRequest(BaseModel):
    agent_id: str
    name: str
    framework: str
    framework_version: str = ""
    endpoint_or_command: str = ""
    model_cfg: ModelConfig = Field(alias="model_config")
    prompts: PromptsConfig
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
    max_claims: int = 10
    include_evidence: bool = True
    generate_patch: bool = True
    persist: bool = True


class VerifyRequest(BaseModel):
    text: str
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
    agent_ids: list[str]
    suite_id: str = "temporal_traps_v1"
    limit_tasks: int | None = None


class BenchmarkAgentScore(BaseModel):
    agent_id: str
    truth_score: float
    eff_score: float
    chronos_score: float
    tokens_used: int
    latency_ms: int


class BenchmarkRunResponse(BaseModel):
    benchmark_id: str
    status: str
    leaderboard: list[BenchmarkAgentScore]
