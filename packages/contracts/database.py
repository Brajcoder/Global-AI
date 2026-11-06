from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlmodel import Field, Session, SQLModel, create_engine

from packages.contracts.settings import get_settings, resolve_project_path


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AgentManifestDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    framework: str
    framework_version: str = ""
    endpoint_or_command: str = ""
    model_name: str
    provider: str = ""
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    system_prompt: str = ""
    instruction_template: str = ""
    prompt_hash: str = ""
    mcp_servers_json: str = "[]"
    tools_schema_json: str = "[]"
    created_at: datetime = Field(default_factory=utcnow)


class BenchmarkRunDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    suite_id: str = "temporal_traps_v1"
    status: str = "running"
    started_at: datetime = Field(default_factory=utcnow)
    completed_at: Optional[datetime] = None


class AgentTrajectoryDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    benchmark_id: str
    agent_id: str
    task_id: str
    observed_model: str
    observed_params_json: str
    observed_tools_json: str
    invoked_tool_calls_json: str = "[]"
    step_count: int = 1
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    loop_count: int = 0
    raw_output: str = ""
    eff_score: float = 0.0


class VerificationRunDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    trajectory_id: Optional[str] = None
    request_id: str
    mode: str = "advisory"
    model: str = ""
    status: str = "completed"
    truth_score: float = 0.0
    created_at: datetime = Field(default_factory=utcnow)
    latency_ms: int = 0


class ClaimDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    run_id: str
    normalized_text: str
    offsets_json: str
    category: str
    volatility: str
    status: str
    suggested_patch: str = ""
    first_seen_at: datetime = Field(default_factory=utcnow)


class EvidenceItemDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    claim_id: str
    url: str
    title: str
    publisher: str
    published_at: Optional[datetime] = None
    retrieved_at: datetime = Field(default_factory=utcnow)
    excerpt: str = ""
    stance: str = "contextual"


class VerificationScoreDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    claim_id: str
    freshness: float = 0.0
    source_quality: float = 0.0
    support_confidence: float = 0.0
    conflict_level: str = "low"


class ClaimHistoryDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    claim_id: str
    previous_status: str
    new_status: str
    changed_at: datetime = Field(default_factory=utcnow)
    reason: str = ""


class ProviderUsageDB(SQLModel, table=True):
    id: str = Field(primary_key=True)
    run_id: str
    provider: str
    operation: str
    request_count: int = 0
    credits_or_tokens: float = 0.0
    latency_ms: int = 0


settings = get_settings()


def _prepare_database_url(url: str) -> str:
    """Make relative SQLite paths project-relative and create the parent folder.

    SQLite cannot create missing parent folders, so a fresh clone without ./data would fail to start.
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        return url
    raw_path = url[len(prefix):]
    if not raw_path or raw_path == ":memory:":
        return url
    db_path = resolve_project_path(raw_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"{prefix}{db_path.as_posix()}"


database_url = _prepare_database_url(settings.database_url)
connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
engine = create_engine(database_url, echo=False, connect_args=connect_args)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_session() -> Session:
    session = Session(engine)
    try:
        yield session
    finally:
        session.close()


def session_dep():
    """FastAPI dependency that yields one DB session per request."""
    with get_session() as session:
        yield session
