from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, so file paths work no matter which directory the server is started from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    bifrost_gateway_url: str = "http://localhost:8080/v1"
    default_llm_provider: str = "groq"
    default_llm_model: str = "llama-3.3-70b-versatile"

    tavily_api_key: str = ""
    tavily_cache_enabled: bool = True
    tavily_cache_db_path: str = "./data/tavily_cache.sqlite"
    # Cached search results older than this are ignored, so "live" verification stays live.
    tavily_cache_ttl_hours: int = 24
    # "basic" costs 1 Tavily credit per search, "advanced" costs 2.
    tavily_search_depth: str = "advanced"
    tavily_max_results: int = 3

    database_url: str = "sqlite:///./data/chronos_veritygate.db"

    verification_default_mode: str = "advisory"
    max_claims_per_request: int = 10
    max_searches_per_claim: int = 3
    max_agent_steps_per_task: int = 10
    request_timeout_seconds: int = 20


def resolve_project_path(path: str) -> Path:
    """Resolve a relative path against the project root instead of the current directory."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
