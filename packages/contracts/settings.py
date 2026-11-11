from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, so file paths work no matter which directory the server is started from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # --- Default LLM (any OpenAI-compatible endpoint) -------------------------------------------
    # Set these three and every agent / the stance judge can call a model. LLM_BASE_URL wins over
    # DEFAULT_LLM_PROVIDER when both are set.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Preset route: pick a built-in provider name and supply its key below.
    default_llm_provider: str = "groq"
    default_llm_model: str = "llama-3.3-70b-versatile"
    openai_api_key: str = ""
    groq_api_key: str = ""
    gemini_api_key: str = ""
    openrouter_api_key: str = ""
    together_api_key: str = ""
    deepseek_api_key: str = ""
    mistral_api_key: str = ""
    bifrost_gateway_url: str = "http://localhost:8080/v1"
    bifrost_api_key: str = ""

    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2
    # Use the default LLM to judge claim-vs-evidence stance. Falls back to heuristics when no LLM
    # is configured or the call fails.
    llm_judge_enabled: bool = True

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

    # --- Security ----------------------------------------------------------------------------
    # When set, every write endpoint (POST/DELETE) requires the header "X-API-Key: <value>".
    api_key: str = ""
    # Comma-separated list of allowed browser origins, or "*" for any.
    cors_allow_origins: str = "*"


def resolve_project_path(path: str) -> Path:
    """Resolve a relative path against the project root instead of the current directory."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
