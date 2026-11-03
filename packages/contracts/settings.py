from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    bifrost_gateway_url: str = "http://localhost:8080/v1"
    default_llm_provider: str = "groq"
    default_llm_model: str = "llama-3.3-70b-versatile"

    tavily_api_key: str = ""
    tavily_cache_enabled: bool = True
    tavily_cache_db_path: str = "./data/tavily_cache.sqlite"

    database_url: str = "sqlite:///./data/chronos_veritygate.db"

    verification_default_mode: str = "advisory"
    max_claims_per_request: int = 10
    max_searches_per_claim: int = 3
    max_agent_steps_per_task: int = 10
    request_timeout_seconds: int = 20


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
