"""Resolve "which model do I call?" into a concrete base URL + API key + model name.

Lookup order for a provider name:

1. An inline ``base_url`` (and ``api_key``) given on the request or the agent manifest.
2. A provider registered at runtime through ``POST /api/v1/providers`` (stored in the database).
3. A built-in preset (openai, groq, gemini, ...) whose key is set in the environment.
4. The server default: ``LLM_BASE_URL`` / ``LLM_API_KEY`` / ``LLM_MODEL``, or failing that the
   ``DEFAULT_LLM_PROVIDER`` preset with ``DEFAULT_LLM_MODEL``.

``resolve_llm`` returns ``None`` when nothing usable is configured; callers then run offline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from sqlmodel import select

from integrations.llm.client import LLMConfig
from packages.contracts.database import LLMProviderDB
from packages.contracts.settings import Settings, get_settings

DEFAULT_NAMES = {"", "default"}


@dataclass(frozen=True)
class Preset:
    base_url: str
    key_setting: str
    requires_key: bool = True


PRESETS: dict[str, Preset] = {
    "openai": Preset("https://api.openai.com/v1", "openai_api_key"),
    "groq": Preset("https://api.groq.com/openai/v1", "groq_api_key"),
    "gemini": Preset("https://generativelanguage.googleapis.com/v1beta/openai", "gemini_api_key"),
    "openrouter": Preset("https://openrouter.ai/api/v1", "openrouter_api_key"),
    "together": Preset("https://api.together.xyz/v1", "together_api_key"),
    "deepseek": Preset("https://api.deepseek.com/v1", "deepseek_api_key"),
    "mistral": Preset("https://api.mistral.ai/v1", "mistral_api_key"),
    # Local / self-hosted endpoints do not need a key.
    "ollama": Preset("http://localhost:11434/v1", "", requires_key=False),
    "lmstudio": Preset("http://localhost:1234/v1", "", requires_key=False),
    "bifrost": Preset("", "bifrost_api_key", requires_key=False),
}


def mask_key(key: str) -> str:
    if not key:
        return ""
    return f"{key[:3]}...{key[-4:]}" if len(key) > 10 else "***"


def _preset_connection(name: str, settings: Settings) -> tuple[str, str] | None:
    preset = PRESETS.get(name)
    if preset is None:
        return None
    base_url = settings.bifrost_gateway_url if name == "bifrost" else preset.base_url
    key = getattr(settings, preset.key_setting, "") if preset.key_setting else ""
    if not base_url or (preset.requires_key and not key):
        return None
    return base_url, key


def _base_config(provider: str, base_url: str, api_key: str, model: str, settings: Settings) -> LLMConfig:
    return LLMConfig(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def default_llm(session=None) -> LLMConfig | None:
    """The server-wide default model, or None when no LLM is configured."""
    settings = get_settings()
    if settings.llm_base_url:
        model = settings.llm_model or settings.default_llm_model
        return _base_config("default", settings.llm_base_url, settings.llm_api_key, model, settings) if model else None

    name = settings.default_llm_provider.strip().lower()
    model = settings.llm_model or settings.default_llm_model
    if not name or not model:
        return None
    if session is not None:
        row = session.get(LLMProviderDB, name)
        if row:
            return _base_config(name, row.base_url, row.api_key, row.default_model or model, settings)
    conn = _preset_connection(name, settings)
    return _base_config(name, conn[0], conn[1], model, settings) if conn else None


def resolve_llm(
    session,
    provider: str = "",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
) -> LLMConfig | None:
    settings = get_settings()
    name = (provider or "").strip().lower()
    model = "" if (model or "").strip().lower() == "default" else (model or "").strip()
    config: LLMConfig | None = None

    if base_url:
        if model:
            config = _base_config(name or "custom", base_url, api_key, model, settings)
    elif name not in DEFAULT_NAMES:
        row = session.get(LLMProviderDB, name) if session is not None else None
        if row:
            chosen = model or row.default_model
            if chosen:
                config = _base_config(name, row.base_url, api_key or row.api_key, chosen, settings)
        else:
            conn = _preset_connection(name, settings)
            if api_key and name in PRESETS and not conn:
                # Key supplied inline for a preset that has none in the environment.
                preset_url = settings.bifrost_gateway_url if name == "bifrost" else PRESETS[name].base_url
                conn = (preset_url, api_key) if preset_url else None
            if conn and model:
                config = _base_config(name, conn[0], api_key or conn[1], model, settings)

    if config is None:
        # Unknown or unconfigured provider: use the server default connection. The requested model
        # name is only kept when the caller explicitly asked for the default provider, because a
        # model name from one provider is rarely valid on another.
        config = default_llm(session)
        if config is not None and name in DEFAULT_NAMES and model:
            config = replace(config, model=model)

    if config is None:
        return None
    overrides = {k: v for k, v in {"temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}.items() if v is not None}
    return replace(config, **overrides) if overrides else config


def describe_providers(session) -> list[dict]:
    """Registered providers plus built-in presets, with keys masked. Never returns a raw key."""
    settings = get_settings()
    out: list[dict] = []
    seen: set[str] = set()
    for row in session.exec(select(LLMProviderDB).order_by(LLMProviderDB.id)).all():
        seen.add(row.id)
        out.append(
            {
                "name": row.id,
                "kind": "registered",
                "base_url": row.base_url,
                "default_model": row.default_model,
                "api_key": mask_key(row.api_key),
                "configured": True,
            }
        )
    for name, preset in PRESETS.items():
        if name in seen:
            continue
        conn = _preset_connection(name, settings)
        out.append(
            {
                "name": name,
                "kind": "preset",
                "base_url": settings.bifrost_gateway_url if name == "bifrost" else preset.base_url,
                "default_model": "",
                "api_key": mask_key(conn[1]) if conn else "",
                "configured": conn is not None,
                "requires_key": preset.requires_key,
                "env_var": preset.key_setting.upper(),
            }
        )
    return out
