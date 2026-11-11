from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session

from apps.api.security import require_api_key
from integrations.llm import client as llm_client
from integrations.llm.client import LLMError
from integrations.llm.providers import default_llm, describe_providers, mask_key, resolve_llm
from packages.contracts.database import LLMProviderDB, session_dep
from packages.contracts.schemas import ProviderRegisterRequest, ProviderTestRequest

router = APIRouter(prefix="/api/v1", tags=["providers"])


def _public(row: LLMProviderDB) -> dict:
    return {
        "name": row.id,
        "kind": "registered",
        "base_url": row.base_url,
        "default_model": row.default_model,
        "api_key": mask_key(row.api_key),
        "configured": True,
    }


@router.post("/providers", dependencies=[Depends(require_api_key)])
def register_provider(payload: ProviderRegisterRequest, session: Session = Depends(session_dep)):
    """Register (or update) an OpenAI-compatible endpoint: a name, its base URL, a key and a default model."""
    name = payload.name.lower()
    row = session.get(LLMProviderDB, name)
    status = "updated" if row else "registered"
    if row is None:
        row = LLMProviderDB(id=name, base_url=payload.base_url.rstrip("/"))
    row.base_url = payload.base_url.rstrip("/")
    row.default_model = payload.default_model
    # An empty key on update keeps the stored key, so the UI never has to echo secrets back.
    if payload.api_key or status == "registered":
        row.api_key = payload.api_key
    session.add(row)
    session.commit()
    session.refresh(row)
    return {"status": status, "provider": _public(row)}


@router.get("/providers")
def list_providers(session: Session = Depends(session_dep)):
    """Registered providers and built-in presets. API keys are masked and never returned in full."""
    default = default_llm(session)
    return {
        "default": {"provider": default.provider, "model": default.model, "base_url": default.base_url} if default else None,
        "providers": describe_providers(session),
    }


@router.delete("/providers/{name}", dependencies=[Depends(require_api_key)])
def delete_provider(name: str, session: Session = Depends(session_dep)):
    row = session.get(LLMProviderDB, name.lower())
    if not row:
        raise HTTPException(status_code=404, detail="Provider not found")
    session.delete(row)
    session.commit()
    return {"status": "deleted", "name": name.lower()}


@router.post("/providers/{name}/test", dependencies=[Depends(require_api_key)])
async def test_provider(name: str, payload: ProviderTestRequest | None = None, session: Session = Depends(session_dep)):
    """Send a one-line chat request to check the base URL, key and model actually work."""
    model = payload.model if payload else ""
    config = resolve_llm(session, name, model, max_tokens=16, temperature=0.0)
    if config is None or (name.lower() not in {"", "default"} and config.provider != name.lower()):
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{name}' is not usable: it needs a base URL, an API key (if required) and a model name.",
        )
    try:
        result = await llm_client.chat(config, [{"role": "user", "content": "Reply with the single word: ok"}])
    except LLMError as exc:
        return {"ok": False, "provider": config.provider, "model": config.model, "error": str(exc)}
    return {
        "ok": True,
        "provider": config.provider,
        "model": result.model,
        "reply": result.content[:200],
        "latency_ms": result.latency_ms,
        "tokens": result.prompt_tokens + result.completion_tokens,
    }
