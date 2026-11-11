from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from agents.runtime import DEFAULT_PROFILE, PROFILES
from apps.api.security import require_api_key
from apps.api.services import hash_prompt, manifest_to_fingerprint
from integrations.llm.providers import mask_key, resolve_llm
from packages.contracts.database import AgentManifestDB, session_dep
from packages.contracts.schemas import AgentRegisterRequest

router = APIRouter(prefix="/api/v1", tags=["agents"])


def execution_mode(manifest: AgentManifestDB, session: Session) -> str:
    """How this agent will run in a benchmark: external | llm | offline."""
    if manifest.endpoint_or_command.startswith(("http://", "https://")):
        return "external"
    config = resolve_llm(session, manifest.provider, manifest.model_name, manifest.base_url, manifest.api_key)
    return "llm" if config else "offline"


@router.post("/agents/register", dependencies=[Depends(require_api_key)])
def register_agent(payload: AgentRegisterRequest, session: Session = Depends(session_dep)):
    existing = session.get(AgentManifestDB, payload.agent_id)
    prompt_hash = hash_prompt(payload.prompts.system_prompt + payload.prompts.instruction_template)
    profile = payload.profile or (payload.agent_id if payload.agent_id in PROFILES else DEFAULT_PROFILE)

    row = AgentManifestDB(
        id=payload.agent_id,
        name=payload.name,
        framework=payload.framework,
        framework_version=payload.framework_version,
        endpoint_or_command=payload.endpoint_or_command.strip(),
        model_name=payload.model_cfg.model_name,
        provider=payload.model_cfg.provider,
        temperature=payload.model_cfg.temperature,
        top_p=payload.model_cfg.top_p,
        max_tokens=payload.model_cfg.max_tokens,
        system_prompt=payload.prompts.system_prompt,
        instruction_template=payload.prompts.instruction_template,
        prompt_hash=prompt_hash,
        mcp_servers_json=json.dumps([s.model_dump() for s in payload.capabilities.mcp_servers]),
        tools_schema_json=json.dumps(payload.capabilities.native_tools),
        profile=profile,
        base_url=payload.model_cfg.base_url.rstrip("/"),
        api_key=payload.model_cfg.api_key,
    )

    if existing:
        # Re-registering without a key keeps the stored one (as long as it targets the same endpoint).
        if not row.api_key and (existing.provider, existing.base_url) == (row.provider, row.base_url):
            row.api_key = existing.api_key
        for key, value in row.model_dump(exclude={"created_at"}).items():
            setattr(existing, key, value)
        row, status = existing, "updated"
    else:
        status = "registered"

    session.add(row)
    session.commit()
    session.refresh(row)
    return {"status": status, "agent": {**manifest_to_fingerprint(row), "execution": execution_mode(row, session)}}


@router.get("/agents")
def list_agents(session: Session = Depends(session_dep)):
    rows = session.exec(select(AgentManifestDB).order_by(AgentManifestDB.id)).all()
    return [{**manifest_to_fingerprint(r), "execution": execution_mode(r, session)} for r in rows]


@router.get("/agents/{agent_id}")
def get_agent(agent_id: str, session: Session = Depends(session_dep)):
    row = session.get(AgentManifestDB, agent_id)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {
        **manifest_to_fingerprint(row),
        "execution": execution_mode(row, session),
        "framework_version": row.framework_version,
        "endpoint_or_command": row.endpoint_or_command,
        "base_url": row.base_url,
        "api_key": mask_key(row.api_key),  # never returned in full
        "system_prompt": row.system_prompt,
        "instruction_template": row.instruction_template,
        "mcp_servers": json.loads(row.mcp_servers_json),
        "native_tools": json.loads(row.tools_schema_json),
    }


@router.delete("/agents/{agent_id}", dependencies=[Depends(require_api_key)])
def delete_agent(agent_id: str, session: Session = Depends(session_dep)):
    row = session.get(AgentManifestDB, agent_id)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    session.delete(row)
    session.commit()
    return {"status": "deleted", "agent_id": agent_id}
