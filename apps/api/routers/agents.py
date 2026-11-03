from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from apps.api.services import hash_prompt, manifest_to_fingerprint
from packages.contracts.database import AgentManifestDB, get_session
from packages.contracts.schemas import AgentRegisterRequest

router = APIRouter(prefix="/api/v1", tags=["agents"])


def _session_dep():
    with get_session() as session:
        yield session


@router.post("/agents/register")
def register_agent(payload: AgentRegisterRequest, session: Session = Depends(_session_dep)):
    existing = session.get(AgentManifestDB, payload.agent_id)
    prompt_hash = hash_prompt(payload.prompts.system_prompt + payload.prompts.instruction_template)

    row = AgentManifestDB(
        id=payload.agent_id,
        name=payload.name,
        framework=payload.framework,
        framework_version=payload.framework_version,
        endpoint_or_command=payload.endpoint_or_command,
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
    )

    if existing:
        for key, value in row.model_dump().items():
            setattr(existing, key, value)
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return {"status": "updated", "agent": manifest_to_fingerprint(existing)}

    session.add(row)
    session.commit()
    session.refresh(row)
    return {"status": "registered", "agent": manifest_to_fingerprint(row)}


@router.get("/agents")
def list_agents(session: Session = Depends(_session_dep)):
    rows = session.exec(select(AgentManifestDB)).all()
    return [manifest_to_fingerprint(r) for r in rows]


@router.get("/agents/{agent_id}")
def get_agent(agent_id: str, session: Session = Depends(_session_dep)):
    row = session.get(AgentManifestDB, agent_id)
    if not row:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {
        **manifest_to_fingerprint(row),
        "framework_version": row.framework_version,
        "endpoint_or_command": row.endpoint_or_command,
        "mcp_servers": json.loads(row.mcp_servers_json),
        "native_tools": json.loads(row.tools_schema_json),
    }
