from __future__ import annotations

from packages.contracts.schemas import AgentRegisterRequest


def validate_manifest(payload: dict) -> AgentRegisterRequest:
    return AgentRegisterRequest(**payload)
