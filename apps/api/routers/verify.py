from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from apps.api.services import log_claim_reverification, verify_text
from packages.contracts.database import ClaimDB, ClaimHistoryDB, VerificationRunDB, session_dep
from packages.contracts.schemas import VerifyRequest

router = APIRouter(prefix="/api/v1", tags=["verification"])


@router.post("/verify")
async def verify(payload: VerifyRequest, session: Session = Depends(session_dep)):
    return await verify_text(payload, session)


@router.get("/verification/{verification_id}")
def get_verification(verification_id: str, session: Session = Depends(session_dep)):
    run = session.get(VerificationRunDB, verification_id)
    if not run:
        raise HTTPException(status_code=404, detail="Verification run not found")

    claims = session.exec(select(ClaimDB).where(ClaimDB.run_id == verification_id)).all()
    return {
        "verification": run.model_dump(),
        "claims": [c.model_dump() for c in claims],
    }


@router.get("/claims/{claim_id}/history")
def claim_history(claim_id: str, session: Session = Depends(session_dep)):
    rows = session.exec(select(ClaimHistoryDB).where(ClaimHistoryDB.claim_id == claim_id)).all()
    return [r.model_dump() for r in rows]


@router.post("/claims/{claim_id}/reverify")
async def claim_reverify(claim_id: str, session: Session = Depends(session_dep)):
    claim = session.get(ClaimDB, claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found")

    payload = VerifyRequest(text=claim.normalized_text)
    result = await verify_text(payload, session, request_id=f"reverify_{claim_id}")

    if not result.claims:
        return {"status": "no_result"}

    new_status = result.claims[0].status
    old_status = claim.status
    claim.status = new_status
    claim.suggested_patch = result.claims[0].suggested_patch
    session.add(claim)
    session.commit()

    if new_status != old_status:
        log_claim_reverification(
            session,
            claim_id=claim_id,
            old_status=old_status,
            new_status=new_status,
            reason="Status changed after live re-verification.",
        )

    return {
        "claim_id": claim_id,
        "previous_status": old_status,
        "new_status": new_status,
        "verification_id": result.verification_id,
        "evidence_count": len(result.claims[0].evidence),
    }
