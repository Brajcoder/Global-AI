from __future__ import annotations


from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from apps.api.security import require_api_key
from apps.api.services import log_claim_reverification, verify_text
from packages.contracts.database import (
    ClaimDB,
    ClaimHistoryDB,
    EvidenceItemDB,
    VerificationRunDB,
    VerificationScoreDB,
    session_dep,
)
from packages.contracts.schemas import VerifyRequest

router = APIRouter(prefix="/api/v1", tags=["verification"])


@router.post("/verify", dependencies=[Depends(require_api_key)])
async def verify(payload: VerifyRequest, session: Session = Depends(session_dep)):
    return await verify_text(payload, session)


@router.get("/verifications")
def list_verifications(limit: int = 20, session: Session = Depends(session_dep)):
    """Most recent verification runs, newest first."""
    limit = max(1, min(limit, 100))
    runs = session.exec(select(VerificationRunDB).order_by(VerificationRunDB.created_at.desc()).limit(limit)).all()
    return [r.model_dump() for r in runs]


@router.get("/verification/{verification_id}")
def get_verification(verification_id: str, session: Session = Depends(session_dep)):
    run = session.get(VerificationRunDB, verification_id)
    if not run:
        raise HTTPException(status_code=404, detail="Verification run not found")

    claims = session.exec(select(ClaimDB).where(ClaimDB.run_id == verification_id)).all()
    claim_ids = [c.id for c in claims]
    evidence: dict[str, list[dict]] = {cid: [] for cid in claim_ids}
    scores: dict[str, dict] = {}
    if claim_ids:
        for ev in session.exec(select(EvidenceItemDB).where(EvidenceItemDB.claim_id.in_(claim_ids))).all():
            evidence[ev.claim_id].append(ev.model_dump(exclude={"id", "claim_id"}))
        for sc in session.exec(select(VerificationScoreDB).where(VerificationScoreDB.claim_id.in_(claim_ids))).all():
            scores[sc.claim_id] = sc.model_dump(exclude={"id", "claim_id"})
    return {
        "verification": run.model_dump(),
        "claims": [{**c.model_dump(), "scores": scores.get(c.id, {}), "evidence": evidence[c.id]} for c in claims],
    }


@router.get("/claims/{claim_id}/history")
def claim_history(claim_id: str, session: Session = Depends(session_dep)):
    rows = session.exec(select(ClaimHistoryDB).where(ClaimHistoryDB.claim_id == claim_id)).all()
    return [r.model_dump() for r in rows]


@router.post("/claims/{claim_id}/reverify", dependencies=[Depends(require_api_key)])
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
