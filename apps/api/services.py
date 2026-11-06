from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from integrations.tavily.client import TavilyClient
from integrations.tavily.normalizer import normalize_evidence_items
from packages.claim_engine.extractor import AtomicClaim, extract_atomic_claims
from packages.contracts.database import (
    AgentManifestDB,
    ClaimDB,
    ClaimHistoryDB,
    EvidenceItemDB,
    ProviderUsageDB,
    VerificationRunDB,
    VerificationScoreDB,
)
from packages.contracts.schemas import ClaimResult, Evidence, VerifyRequest, VerifyResponse
from packages.contracts.settings import get_settings
from packages.evidence_engine.stancer import (
    infer_stance,
    outdated_terms_in_claim,
    source_quality_score,
    support_confidence,
)
from packages.scoring_engine.scoring import (
    UNCHECKED_FRESHNESS,
    UNCHECKED_SOURCE_QUALITY,
    UNCHECKED_SUPPORT,
    freshness_score,
    truth_score,
)
from packages.temporal_engine.classifier import VolatilityDecision, classify_claim

logger = logging.getLogger(__name__)

PROBLEM_STATUSES = {"STALE", "CONTRADICTED"}
CLEAN_STATUSES = {"VERIFIED", "STABLE"}


def new_id(prefix: str, length: int = 12) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:length]}"


async def verify_text(request: VerifyRequest, session, request_id: str | None = None, trajectory_id: str | None = None) -> VerifyResponse:
    settings = get_settings()
    start = datetime.now(timezone.utc)
    verification_id = new_id("vg")
    req_id = request_id or new_id("req")

    max_claims = max(0, min(request.options.max_claims, settings.max_claims_per_request))
    extracted = extract_atomic_claims(request.text, max_claims=max_claims)
    decisions = [classify_claim(c.text) for c in extracted]

    tavily = TavilyClient()
    live_indexes = [i for i, d in enumerate(decisions) if d.verify_live]
    # Run all live searches concurrently instead of one after another.
    search_results = await asyncio.gather(
        *(tavily.search(extracted[i].text) for i in live_indexes), return_exceptions=True
    )
    payload_by_index = dict(zip(live_indexes, search_results))

    claim_results: list[ClaimResult] = []
    usage = {"tavily_search_calls": 0, "cache_hits": 0, "search_errors": 0, "credits_used": 0, "evidence_items_scored": 0}

    for idx, (claim, decision) in enumerate(zip(extracted, decisions)):
        if not decision.verify_live:
            claim_results.append(_unchecked_claim(claim, decision))
            continue

        payload = payload_by_index[idx]
        usage["tavily_search_calls"] += 1
        if isinstance(payload, BaseException):
            usage["search_errors"] += 1
            logger.warning("Tavily search failed for claim %r: %s", claim.text, payload)
            claim_results.append(_search_failed_claim(claim, decision, payload, request))
            continue

        usage["cache_hits"] += int(bool(payload.get("_cache_hit")))
        usage["credits_used"] += int(payload.get("_credits", 0))
        evidence_rows = normalize_evidence_items(payload)
        usage["evidence_items_scored"] += len(evidence_rows)
        claim_results.append(_score_live_claim(claim, decision, evidence_rows, request))

    t_score = round(truth_score([_score_row(c) for c in claim_results]), 2)
    overall = _overall_decision(claim_results)

    created = datetime.now(timezone.utc)
    response = VerifyResponse(
        verification_id=verification_id,
        status="completed",
        overall_decision=overall,
        truth_score=t_score,
        created_at=created,
        claims=claim_results,
        policy={"mode": request.mode, "recommended_action": _recommended_action(request.mode, overall)},
        usage={**usage, "offline_mode": tavily.offline},
    )

    if request.options.persist:
        _persist(session, response, request, req_id, trajectory_id, start, usage)

    return response


def _unchecked_claim(claim: AtomicClaim, decision: VolatilityDecision) -> ClaimResult:
    return ClaimResult(
        claim_id=new_id("clm"),
        text=claim.text,
        offsets=[claim.start, claim.end],
        category=decision.category,
        volatility=decision.volatility,
        time_sensitive=decision.time_sensitive,
        status="STABLE",
        freshness_score=UNCHECKED_FRESHNESS,
        source_quality=UNCHECKED_SOURCE_QUALITY,
        support_confidence=UNCHECKED_SUPPORT,
        conflict_level="low",
        reason="Claim classified as stable or opinion; live verification skipped.",
    )


def _search_failed_claim(claim: AtomicClaim, decision: VolatilityDecision, error: BaseException, request: VerifyRequest) -> ClaimResult:
    detail = f"HTTP {error.response.status_code}" if isinstance(error, httpx.HTTPStatusError) else type(error).__name__
    return ClaimResult(
        claim_id=new_id("clm"),
        text=claim.text,
        offsets=[claim.start, claim.end],
        category=decision.category,
        volatility=decision.volatility,
        time_sensitive=decision.time_sensitive,
        status="UNVERIFIED",
        freshness_score=0.0,
        source_quality=0.0,
        support_confidence=0.0,
        conflict_level="low",
        reason=f"Live search failed ({detail}); claim could not be verified.",
        suggested_patch="Retry verification or cite a primary source." if request.options.generate_patch else "",
    )


def _score_live_claim(claim: AtomicClaim, decision: VolatilityDecision, evidence_rows: list[dict[str, Any]], request: VerifyRequest) -> ClaimResult:
    evidences: list[Evidence] = []
    supporting: list[tuple[float, float, float]] = []
    contradicting: list[tuple[float, float, float]] = []
    replacements: dict[str, str | None] = {}

    for er in evidence_rows:
        stance = infer_stance(claim.text, er["excerpt"])
        q = source_quality_score(er["url"])
        conf = support_confidence(claim.text, er["excerpt"], stance)
        fresh = freshness_score(er.get("published_at"), er["retrieved_at"], decision.half_life_days)

        if stance == "supports":
            supporting.append((conf, fresh, q))
        elif stance == "contradicts":
            contradicting.append((conf, fresh, q))
            for term in outdated_terms_in_claim(claim.text, er["excerpt"]):
                replacements.setdefault(term.old, term.new)

        evidences.append(
            Evidence(
                url=er["url"],
                title=er["title"],
                publisher=er["publisher"],
                published_at=er.get("published_at"),
                retrieved_at=er["retrieved_at"],
                stance=stance,
                excerpt=er["excerpt"],
            )
        )

    if contradicting and not supporting:
        status = "STALE" if replacements else "CONTRADICTED"
        reason = (
            "Current sources mark terms in this claim as outdated: " + ", ".join(sorted(replacements)) + "."
            if replacements
            else "Retrieved evidence contradicts this claim."
        )
        patch = _replacement_patch(replacements) or "Review against current primary source and update this statement."
        decisive = contradicting
    elif contradicting and supporting:
        status = "NEEDS_REVIEW"
        reason = "Evidence is mixed across sources; claim needs manual confirmation."
        patch = "Add uncertainty and cite both sources with dates."
        decisive = supporting + contradicting
    elif supporting:
        status = "VERIFIED"
        reason = "Retrieved evidence supports the claim."
        patch = ""
        decisive = supporting
    else:
        status = "UNVERIFIED"
        reason = "No decisive supporting evidence was found." if evidence_rows else "No evidence was found for this claim."
        patch = "Cite a primary source or avoid asserting this as a fact."
        decisive = []

    # Scores come only from evidence that actually decided the status, not from irrelevant hits.
    best_support = max((d[0] for d in decisive), default=25.0)
    best_fresh = max((d[1] for d in decisive), default=35.0)
    best_quality = max((d[2] for d in decisive), default=35.0)

    return ClaimResult(
        claim_id=new_id("clm"),
        text=claim.text,
        offsets=[claim.start, claim.end],
        category=decision.category,
        volatility=decision.volatility,
        time_sensitive=decision.time_sensitive,
        status=status,
        freshness_score=round(best_fresh, 2),
        source_quality=round(best_quality, 2),
        support_confidence=round(best_support, 2),
        conflict_level="high" if (supporting and contradicting) else "low",
        reason=reason,
        suggested_patch=patch if request.options.generate_patch else "",
        evidence=evidences if request.options.include_evidence else [],
    )


def _replacement_patch(replacements: dict[str, str | None]) -> str:
    parts = []
    for old, new in sorted(replacements.items()):
        parts.append(f"replace `{old}` with `{new}`" if new else f"`{old}` is deprecated; use its current replacement")
    return ("Per current docs: " + "; ".join(parts) + ".") if parts else ""


def _score_row(claim: ClaimResult) -> dict[str, Any]:
    return {
        "status": claim.status,
        "freshness_score": claim.freshness_score,
        "source_quality": claim.source_quality,
        "support_confidence": claim.support_confidence,
    }


def _overall_decision(claims: list[ClaimResult]) -> str:
    if not claims:
        return "no_claims"
    statuses = {c.status for c in claims}
    if statuses & PROBLEM_STATUSES:
        return "stale_detected"
    if statuses <= CLEAN_STATUSES:
        return "clean"
    return "needs_review"


def _recommended_action(mode: str, overall: str) -> str:
    if overall in {"clean", "no_claims"}:
        return "return"
    if mode == "strict" and overall == "stale_detected":
        return "block"
    return "return_with_warning"


def _persist(session, response: VerifyResponse, request: VerifyRequest, req_id: str, trajectory_id: str | None, start: datetime, usage: dict[str, int]) -> None:
    session.add(
        VerificationRunDB(
            id=response.verification_id,
            trajectory_id=trajectory_id,
            request_id=req_id,
            mode=request.mode,
            model=request.context.model_used or "",
            status="completed",
            truth_score=response.truth_score,
            created_at=response.created_at,
            latency_ms=int((response.created_at - start).total_seconds() * 1000),
        )
    )

    for claim in response.claims:
        session.add(
            ClaimDB(
                id=claim.claim_id,
                run_id=response.verification_id,
                normalized_text=claim.text,
                offsets_json=json.dumps(claim.offsets),
                category=claim.category,
                volatility=claim.volatility,
                status=claim.status,
                suggested_patch=claim.suggested_patch,
            )
        )
        session.add(
            VerificationScoreDB(
                id=new_id("scr"),
                claim_id=claim.claim_id,
                freshness=claim.freshness_score,
                source_quality=claim.source_quality,
                support_confidence=claim.support_confidence,
                conflict_level=claim.conflict_level,
            )
        )
        for ev in claim.evidence:
            session.add(
                EvidenceItemDB(
                    id=new_id("ev"),
                    claim_id=claim.claim_id,
                    url=ev.url,
                    title=ev.title,
                    publisher=ev.publisher,
                    published_at=ev.published_at,
                    retrieved_at=ev.retrieved_at,
                    excerpt=ev.excerpt,
                    stance=ev.stance,
                )
            )

    session.add(
        ProviderUsageDB(
            id=new_id("pu"),
            run_id=response.verification_id,
            provider="tavily",
            operation="search",
            request_count=usage["tavily_search_calls"] - usage["cache_hits"] - usage["search_errors"],
            credits_or_tokens=float(usage["credits_used"]),
            latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
        )
    )
    session.commit()


def hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def manifest_to_fingerprint(manifest: AgentManifestDB) -> dict[str, Any]:
    return {
        "agent_id": manifest.id,
        "name": manifest.name,
        "framework": manifest.framework,
        "model_name": manifest.model_name,
        "temperature": manifest.temperature,
        "top_p": manifest.top_p,
        "max_tokens": manifest.max_tokens,
        "prompt_hash": manifest.prompt_hash,
    }


def log_claim_reverification(session, claim_id: str, old_status: str, new_status: str, reason: str) -> None:
    session.add(
        ClaimHistoryDB(
            id=new_id("ch"),
            claim_id=claim_id,
            previous_status=old_status,
            new_status=new_status,
            reason=reason,
        )
    )
    session.commit()
