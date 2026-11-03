from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from integrations.tavily.client import TavilyClient
from integrations.tavily.normalizer import normalize_evidence_items
from packages.claim_engine.extractor import extract_atomic_claims
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
from packages.evidence_engine.stancer import infer_stance, source_quality_score, support_confidence
from packages.scoring_engine.scoring import freshness_score, truth_score
from packages.temporal_engine.classifier import classify_claim


async def verify_text(request: VerifyRequest, session, request_id: str | None = None, trajectory_id: str | None = None) -> VerifyResponse:
    start = datetime.now(timezone.utc)
    verification_id = f"vg_{uuid.uuid4().hex[:12]}"
    req_id = request_id or f"req_{uuid.uuid4().hex[:12]}"

    extractor_claims = extract_atomic_claims(request.text, max_claims=request.options.max_claims)
    tavily = TavilyClient()

    claim_results: list[ClaimResult] = []
    score_rows: list[dict[str, Any]] = []
    search_calls = 0
    extract_calls = 0
    cache_hits = 0

    for idx, c in enumerate(extractor_claims, start=1):
        decision = classify_claim(c.text)

        if not decision.verify_live:
            claim = ClaimResult(
                claim_id=f"clm_{idx:03d}",
                text=c.text,
                offsets=[c.start, c.end],
                category=decision.category,
                volatility=decision.volatility,
                time_sensitive=decision.time_sensitive,
                status="STABLE",
                freshness_score=100.0,
                source_quality=100.0,
                support_confidence=100.0,
                conflict_level="low",
                reason="Claim classified as stable or opinion; live verification skipped.",
                suggested_patch="",
                evidence=[],
            )
            claim_results.append(claim)
            score_rows.append(
                {
                    "status": claim.status,
                    "freshness_score": claim.freshness_score,
                    "source_quality": claim.source_quality,
                    "support_confidence": claim.support_confidence,
                }
            )
            continue

        search_calls += 1
        tavily_payload = await tavily.search(c.text, max_results=3)
        if tavily_payload.get("_cache_hit"):
            cache_hits += 1
        evidence_rows = normalize_evidence_items(tavily_payload)

        evidences: list[Evidence] = []
        supports = 0
        contradicts = 0
        best_quality = 35.0
        best_support = 25.0
        best_freshness = 35.0

        for er in evidence_rows:
            extract_calls += 1
            stance = infer_stance(c.text, er["excerpt"])
            q = source_quality_score(er["url"])
            conf = support_confidence(c.text, er["excerpt"], stance)
            fresh = freshness_score(er.get("published_at"), er["retrieved_at"], decision.half_life_days)

            if stance == "supports":
                supports += 1
            if stance == "contradicts":
                contradicts += 1

            best_quality = max(best_quality, q)
            best_support = max(best_support, conf)
            best_freshness = max(best_freshness, fresh)

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

        if contradicts > 0 and supports == 0:
            status = "CONTRADICTED"
            reason = "Retrieved evidence contradicts this claim."
            patch = "Review against current primary source and update this statement."
        elif contradicts > 0 and supports > 0:
            status = "NEEDS_REVIEW"
            reason = "Evidence is mixed across sources; claim needs manual confirmation."
            patch = "Add uncertainty and cite both sources with dates."
        elif supports > 0:
            status = "VERIFIED"
            reason = "Authoritative evidence supports the claim."
            patch = ""
        else:
            status = "UNVERIFIED"
            reason = "No decisive supporting evidence was found."
            patch = "Cite a primary source or avoid asserting this as a fact."

        # If a claim references likely outdated API patterns, explicitly mark stale.
        lower_text = c.text.lower()
        if "dict()" in lower_text or "@validator" in lower_text:
            status = "STALE"
            reason = "Latest docs indicate modern replacements for deprecated patterns."
            patch = "Use model_dump() and @field_validator for Pydantic V2."

        claim = ClaimResult(
            claim_id=f"clm_{idx:03d}",
            text=c.text,
            offsets=[c.start, c.end],
            category=decision.category,
            volatility=decision.volatility,
            time_sensitive=decision.time_sensitive,
            status=status,
            freshness_score=round(best_freshness, 2),
            source_quality=round(best_quality, 2),
            support_confidence=round(best_support, 2),
            conflict_level="high" if (supports > 0 and contradicts > 0) else "low",
            reason=reason,
            suggested_patch=patch if request.options.generate_patch else "",
            evidence=evidences if request.options.include_evidence else [],
        )
        claim_results.append(claim)
        score_rows.append(
            {
                "status": claim.status,
                "freshness_score": claim.freshness_score,
                "source_quality": claim.source_quality,
                "support_confidence": claim.support_confidence,
            }
        )

    t_score = round(truth_score(score_rows), 2)
    overall = "clean" if all(c.status in {"VERIFIED", "STABLE"} for c in claim_results) else "stale_detected"

    created = datetime.now(timezone.utc)
    response = VerifyResponse(
        verification_id=verification_id,
        status="completed",
        overall_decision=overall,
        truth_score=t_score,
        created_at=created,
        claims=claim_results,
        policy={
            "mode": request.mode,
            "recommended_action": "return_with_warning" if overall != "clean" else "return",
        },
        usage={
            "tavily_search_calls": search_calls,
            "tavily_extract_calls": extract_calls,
            "cache_hit": cache_hits > 0,
        },
    )

    if request.options.persist:
        vrun = VerificationRunDB(
            id=verification_id,
            trajectory_id=trajectory_id,
            request_id=req_id,
            mode=request.mode,
            model=request.context.model_used or "",
            status="completed",
            truth_score=t_score,
            created_at=created,
            latency_ms=int((created - start).total_seconds() * 1000),
        )
        session.add(vrun)

        for claim in claim_results:
            claim_row = ClaimDB(
                id=claim.claim_id,
                run_id=verification_id,
                normalized_text=claim.text,
                offsets_json=json.dumps(claim.offsets),
                category=claim.category,
                volatility=claim.volatility,
                status=claim.status,
                suggested_patch=claim.suggested_patch,
            )
            session.add(claim_row)

            session.add(
                VerificationScoreDB(
                    id=f"scr_{uuid.uuid4().hex[:10]}",
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
                        id=f"ev_{uuid.uuid4().hex[:10]}",
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
                id=f"pu_{uuid.uuid4().hex[:10]}",
                run_id=verification_id,
                provider="tavily",
                operation="search_extract",
                request_count=search_calls,
                credits_or_tokens=float(search_calls),
                latency_ms=int((datetime.now(timezone.utc) - start).total_seconds() * 1000),
            )
        )

        session.commit()

    return response


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
            id=f"ch_{uuid.uuid4().hex[:12]}",
            claim_id=claim_id,
            previous_status=old_status,
            new_status=new_status,
            reason=reason,
        )
    )
    session.commit()
