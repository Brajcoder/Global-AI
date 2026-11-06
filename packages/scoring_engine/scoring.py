from __future__ import annotations

import math
from datetime import datetime, timezone

STATUS_MULTIPLIER = {
    "VERIFIED": 1.0,
    "STABLE": 1.0,
    "NEEDS_REVIEW": 0.75,
    "UNVERIFIED": 0.25,
    "STALE": 0.0,
    "CONTRADICTED": 0.0,
}

# Evidence without a publish date is neither fresh nor stale: score it neutral instead of perfect.
UNKNOWN_DATE_FRESHNESS = 50.0

# Claims that were not checked live (stable facts, opinions) get neutral scores instead of 100s,
# so an answer full of vague, uncheckable statements cannot outscore one with verified claims.
UNCHECKED_SUPPORT = 70.0
UNCHECKED_SOURCE_QUALITY = 70.0
UNCHECKED_FRESHNESS = 100.0


def freshness_score(published_at: datetime | None, retrieved_at: datetime, half_life_days: int) -> float:
    if half_life_days <= 0:
        return 100.0
    if published_at is None:
        return UNKNOWN_DATE_FRESHNESS

    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    if retrieved_at.tzinfo is None:
        retrieved_at = retrieved_at.replace(tzinfo=timezone.utc)

    age_days = max(0.0, (retrieved_at - published_at).total_seconds() / 86400.0)
    return float(100.0 * (2.0 ** (-(age_days / half_life_days))))


def truth_score(claim_rows: list[dict]) -> float:
    if not claim_rows:
        return 0.0

    total = 0.0
    for row in claim_rows:
        status = row["status"]
        m = STATUS_MULTIPLIER.get(status, 0.25)
        c_support = row["support_confidence"]
        fresh = row["freshness_score"]
        q = row["source_quality"]
        total += m * (0.60 * c_support + 0.25 * fresh + 0.15 * q)

    return total / len(claim_rows)


def eff_score(tokens_used: int, latency_ms: int, loop_ratio: float, token_budget: int = 4000, latency_budget: int = 10000) -> float:
    loop_ratio = min(max(loop_ratio, 0.0), 1.0)
    return 100.0 * (
        0.50 * math.exp(-tokens_used / token_budget)
        + 0.30 * math.exp(-latency_ms / latency_budget)
        + 0.20 * (1.0 - loop_ratio)
    )


def chronos_score(truth: float, efficiency: float) -> float:
    return 0.65 * truth + 0.35 * efficiency
