from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class VolatilityDecision:
    category: str
    volatility: str
    time_sensitive: bool
    verify_live: bool
    half_life_days: int


# All patterns use word boundaries: plain substrings matched "deliver" as "live",
# "rapid" as "api" and "conversion" as "version".
HIGH_VOL_PATTERNS = [r"\bstocks?\b", r"\breal[- ]?time\b", r"\buptime\b", r"\blive\b", r"\boutages?\b"]
SOFTWARE_PATTERNS = [
    r"\bversions?\b",
    r"\bv\d+(\.\d+)*\b",
    r"\bsdks?\b",
    r"\bapis?\b",
    r"\bdeprecat\w*",
    r"\breleases?\b",
    # Code-level API usage such as .dict(), model_dump() or @validator.
    r"\.\w+\(\)",
    r"\b\w+\(\)",
    r"@\w+",
]
ORG_PRICING_PATTERNS = [
    r"\bpric(e|es|ing)\b",
    r"\bcosts?\b",
    r"\$\d",
    r"\bceo\b",
    r"\bleader(s|ship)?\b",
    r"\bregulations?\b",
    r"\bcompliance\b",
]
MODERATE_PATTERNS = [r"\bcapabilit(y|ies)\b", r"\bfeatures?\b", r"\broadmap\b", r"\bmilestones?\b"]
OPINION_PATTERNS = [r"\bi think\b", r"\bin my opinion\b", r"\bi prefer\b", r"\bi believe\b"]


def classify_claim(claim: str) -> VolatilityDecision:
    c = claim.lower()

    # Checkable signals take priority over hedging words: "X is likely the best API" still needs checking.
    if _has_any(c, HIGH_VOL_PATTERNS):
        return VolatilityDecision("live_status", "highly_volatile", True, True, 1)
    if _has_any(c, SOFTWARE_PATTERNS):
        return VolatilityDecision("software_version", "time_sensitive", True, True, 90)
    if _has_any(c, ORG_PRICING_PATTERNS):
        return VolatilityDecision("pricing_or_org", "time_sensitive", True, True, 90)
    if _has_any(c, OPINION_PATTERNS):
        return VolatilityDecision("opinion", "opinion", False, False, 0)
    if _has_any(c, MODERATE_PATTERNS):
        return VolatilityDecision("product_capability", "moderately_stable", False, True, 365)

    return VolatilityDecision("general_fact", "stable", False, False, 0)


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)
