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


HIGH_VOL_PATTERNS = [r"stock", r"real[- ]?time", r"uptime", r"live"]
TIME_SENSITIVE_PATTERNS = [
    r"version",
    r"sdk",
    r"api",
    r"price",
    r"pricing",
    r"cost",
    r"ceo",
    r"leader",
    r"regulation",
    r"compliance",
    r"deprecated",
]
MODERATE_PATTERNS = [r"capability", r"feature", r"roadmap", r"milestone"]
OPINION_PATTERNS = [r"i think", r"best", r"prefer", r"opinion", r"likely"]


def classify_claim(claim: str) -> VolatilityDecision:
    c = claim.lower()

    if _has_any(c, OPINION_PATTERNS):
        return VolatilityDecision("opinion", "opinion", False, False, 0)
    if _has_any(c, HIGH_VOL_PATTERNS):
        return VolatilityDecision("live_status", "highly_volatile", True, True, 1)
    if _has_any(c, TIME_SENSITIVE_PATTERNS):
        category = "software_version" if re.search(r"version|sdk|api|deprecated", c) else "pricing_or_org"
        return VolatilityDecision(category, "time_sensitive", True, True, 90)
    if _has_any(c, MODERATE_PATTERNS):
        return VolatilityDecision("product_capability", "moderately_stable", False, True, 365)

    return VolatilityDecision("general_fact", "stable", False, False, 0)


def _has_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)
