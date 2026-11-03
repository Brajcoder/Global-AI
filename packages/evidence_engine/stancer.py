from __future__ import annotations

from urllib.parse import urlparse


def source_quality_score(url: str) -> float:
    host = urlparse(url).netloc.lower()
    primary = ["pypi.org", "npmjs.com", "docs.", ".gov", "openai.com", "ai.google.dev", "python.org"]
    secondary = ["github.com", "medium.com", "techcrunch.com", "blog"]

    if any(p in host for p in primary):
        return 100.0
    if any(p in host for p in secondary):
        return 70.0
    return 35.0


def infer_stance(claim: str, excerpt: str) -> str:
    c = claim.lower()
    e = excerpt.lower()

    contradiction_hints = ["deprecated", "replaced", "no longer", "removed", "not supported", "former"]
    if any(h in e for h in contradiction_hints) and any(k in c for k in ["use", "version", "price", "ceo", "current"]):
        return "contradicts"

    support_hints = ["official", "current", "latest", "introduced", "supports", "is"]
    if any(h in e for h in support_hints):
        return "supports"

    return "contextual"


def support_confidence(claim: str, excerpt: str, stance: str) -> float:
    overlap = len(set(claim.lower().split()) & set(excerpt.lower().split()))
    base = min(100.0, 30.0 + overlap * 4.0)

    if stance == "supports":
        return max(base, 65.0)
    if stance == "contradicts":
        return max(base, 85.0)
    return min(base, 60.0)
