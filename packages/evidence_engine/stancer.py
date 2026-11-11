"""Heuristic stance detection between a claim and an evidence excerpt.

This is deliberately simple and explainable. It is the fallback until an LLM/NLI judge is wired in
(see NEXT_STEPS.md). Rules:

* contradicts: the excerpt says a term the claim relies on is deprecated/replaced/removed,
  and the claim does not also mention the replacement.
* supports:    the claim uses the current replacement term the excerpt recommends, or
               most of the claim's meaningful terms appear in the excerpt.
* irrelevant:  no meaningful overlap at all.
* contextual:  anything in between.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

# Host suffixes, matched against the end of the hostname (not as substrings anywhere).
PRIMARY_HOST_SUFFIXES = ["pypi.org", "npmjs.com", "python.org", "openai.com", "ai.google.dev", "anthropic.com", ".gov"]
SECONDARY_HOST_SUFFIXES = ["github.com", "medium.com", "techcrunch.com", "stackoverflow.com", "dev.to"]

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with", "by", "is", "are", "was",
    "were", "be", "been", "it", "its", "this", "that", "these", "those", "as", "at", "from", "use", "uses",
    "using", "used", "can", "should", "will", "has", "have", "had", "not", "you", "your", "we", "our",
    "about", "into", "than", "then", "also", "some", "any", "all", "more", "most", "such", "based", "per",
}

_TOKEN = re.compile(r"[@.]?[A-Za-z_][\w\-]*(?:\(\))?|\$?\d+(?:\.\d+)*")

# Patterns that name an outdated term and (optionally) its replacement. Groups: old, new.
_OUTDATED_PATTERNS: list[tuple[re.Pattern[str], int, int | None]] = [
    (re.compile(r"replac\w*\s+(\S+)\s+with\s+(\S+)", re.I), 1, 2),
    (re.compile(r"(\S+)\s+(?:is|was|has been|have been|are|were)\s+replaced\s+by\s+(\S+)", re.I), 1, 2),
    (re.compile(r"use\s+(\S+)\s+instead\s+of\s+(\S+)", re.I), 2, 1),
    (re.compile(r"(\S+)\s+(?:is|was|has been|are|were)\s+(?:now\s+)?(?:deprecated|removed|discontinued)", re.I), 1, None),
    (re.compile(r"(\S+)\s+(?:is|are)\s+no longer\s+(?:supported|available|recommended)", re.I), 1, None),
]
# Follow-on pairs in the same sentence: "replaces dict() with model_dump() and validator with field_validator".
_FOLLOW_ON_PAIR = re.compile(r"(?:,|\band\b)\s+(\S+)\s+with\s+(\S+)", re.I)


@dataclass
class OutdatedTerm:
    old: str
    new: str | None


def normalize_term(token: str) -> str:
    t = token.lower().strip(" \t\r\n.,;:!?'\"`()[]{}")
    t = t.lstrip("@.")
    if t.endswith("()"):
        t = t[:-2]
    return t.strip("()")


def significant_terms(text: str) -> set[str]:
    terms = set()
    for raw in _TOKEN.findall(text):
        norm = normalize_term(raw)
        if len(norm) >= 3 and norm not in STOPWORDS:
            terms.add(norm)
    return terms


def find_outdated_terms(excerpt: str) -> list[OutdatedTerm]:
    found: list[OutdatedTerm] = []
    for sentence in re.split(r"(?<=[.!?])\s+", excerpt):
        for pattern, old_group, new_group in _OUTDATED_PATTERNS:
            for m in pattern.finditer(sentence):
                old = normalize_term(m.group(old_group))
                new = normalize_term(m.group(new_group)) if new_group else None
                if old and old not in STOPWORDS:
                    found.append(OutdatedTerm(old=old, new=new))
        if re.search(r"replac\w*", sentence, re.I):
            for m in _FOLLOW_ON_PAIR.finditer(sentence):
                old, new = normalize_term(m.group(1)), normalize_term(m.group(2))
                if old and old not in STOPWORDS:
                    found.append(OutdatedTerm(old=old, new=new))
    return found


def outdated_terms_in_claim(claim: str, excerpt: str) -> list[OutdatedTerm]:
    """Outdated terms from the excerpt that the claim uses without mentioning their replacement."""
    claim_terms = significant_terms(claim)
    hits: list[OutdatedTerm] = []
    for term in find_outdated_terms(excerpt):
        if term.old in claim_terms and (term.new is None or term.new not in claim_terms):
            if all(h.old != term.old for h in hits):
                hits.append(term)
    return hits


def recommended_terms_in_claim(claim: str, excerpt: str) -> list[str]:
    """Replacement terms ("Y" in "X was replaced by Y") that the claim uses: evidence the claim is current."""
    claim_terms = significant_terms(claim)
    return sorted({t.new for t in find_outdated_terms(excerpt) if t.new and t.new in claim_terms})


def source_quality_score(url: str) -> float:
    host = urlparse(url).netloc.lower().split(":")[0]
    if not host:
        return 35.0
    if host.startswith("docs.") or _host_matches(host, PRIMARY_HOST_SUFFIXES):
        return 100.0
    if host.startswith("blog.") or _host_matches(host, SECONDARY_HOST_SUFFIXES):
        return 70.0
    return 35.0


def _host_matches(host: str, suffixes: list[str]) -> bool:
    for suffix in suffixes:
        if suffix.startswith("."):
            if host.endswith(suffix):
                return True
        elif host == suffix or host.endswith("." + suffix):
            return True
    return False


def term_overlap(claim: str, excerpt: str) -> float:
    claim_terms = significant_terms(claim)
    if not claim_terms:
        return 0.0
    return len(claim_terms & significant_terms(excerpt)) / len(claim_terms)


def infer_stance(claim: str, excerpt: str) -> str:
    if outdated_terms_in_claim(claim, excerpt):
        return "contradicts"
    if recommended_terms_in_claim(claim, excerpt):
        return "supports"

    overlap = term_overlap(claim, excerpt)
    if overlap == 0.0:
        return "irrelevant"
    if overlap >= 0.5 and len(significant_terms(claim) & significant_terms(excerpt)) >= 2:
        return "supports"
    return "contextual"


def support_confidence(claim: str, excerpt: str, stance: str) -> float:
    """Confidence (0-100) that the stance call is right, driven by how much of the claim the excerpt covers."""
    overlap = term_overlap(claim, excerpt)
    if stance == "contradicts":
        return round(min(100.0, 75.0 + overlap * 25.0), 2)
    if stance == "supports":
        return round(min(100.0, 55.0 + overlap * 45.0), 2)
    if stance == "contextual":
        return round(min(50.0, 20.0 + overlap * 60.0), 2)
    return 0.0
