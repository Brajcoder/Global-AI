from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AtomicClaim:
    text: str
    start: int
    end: int


# Split after . ! ? only when the next sentence starts with an uppercase letter, digit or quote.
# This keeps "e.g. foo" and "call .dict() to serialize" inside one sentence.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(])")
# Only split on semicolons. Splitting on "and"/"but" separated "use .dict() and @validator"
# into fragments too short to classify, which hid exactly the stale claims we want to catch.
_CLAUSE_SPLIT = re.compile(r";")
MIN_CLAIM_CHARS = 12
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "approx.", "mr.", "mrs.", "ms.", "dr.", "inc.", "ltd.")


def extract_atomic_claims(text: str, max_claims: int = 10) -> list[AtomicClaim]:
    claims: list[AtomicClaim] = []
    if max_claims <= 0:
        return claims

    for s_start, s_end in _sentence_spans(text):
        for c_start, c_end in _iter_spans(text, _CLAUSE_SPLIT, s_start, s_end):
            # Trim whitespace and trailing commas while keeping offsets exact.
            while c_start < c_end and text[c_start] in " \t\r\n,":
                c_start += 1
            while c_end > c_start and text[c_end - 1] in " \t\r\n,":
                c_end -= 1
            part = text[c_start:c_end]
            if len(part) < MIN_CLAIM_CHARS:
                continue
            claims.append(AtomicClaim(text=part, start=c_start, end=c_end))
            if len(claims) >= max_claims:
                return claims

    return claims


def _sentence_spans(text: str):
    """Sentence spans, ignoring split points that directly follow a common abbreviation."""
    cursor = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        if text[max(0, match.start() - 8) : match.start()].lower().endswith(_ABBREVIATIONS):
            continue
        yield cursor, match.start()
        cursor = match.end()
    yield cursor, len(text)


def _iter_spans(text: str, splitter: re.Pattern[str], start: int, end: int):
    """Yield (start, end) spans of text[start:end] between splitter matches, with exact offsets."""
    cursor = start
    for match in splitter.finditer(text, start, end):
        yield cursor, match.start()
        cursor = match.end()
    yield cursor, end
