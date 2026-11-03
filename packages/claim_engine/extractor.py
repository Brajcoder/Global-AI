from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class AtomicClaim:
    text: str
    start: int
    end: int


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def extract_atomic_claims(text: str, max_claims: int = 10) -> list[AtomicClaim]:
    claims: list[AtomicClaim] = []
    cursor = 0

    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        idx = text.find(sentence, cursor)
        if idx == -1:
            idx = cursor
        end = idx + len(sentence)
        cursor = end

        # Split by conjunctions for finer atomicity.
        parts = re.split(r"\s+and\s+|\s+but\s+|;", sentence)
        local_cursor = idx
        for part in parts:
            part = part.strip(" ,")
            if len(part) < 12:
                continue
            local_idx = text.find(part, local_cursor)
            if local_idx == -1:
                local_idx = local_cursor
            local_end = local_idx + len(part)
            local_cursor = local_end
            claims.append(AtomicClaim(text=part, start=local_idx, end=local_end))
            if len(claims) >= max_claims:
                return claims

    return claims
