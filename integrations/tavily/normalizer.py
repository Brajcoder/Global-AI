from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def normalize_evidence_items(tavily_payload: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for item in tavily_payload.get("results", []):
        published_at = None
        if item.get("published_date"):
            try:
                published_at = datetime.fromisoformat(item["published_date"].replace("Z", "+00:00"))
            except ValueError:
                published_at = None

        out.append(
            {
                "url": item.get("url", ""),
                "title": item.get("title", "Untitled"),
                "publisher": _publisher_from_url(item.get("url", "")),
                "published_at": published_at,
                "retrieved_at": now,
                "excerpt": item.get("content", ""),
            }
        )

    return out


def _publisher_from_url(url: str) -> str:
    if "//" not in url:
        return "unknown"
    return url.split("//", 1)[1].split("/", 1)[0]
