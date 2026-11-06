from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from packages.contracts.settings import get_settings, resolve_project_path

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
CREDITS_PER_SEARCH = {"basic": 1, "advanced": 2}

# Offline demo fixtures, used only when TAVILY_API_KEY is empty. Matched by keyword so unrelated
# claims get no evidence instead of the same Pydantic page for every query.
OFFLINE_FIXTURES: list[tuple[tuple[str, ...], dict[str, Any]]] = [
    (
        ("pydantic", "dict()", "@validator", "model_dump", "field_validator"),
        {
            "url": "https://docs.pydantic.dev/latest/migration/",
            "title": "Pydantic Migration Guide",
            "content": "Pydantic V2 replaces dict() with model_dump() and validator with field_validator.",
            "published_date": "2026-09-15",
        },
    ),
]


class TavilyClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.api_key = self.settings.tavily_api_key
        self.search_depth = self.settings.tavily_search_depth
        self.cache_path: Path = resolve_project_path(self.settings.tavily_cache_db_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()

    @property
    def offline(self) -> bool:
        return not self.api_key

    def _cache_key(self, query: str, max_results: int) -> str:
        return f"{self.search_depth}|{max_results}|{query.strip().lower()}"

    def _init_cache(self) -> None:
        conn = sqlite3.connect(self.cache_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tavily_cache (
                    query TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    cached_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _get_cache(self, key: str) -> dict[str, Any] | None:
        if not self.settings.tavily_cache_enabled:
            return None
        conn = sqlite3.connect(self.cache_path)
        try:
            row = conn.execute("SELECT response_json, cached_at FROM tavily_cache WHERE query = ?", (key,)).fetchone()
        finally:
            conn.close()
        if not row:
            return None
        cached_at = datetime.fromisoformat(row[1])
        if datetime.now(timezone.utc) - cached_at > timedelta(hours=self.settings.tavily_cache_ttl_hours):
            return None  # expired: verification must stay live
        return json.loads(row[0])

    def _set_cache(self, key: str, payload: dict[str, Any]) -> None:
        if not self.settings.tavily_cache_enabled:
            return
        clean = {k: v for k, v in payload.items() if not k.startswith("_")}
        conn = sqlite3.connect(self.cache_path)
        try:
            conn.execute(
                "REPLACE INTO tavily_cache(query, response_json, cached_at) VALUES (?, ?, ?)",
                (key, json.dumps(clean), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    def _offline_results(self, query: str) -> dict[str, Any]:
        q = query.lower()
        results = [item for keywords, item in OFFLINE_FIXTURES if any(k in q for k in keywords)]
        # Never cached: offline fixtures must not shadow real results once a key is configured.
        return {"query": query, "results": results, "_cache_hit": False, "_offline": True, "_credits": 0}

    async def search(self, query: str, max_results: int | None = None) -> dict[str, Any]:
        max_results = max_results or self.settings.tavily_max_results

        if self.offline:
            return self._offline_results(query)

        key = self._cache_key(query, max_results)
        cached = self._get_cache(key)
        if cached is not None:
            return {**cached, "_cache_hit": True, "_offline": False, "_credits": 0}

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "query": query,
            "search_depth": self.search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            res = await client.post(TAVILY_SEARCH_URL, headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()

        self._set_cache(key, data)
        return {**data, "_cache_hit": False, "_offline": False, "_credits": CREDITS_PER_SEARCH.get(self.search_depth, 1)}
