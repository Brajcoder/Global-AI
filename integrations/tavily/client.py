from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from packages.contracts.settings import get_settings


class TavilyClient:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.api_key = self.settings.tavily_api_key
        self.cache_path = Path(self.settings.tavily_cache_db_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()

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

    def _get_cache(self, query: str) -> dict[str, Any] | None:
        if not self.settings.tavily_cache_enabled:
            return None
        conn = sqlite3.connect(self.cache_path)
        try:
            cur = conn.execute("SELECT response_json FROM tavily_cache WHERE query = ?", (query,))
            row = cur.fetchone()
            if not row:
                return None
            return json.loads(row[0])
        finally:
            conn.close()

    def _set_cache(self, query: str, payload: dict[str, Any]) -> None:
        if not self.settings.tavily_cache_enabled:
            return
        conn = sqlite3.connect(self.cache_path)
        try:
            conn.execute(
                "REPLACE INTO tavily_cache(query, response_json, cached_at) VALUES (?, ?, ?)",
                (query, json.dumps(payload), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    async def search(self, query: str, max_results: int = 3) -> dict[str, Any]:
        cached = self._get_cache(query)
        if cached:
            cached["_cache_hit"] = True
            return cached

        # Offline-safe fallback when key is absent.
        if not self.api_key:
            fallback = {
                "query": query,
                "results": [
                    {
                        "url": "https://docs.pydantic.dev/latest/migration/",
                        "title": "Pydantic Migration Guide",
                        "content": "Pydantic V2 replaces dict() with model_dump() and validator with field_validator.",
                        "published_date": "2026-09-15",
                    }
                ],
                "_cache_hit": False,
                "_offline": True,
            }
            self._set_cache(query, fallback)
            return fallback

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "query": query,
            "search_depth": "advanced",
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
        }

        async with httpx.AsyncClient(timeout=self.settings.request_timeout_seconds) as client:
            res = await client.post("https://api.tavily.com/search", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            data["_cache_hit"] = False
            self._set_cache(query, data)
            return data
