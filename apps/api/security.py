from __future__ import annotations

import hmac

from fastapi import Header, HTTPException

from packages.contracts.settings import get_settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard for write endpoints. A no-op unless API_KEY is set in the environment."""
    expected = get_settings().api_key
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
