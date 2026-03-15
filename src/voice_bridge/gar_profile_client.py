"""
GAR observer API client.

Fetches "runtime_profile" snapshot produced by gar-llm relay_server:
  GET /v1/gar/runtime_profile?completion_id=chatcmpl-...

This keeps OpenAI chat responses untouched and allows external tools (voice-bridge, etc.)
to retrieve persona/emotion/voice settings that were used for a specific completion.
"""

from __future__ import annotations

import httpx


async def fetch_runtime_profile_async(base_url: str, completion_id: str, timeout_s: int = 30) -> dict:
    """
    base_url example: http://127.0.0.1:8081  (GAR relay server base)
    """
    url = base_url.rstrip("/") + "/v1/gar/runtime_profile"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url, params={"completion_id": completion_id})
        r.raise_for_status()
        return r.json()


def fetch_runtime_profile_sync(base_url: str, completion_id: str, timeout_s: int = 30) -> dict:
    url = base_url.rstrip("/") + "/v1/gar/runtime_profile"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(url, params={"completion_id": completion_id})
        r.raise_for_status()
        return r.json()
