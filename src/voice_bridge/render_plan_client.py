"""GAR render_plan client for voice-bridge."""

from __future__ import annotations

import httpx


async def fetch_render_plan_async(base_url: str, completion_id: str, timeout_s: int = 10) -> dict:
    url = base_url.rstrip("/") + "/v1/gar/render_plan"
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url, params={"completion_id": completion_id})
        r.raise_for_status()
        return r.json()


def fetch_render_plan_sync(base_url: str, completion_id: str, timeout_s: int = 10) -> dict:
    url = base_url.rstrip("/") + "/v1/gar/render_plan"
    with httpx.Client(timeout=timeout_s) as client:
        r = client.get(url, params={"completion_id": completion_id})
        r.raise_for_status()
        return r.json()
