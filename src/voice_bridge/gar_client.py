"""
LLM client (OpenAI-compatible chat endpoint).
- In server mode (FastAPI), we use async.
- In standalone mode, we use sync.
"""

import httpx


async def chat_async(url: str, body: dict, extra_headers: dict | None = None, timeout_s: int = 120) -> dict:
    headers = {}
    if extra_headers:
        headers.update(extra_headers)

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.post(url, json=body, headers=headers)
        r.raise_for_status()
        return r.json()


def chat_sync(url: str, body: dict, extra_headers: dict | None = None, timeout_s: int = 120) -> dict:
    headers = {}
    if extra_headers:
        headers.update(extra_headers)

    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, json=body, headers=headers)
        r.raise_for_status()
        return r.json()
