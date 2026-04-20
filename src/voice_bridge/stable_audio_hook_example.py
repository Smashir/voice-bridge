"""Optional Stable Audio Open hook example.

This file is NOT wired by default.
Use it only after deciding your actual runtime shape:
- in-process Python model
- local HTTP wrapper service
- external generation worker

Why this is separate:
- the exact invocation contract is deployment-specific
- I did not want to invent a fake Stable Audio Open API in the main path
"""

from __future__ import annotations

import io
import os
import wave
from typing import Any

import httpx
import numpy as np


def _decode_wav_bytes_mono_f32(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        sampwidth = wf.getsampwidth()
        nch = wf.getnchannels()
        raw = wf.readframes(n)

    if sampwidth != 2:
        raise RuntimeError(f"unsupported wav sampwidth={sampwidth}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1).astype(np.float32)
    return audio, int(sr)


async def generate_sfx_via_http(
    *,
    prompt: str,
    duration_s: float,
    kind: str,
) -> tuple[np.ndarray, int]:
    """
    Example contract for your own local wrapper service.

    Expected environment variable:
      VOICE_BRIDGE_SFX_GENERATOR_URL=http://127.0.0.1:9009/generate

    Request JSON:
      {
        "prompt": "...",
        "duration_s": 2.0,
        "kind": "foley"
      }

    Response:
      audio/wav bytes
    """
    url = os.getenv("VOICE_BRIDGE_SFX_GENERATOR_URL", "").strip()
    if not url:
        raise RuntimeError("VOICE_BRIDGE_SFX_GENERATOR_URL is empty")

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            url,
            json={
                "prompt": prompt,
                "duration_s": float(duration_s),
                "kind": str(kind),
            },
        )
        r.raise_for_status()
        return _decode_wav_bytes_mono_f32(r.content)
