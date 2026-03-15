"""
Generic external HTTP TTS client.

This file intentionally contains NO engine-specific names.
It can talk to any external TTS service that:
- Accepts JSON with a text field (default key: "text")
- Returns JSON containing:
    - audio: list of samples (either int16-like or float)
    - sr: sample rate

You can configure keys/scale via env vars (see tts_base.py).
"""

import json
import numpy as np
import httpx


class HttpTTS:
    def __init__(
        self,
        tts_url: str,
        api_key: str = "",
        api_key_header: str = "api_key",
        init_url: str = "",
        init_payload_json: str = "",
    ):
        self.tts_url = tts_url
        self.api_key = api_key
        self.api_key_header = api_key_header
        self.init_url = init_url
        self.init_payload_json = init_payload_json

    def initialize_if_needed(self) -> None:
        """Optional init step (some TTS servers require model selection, etc.)."""
        if not self.init_url:
            return

        payload = {}
        if self.init_payload_json:
            payload = json.loads(self.init_payload_json)

        headers = {}
        if self.api_key:
            headers[self.api_key_header] = self.api_key

        with httpx.Client(timeout=300) as client:
            r = client.post(self.init_url, headers=headers, json=payload)
            r.raise_for_status()

    def synthesize(
        self,
        text: str,
        *,
        text_key: str = "text",
        audio_key: str = "audio",
        sr_key: str = "sr",
        scale: str = "int16",  # "int16" or "float"
        style: str | None = None,
        style_weight: float | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Call external TTS.

        scale:
          - "int16": audio list is int16-like, normalize by 32768
          - "float": audio list already float32 [-1,1]
        """
        payload: dict = {text_key: text}

        # Optional style hints. Many engines ignore unknown keys -> safe to send only when provided.
        if style is not None:
            payload["style"] = style
        if style_weight is not None:
            payload["style_weight"] = float(style_weight)

        headers = {}
        if self.api_key:
            headers[self.api_key_header] = self.api_key

        with httpx.Client(timeout=300) as client:
            r = client.post(self.tts_url, headers=headers, json=payload)
            r.raise_for_status()
            j = r.json()

        audio = np.array(j[audio_key], dtype=np.float32)
        if scale == "int16":
            audio = audio / 32768.0
        sr = int(j[sr_key])
        return audio, sr
