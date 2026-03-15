"""
TTS backend builder (single source of truth).

Goals:
- Keep the rest of code free from engine-specific names.
- Construct TTS backend from environment variables.
- Provide one function: build_tts() that returns an object with synthesize().
"""

from dataclasses import dataclass
import os
import numpy as np
from typing import Protocol


class TTSClient(Protocol):
    """
    TTS interface.
    Any backend must implement:
      synthesize(text, **kwargs) -> (audio_f32, sr)
    """
    def synthesize(self, text: str, **kwargs) -> tuple[np.ndarray, int]:
        raise NotImplementedError


@dataclass
class TTSConfig:
    # backend: "http" or "dummy" or "sbv2"
    backend: str

    # HTTP backend settings / SBV2 base_url
    url: str
    api_key: str
    api_key_header: str

    # optional init endpoint/payload
    init_url: str
    init_payload: str  # JSON string

    # JSON field keys expected by external TTS
    text_key: str
    audio_key: str
    sr_key: str

    # audio scale interpretation
    scale: str  # "int16" or "float"


def load_tts_config_from_env() -> TTSConfig:
    return TTSConfig(
        backend=os.getenv("TTS_BACKEND", "http").lower(),
        url=os.getenv("TTS_URL", ""),
        api_key=os.getenv("TTS_API_KEY", ""),
        api_key_header=os.getenv("TTS_API_KEY_HEADER", "api_key"),
        init_url=os.getenv("TTS_INIT_URL", ""),
        init_payload=os.getenv("TTS_INIT_PAYLOAD", ""),
        text_key=os.getenv("TTS_TEXT_KEY", "text"),
        audio_key=os.getenv("TTS_AUDIO_KEY", "audio"),
        sr_key=os.getenv("TTS_SR_KEY", "sr"),
        scale=os.getenv("TTS_SCALE", "int16").lower(),
    )


class _HttpBackendWrapper:
    def __init__(self, inner, cfg: TTSConfig):
        self.inner = inner
        self.cfg = cfg

    def synthesize(self, text: str, **kwargs):
        return self.inner.synthesize(
            text,
            text_key=self.cfg.text_key,
            audio_key=self.cfg.audio_key,
            sr_key=self.cfg.sr_key,
            scale=self.cfg.scale,
            **kwargs,
        )


def build_tts(cfg: TTSConfig) -> TTSClient:
    if cfg.backend == "dummy":
        from voice_bridge.tts_dummy import DummyTTS
        return DummyTTS()

    if cfg.backend == "sbv2":
        # Style-Bert-VITS2 backend (audio/wav via /voice)
        if not cfg.url:
            raise RuntimeError("TTS_BACKEND=sbv2 but TTS_URL (SBV2 base_url) is empty")
        from voice_bridge.tts_sbv2 import StyleBertVITS2TTS, SBV2Config
        return StyleBertVITS2TTS(SBV2Config(base_url=cfg.url))

    if cfg.backend == "http":
        if not cfg.url:
            raise RuntimeError("TTS_BACKEND=http but TTS_URL is empty")

        from voice_bridge.tts_http import HttpTTS
        tts = HttpTTS(
            tts_url=cfg.url,
            api_key=cfg.api_key,
            api_key_header=cfg.api_key_header,
            init_url=cfg.init_url,
            init_payload_json=cfg.init_payload,
        )
        tts.initialize_if_needed()
        return _HttpBackendWrapper(tts, cfg)

    raise RuntimeError(f"Unknown TTS_BACKEND={cfg.backend}")
