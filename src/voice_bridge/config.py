"""
Settings loader for Voice Bridge.

- Keeps "engine-specific" names out of code.
- Everything external is described via environment variables.
"""

from dataclasses import dataclass
import os


@dataclass
class Settings:
    # LLM chat endpoint (OpenAI-compatible /v1/chat/completions)
    llm_chat_url: str = os.getenv("LLM_CHAT_URL", "http://127.0.0.1:8000/v1/chat/completions")

    # GAR relay server base URL (for observer API /v1/gar/runtime_profile)
    # If not set, derive from LLM_CHAT_URL by stripping '/v1/chat/completions'.
    gar_base_url: str = os.getenv(
        "GAR_BASE_URL",
        os.getenv("LLM_CHAT_URL", "http://127.0.0.1:8000/v1/chat/completions").rsplit("/v1/chat/completions", 1)[0]
    )

    # Bind for server mode
    host: str = os.getenv("VOICE_BRIDGE_HOST", "127.0.0.1")
    port: int = int(os.getenv("VOICE_BRIDGE_PORT", "8787"))

    # ASR config
    asr_model: str = os.getenv("ASR_MODEL", "small")     # tiny/base/small/medium/large-v3 etc.
    asr_device: str = os.getenv("ASR_DEVICE", "auto")    # cpu/cuda/auto

    # Optional: chat proxy switch
    enable_chat_proxy: bool = os.getenv("ENABLE_CHAT_PROXY", "0") == "1"

    # If you want to request extra meta from LLM server, this is the header name you can send.
    # (Your LLM server may ignore it. This bridge just sends it if proxy is enabled.)
    meta_request_header: str = os.getenv("META_REQUEST_HEADER", "X-Bridge-Meta")
