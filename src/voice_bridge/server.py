"""
Server mode:
- Exposes OpenAI-compatible endpoints used by OpenWebUI.

Endpoints:
- POST /v1/audio/transcriptions  (STT)
- POST /v1/audio/speech          (TTS, returns audio/wav)
- POST /v1/chat/completions      (optional chat proxy; full pass-through)

Important:
- This server DOES NOT store chat history (OpenWebUI does).
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, JSONResponse
from collections import OrderedDict
import time
import os
import traceback

import httpx

from voice_bridge.config import Settings
from voice_bridge.asr_whisper import WhisperASR
from voice_bridge.audio_utils import bytes_to_16k_mono_f32, f32_to_wav_bytes
from voice_bridge.gar_client import chat_async
from voice_bridge.gar_profile_client import fetch_runtime_profile_async
from voice_bridge.tts_base import load_tts_config_from_env, build_tts
from voice_bridge.render_plan_client import fetch_render_plan_async
from voice_bridge.scene_audio import render_scene_audio_async

settings = Settings()
app = FastAPI(title="Voice Bridge", version="3.0")


@app.on_event("startup")
async def _startup():
    # TTS
    try:
        app.state.tts = build_tts(load_tts_config_from_env())
    except Exception as e:
        app.state.tts = None
        print("[voice-bridge] TTS init failed:", repr(e))

    # ASR
    try:
        app.state.asr = WhisperASR()
    except Exception as e:
        app.state.asr = None
        print("[voice-bridge] ASR init failed:", repr(e))


# --- recent completion id cache (client_ip -> completion_id) ---
# Used when OpenWebUI calls /v1/chat/completions and then /v1/audio/speech separately.
# If the TTS request does not include completion_id explicitly, we fall back to the latest
# completion_id seen from the same client IP.
_LAST_COMPLETION_BY_CLIENT: "OrderedDict[str, tuple[float, str]]" = OrderedDict()
_LAST_COMPLETION_TTL_SEC = int(os.getenv("VOICE_BRIDGE_COMPLETION_TTL_SEC", "300"))  # 5 min
_LAST_COMPLETION_MAX = int(os.getenv("VOICE_BRIDGE_COMPLETION_MAX", "1024"))


@app.get("/v1/models")
async def list_models():
    """
    OpenWebUI がモデル一覧を取得するために呼ぶ。
    1) まず upstream (gar-llm relay) の /v1/models をプロキシする
    2) 失敗したら最低限 gar-llm を返す
    """
    # LLM_CHAT_URL = http://.../v1/chat/completions から base を作る    
    base = settings.llm_chat_url.rsplit("/v1/chat/completions", 1)[0].rstrip("/")
    url = base + "/v1/models"

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            r.raise_for_status()
            return JSONResponse(content=r.json())
    except Exception:
        now = int(time.time())
        return {
            "object": "list",
            "data": [{
                "id": "gar-llm",
                "object": "model",
                "created": now,
                "owned_by": "garllm",
                "permission": []
            }]
        }


@app.get("/v1/audio/models")
async def list_audio_models():
    engines = [s.strip() for s in os.getenv("VOICE_BRIDGE_AUDIO_ENGINES", "sbv2").split(",") if s.strip()]
    return {
        "object": "list",
        "data": [{"id": e, "object": "audio.model", "owned_by": "voice-bridge"} for e in engines]
    }


@app.get("/v1/audio/voices")
async def list_audio_voices():
    # OpenWebUIのUI用。ここで在庫を表現しようとしない（将来の複数TTSに備える） 
    return {
        "object": "list",
        "data": [
            {"id": "default", "object": "audio.voice", "owned_by": "voice-bridge"}
        ]
    }


def _remember_completion(client_ip: str, completion_id: str) -> None:
    now = time.time()
    expire_before = now - _LAST_COMPLETION_TTL_SEC

    keys = []
    for k, (ts, _) in _LAST_COMPLETION_BY_CLIENT.items():
        if ts < expire_before:
            keys.append(k)
        else:
            break
    for k in keys:
        _LAST_COMPLETION_BY_CLIENT.pop(k, None)

    _LAST_COMPLETION_BY_CLIENT[client_ip] = (now, completion_id)
    _LAST_COMPLETION_BY_CLIENT.move_to_end(client_ip, last=True)

    while len(_LAST_COMPLETION_BY_CLIENT) > _LAST_COMPLETION_MAX:
        _LAST_COMPLETION_BY_CLIENT.popitem(last=False)

def _get_recent_completion(client_ip: str) -> str | None:
    item = _LAST_COMPLETION_BY_CLIENT.get(client_ip)
    if not item:
        return None
    ts, cid = item
    if (time.time() - ts) > _LAST_COMPLETION_TTL_SEC:
        _LAST_COMPLETION_BY_CLIENT.pop(client_ip, None)
        return None
    return cid

def _emotion_to_style(emotion_axes: dict, baseline: str | None) -> tuple[str | None, float | None]:
    """Map GAR emotion axes to SBV2 style + style_weight."""
    if not isinstance(emotion_axes, dict) or not emotion_axes:
        return None, None

    STYLE_WEIGHT_MAX = 20.0

    def clamp_style_weight(x: float) -> float:
        return 0.0 if x < 0.0 else (STYLE_WEIGHT_MAX if x > STYLE_WEIGHT_MAX else x)

    cands = {
        "Happy": float(emotion_axes.get("joy", 0.0)),
        "Angry": float(emotion_axes.get("anger", 0.0)),
        "Sad": float(emotion_axes.get("sadness", 0.0)),
        "Fear": float(emotion_axes.get("fear", 0.0)),
        "Disgust": float(emotion_axes.get("disgust", 0.0)),
        "Surprise": float(emotion_axes.get("surprise", 0.0)),
    }
    for k in list(cands.keys()):
        if cands[k] < 0.0:
            cands[k] = 0.0

    style, raw = max(cands.items(), key=lambda kv: kv[1])
    if raw <= 0.05:
        return "Neutral", 0.5

    # 0..1想定の感情値を、SBV2 WebUI相当の 0..20 に拡張
    w = clamp_style_weight(raw * STYLE_WEIGHT_MAX)

    b = (baseline or "").lower()
    if b == "calm":
        w *= 0.75
    elif b == "energetic":
        w = min(STYLE_WEIGHT_MAX, w * 1.15)
    elif b == "whisper":
        w *= 0.6

    return style, float(clamp_style_weight(w))

@app.middleware("http")
async def debug_middleware(request: Request, call_next):
    if request.url.path.endswith("/audio/transcriptions"):
        ct = request.headers.get("content-type", "")
        print("=== STT REQUEST ===")
        print("path:", request.url.path)
        print("content-type:", ct)
        print("content-length:", request.headers.get("content-length"))
        if "application/json" in ct:
            try:
                body = await request.body()
                print("json body (head):", body[:500])
            except Exception as e:
                print("json read error:", e)

    if request.url.path.endswith("/audio/speech"):
        ct = request.headers.get("content-type", "")
        print("=== TTS REQUEST ===")
        print("path:", request.url.path)
        print("content-type:", ct)
        print("content-length:", request.headers.get("content-length"))
        if "application/json" in ct:
            try:
                body = await request.body()
                print("tts json body:", body[:2000])
            except Exception as e:
                print("tts json read error:", e)

    try:
        resp = await call_next(request)
        return resp
    except Exception:
        print("=== EXCEPTION ===")
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"detail": "internal error"})

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/audio/transcriptions")
async def audio_transcriptions_raw(request: Request):
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" not in ct:
        raise HTTPException(status_code=400, detail=f"unsupported content-type: {ct}")

    try:
        form = await request.form()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"form parse error: {type(e).__name__}: {e}")

    print("STT form keys:", list(form.keys()))

    upload = None
    for k, v in form.items():
        if hasattr(v, "filename") and hasattr(v, "read"):
            upload = v
            print("STT picked file field:", k, "filename:", getattr(v, "filename", None))
            break

    if upload is None:
        raise HTTPException(status_code=400, detail=f"no file part in form. keys={list(form.keys())}")

    raw = await upload.read()

    try:
        audio_f32 = bytes_to_16k_mono_f32(raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"audio decode error: {e}")

    lang = "ja"
    if "language" in form and isinstance(form["language"], str) and form["language"]:
        lang = form["language"]

    asr = request.app.state.asr
    if asr is None:
        raise HTTPException(status_code=503, detail="ASR is not configured")

    text = asr.transcribe_16k_mono(audio_f32, language=lang)

    t = (text or "").strip()
    if t in {"ん", "ン", "え", "あ", "う", "お", "ま", "な"}:
        return {"text": ""}
    if len(t) <= 1:
        return {"text": ""}

    return {"text": t}

@app.post("/audio/transcriptions")
async def audio_transcriptions_alias(request: Request):
    return await audio_transcriptions_raw(request)

@app.post("/v1/audio/speech")
async def audio_speech(req: Request):
    """
    OpenAI-like TTS endpoint.
    Request body typically:
      { "model": "...", "input": "text", "voice": "...", "format": "wav" }

    Extra (optional):
      - completion_id: chat completion id (chatcmpl-...)
      - voice_id: engine-namespaced voice id (e.g. sbv2:jvnv-F1-jp:0)

    Behavior:
      - if GAR render_plan exists, prefer spoken_text for TTS
      - if render_plan has typed segments, render scene audio
      - otherwise fall back to ordinary single-utterance TTS
      - when chat proxy is enabled, visible_text may be used for WebUI display
    """
    body = await req.json()
    tts = req.app.state.tts
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS is not configured")

    text = body.get("input") or body.get("text")
    if not text:
        raise HTTPException(status_code=400, detail="missing 'input'")

    spoken_text = text
    render_plan = None

    # Optional hints
    style = body.get("style")
    style_weight = body.get("style_weight")

    engines = [s.strip() for s in os.getenv("VOICE_BRIDGE_AUDIO_ENGINES", "sbv2").split(",") if s.strip()]
    engines_set = set(engines)

    def _is_namespaced(v: str) -> bool:
        parts = v.split(":")
        return len(parts) >= 2 and parts[0].strip() in engines_set and all(p.strip() for p in parts[:2])

    DEFAULT_VOICE_ID = os.getenv("DEFAULT_VOICE_ID", "sbv2:jvnv-F1-jp:0")

    # Optional hints
    tts_model = body.get("model")
    voice = body.get("voice")
    voice_id_in = body.get("voice_id")
    voice_id = None

    # 1) explicit voice_id wins
    if isinstance(voice_id_in, str) and voice_id_in.strip():
        v = voice_id_in.strip()
        if not _is_namespaced(v):
            raise HTTPException(status_code=400, detail=f"voice_id must be namespaced like 'sbv2:...': {v}")
        voice_id = v

    # 2) namespaced voice field
    elif isinstance(voice, str) and voice.strip() and _is_namespaced(voice.strip()):
        voice_id = voice.strip()

    # 3) model(=engine) + voice(=local_id)
    elif isinstance(tts_model, str) and tts_model.strip() and tts_model.strip() in engines_set:
        if isinstance(voice, str) and voice.strip():
            voice_id = f"{tts_model.strip()}:{voice.strip()}"
        else:
            voice_id = DEFAULT_VOICE_ID

    completion_id = body.get("completion_id")

    if not completion_id and req.client and req.client.host:
        completion_id = _get_recent_completion(req.client.host)

    if completion_id:
        try:
            render_plan = await fetch_render_plan_async(settings.gar_base_url, completion_id)
            if isinstance(render_plan, dict):
                candidate = render_plan.get("spoken_text") or render_plan.get("speech_text")
                if isinstance(candidate, str) and candidate.strip():
                    spoken_text = candidate.strip()
                    print("[voice-bridge] spoken_text from render_plan =", repr(spoken_text[:200]))
        except Exception as e:
            print("[voice-bridge] render_plan fetch failed:", type(e).__name__, str(e)[:200])

    if completion_id:
        try:
            prof = await fetch_runtime_profile_async(settings.gar_base_url, completion_id)

            baseline = None
            v = prof.get("voice") if isinstance(prof, dict) else None
            if isinstance(v, dict):
                pref = v.get("pref")
                if isinstance(pref, dict) and isinstance(pref.get("id"), str) and pref.get("id"):
                    voice_id = pref["id"]
                delivery = v.get("delivery") if isinstance(v.get("delivery"), dict) else {}
                baseline = delivery.get("baseline") if isinstance(delivery, dict) else None

            emo = prof.get("emotion", {}) if isinstance(prof, dict) else {}
            axes = emo.get("axes") if isinstance(emo, dict) else None
            s2, w2 = _emotion_to_style(axes, baseline)
            print("[voice-bridge] axes =", axes, "baseline =", baseline)
            print("[voice-bridge] mapped style =", s2, "mapped weight =", w2)
            print("[voice-bridge] completion_id =", completion_id)

            if style is None and s2 is not None:
                style = s2
            if style_weight is None and w2 is not None:
                style_weight = w2

        except Exception as e:
            print("[voice-bridge] runtime_profile fetch failed:", type(e).__name__, str(e)[:200])

    if not voice_id:
        voice_id = DEFAULT_VOICE_ID

    kwargs = {"style": style, "style_weight": style_weight}
    if voice_id is not None:
        kwargs["voice_id"] = voice_id

    print("[voice-bridge] FINAL style =", kwargs.get("style"), "weight =", kwargs.get("style_weight"))
    print("[voice-bridge] FINAL voice_id =", kwargs.get("voice_id"))

    try:
        audio_f32, sr = await render_scene_audio_async(
            tts=tts,
            render_plan=render_plan,
            fallback_text=spoken_text,
            tts_kwargs=kwargs,
        )
    except Exception as e:
        print("[voice-bridge] scene render failed, fallback to DEFAULT_VOICE_ID. err=", repr(e))
        kwargs["voice_id"] = os.getenv("DEFAULT_VOICE_ID", "sbv2:jvnv-F1-jp:0")
        audio_f32, sr = await render_scene_audio_async(
            tts=tts,
            render_plan=render_plan,
            fallback_text=spoken_text,
            tts_kwargs=kwargs,
        )

    wav = f32_to_wav_bytes(audio_f32, sr)

    headers = {}
    if kwargs.get("voice_id") is not None:
        headers["x-voice-id"] = str(kwargs["voice_id"])
    if kwargs.get("style") is not None:
        headers["x-style"] = str(kwargs["style"])
    if kwargs.get("style_weight") is not None:
        headers["x-style-weight"] = str(kwargs["style_weight"])

    return Response(content=wav, media_type="audio/wav", headers=headers)


async def _apply_visible_text_override(data: dict) -> dict:
    """
    When chat proxy is used, prefer render_plan.visible_text for UI display.
    This keeps non-speech tags out of WebUI while preserving direct gar-llm compatibility.
    """
    if os.getenv("VOICE_BRIDGE_PREFER_VISIBLE_TEXT", "1") != "1":
        return data

    if not isinstance(data, dict):
        return data

    completion_id = data.get("id")
    if not isinstance(completion_id, str) or not completion_id:
        return data

    try:
        plan = await fetch_render_plan_async(settings.gar_base_url, completion_id)
    except Exception:
        return data

    if not isinstance(plan, dict):
        return data

    visible_text = str(plan.get("visible_text") or "").strip()
    if not visible_text:
        return data

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return data

    first = choices[0]
    if not isinstance(first, dict):
        return data

    message = first.get("message")
    if isinstance(message, dict):
        message["content"] = visible_text

    return data


@app.post("/v1/chat/completions")
async def chat_completions(req: Request):
    """
    Optional chat proxy:
    - Full JSON pass-through to LLM server.
    - Enabled only when ENABLE_CHAT_PROXY=1
    """
    if not settings.enable_chat_proxy:
        raise HTTPException(status_code=404, detail="chat proxy disabled")

    body = await req.json()
    body["stream"] = False
    print("[voice-bridge] llm_chat_url =", settings.llm_chat_url, "gar_base_url =", settings.gar_base_url, "chat_proxy =", settings.enable_chat_proxy)

    extra_headers = {settings.meta_request_header: "1"}
    data = await chat_async(settings.llm_chat_url, body, extra_headers=extra_headers)

    # remember completion_id for this client (for later TTS calls)
    try:
        cid = data.get("id")
        if cid and req.client and req.client.host:
            _remember_completion(req.client.host, cid)
    except Exception:
        pass

    data = await _apply_visible_text_override(data)

    allowed = {"id", "object", "created", "model", "choices", "usage", "system_fingerprint"}
    clean = {k: v for k, v in data.items() if k in allowed}
    return JSONResponse(content=clean if clean.get("choices") else data)
