# 1) import 群にこれを追加
from voice_bridge.scene_audio import render_scene_audio_async


# 2) audio_speech 関数をこの内容で丸ごと置き換え
@app.post("/v1/audio/speech")
async def audio_speech(req: Request):
    """
    OpenAI-like TTS endpoint.
    Request body typically:
      { "model": "...", "input": "text", "voice": "...", "format": "wav" }

    Extra (optional):
      - completion_id: chat completion id (chatcmpl-...)
      - voice_id: engine-namespaced voice id (e.g. sbv2:jvnv-F1-jp:0)

    New behavior:
      - if GAR render_plan exists and has segments, render scene audio
      - otherwise fall back to ordinary single-utterance TTS
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

    # OpenWebUI fields
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
                candidate = render_plan.get("speech_text")
                if isinstance(candidate, str) and candidate.strip():
                    spoken_text = candidate.strip()
                    print("[voice-bridge] speech_text from render_plan =", repr(spoken_text[:200]))
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
