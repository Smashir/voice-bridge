"""Scene-audio renderer for voice-bridge.

Purpose:
- keep ordinary TTS path working as-is
- when render_plan.segments exists, render:
    * speech   -> existing TTS backend
    * foley    -> catalog asset or procedural fallback
    * ambience -> catalog asset or procedural fallback
- return one mono float32 waveform

Notes:
- This is intentionally lightweight and dependency-free (numpy only).
- High-quality ambience/foley should be supplied via VOICE_BRIDGE_SFX_CATALOG.
- Without a catalog, simple procedural fallback is used.
"""

from __future__ import annotations

import json
import math
import os
import re
import wave
from functools import lru_cache
from typing import Any

import numpy as np


DEFAULT_SCENE_SR = 24000


def _db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _normalize_peak(x: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    m = float(np.max(np.abs(x)))
    if m < 1e-8:
        return x.astype(np.float32)
    return (x / m * peak).astype(np.float32)


def _silence(sr: int, ms: int) -> np.ndarray:
    n = max(0, int(sr * ms / 1000.0))
    return np.zeros(n, dtype=np.float32)


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32)
    ratio = float(sr_out) / float(sr_in)
    n_out = int(round(x.size * ratio))
    idx = np.linspace(0, x.size - 1, n_out)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, x.size - 1)
    w = idx - i0
    y = (1.0 - w) * x[i0] + w * x[i1]
    return y.astype(np.float32)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or x.size == 0:
        return x.astype(np.float32)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _lowpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    if hz <= 0.0:
        return np.zeros_like(x, dtype=np.float32)
    win = max(1, int(sr / max(hz, 1.0)))
    return _moving_average(x, win)


def _highpass(x: np.ndarray, sr: int, hz: float) -> np.ndarray:
    lp = _lowpass(x, sr, hz)
    return (x - lp).astype(np.float32)


def _mix(base: np.ndarray, overlay: np.ndarray, offset: int = 0, gain_db: float = 0.0) -> np.ndarray:
    if overlay.size == 0:
        return base.astype(np.float32)

    offset = max(0, int(offset))
    gain = _db_to_gain(gain_db)
    src = (overlay.astype(np.float32) * gain).astype(np.float32)

    n_out = max(base.size, offset + src.size)
    out = np.zeros(n_out, dtype=np.float32)
    if base.size > 0:
        out[: base.size] += base.astype(np.float32)
    out[offset : offset + src.size] += src
    return out


def _concat_chunks(chunks: list[np.ndarray], sr: int, gap_ms: int = 80) -> np.ndarray:
    valid = [c.astype(np.float32) for c in chunks if isinstance(c, np.ndarray) and c.size > 0]
    if not valid:
        return np.zeros(0, dtype=np.float32)
    gap = _silence(sr, gap_ms)
    out: list[np.ndarray] = []
    for i, c in enumerate(valid):
        out.append(c)
        if i < len(valid) - 1 and gap.size > 0:
            out.append(gap)
    return np.concatenate(out).astype(np.float32)


def _load_wav_mono_f32(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        sampwidth = wf.getsampwidth()
        nch = wf.getnchannels()
        raw = wf.readframes(n)

    if sampwidth != 2:
        raise RuntimeError(f"unsupported wav sampwidth={sampwidth}: {path}")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1).astype(np.float32)
    return audio.astype(np.float32), int(sr)


def _normalize_cue(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"[【】\[\]()<>{}「」『』:：\s]+", "", value)
    return value.lower()


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, dict[str, Any]]:
    path = os.getenv("VOICE_BRIDGE_SFX_CATALOG", "").strip()
    if not path:
        return {"foley": {}, "ambience": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"foley": {}, "ambience": {}}
        out = {"foley": {}, "ambience": {}}
        for kind in ("foley", "ambience"):
            src = data.get(kind, {})
            if isinstance(src, dict):
                out[kind] = src
        return out
    except Exception as e:
        print(f"[voice-bridge] WARN: failed to load VOICE_BRIDGE_SFX_CATALOG '{path}': {e}")
        return {"foley": {}, "ambience": {}}


def _catalog_entry(kind: str, cue: str) -> tuple[str | None, float | None]:
    table = _load_catalog().get(kind, {})
    if not isinstance(table, dict) or not table:
        return None, None

    raw_key = (cue or "").strip()
    norm_key = _normalize_cue(raw_key)

    candidates = [raw_key, norm_key]
    for key in candidates:
        val = table.get(key)
        if isinstance(val, str):
            return val, None
        if isinstance(val, dict):
            path = val.get("path")
            gain_db = val.get("gain_db")
            return (str(path) if path else None), (float(gain_db) if gain_db is not None else None)

    for k, v in table.items():
        nk = _normalize_cue(str(k))
        if nk == norm_key or nk in norm_key or norm_key in nk:
            if isinstance(v, str):
                return v, None
            if isinstance(v, dict):
                path = v.get("path")
                gain_db = v.get("gain_db")
                return (str(path) if path else None), (float(gain_db) if gain_db is not None else None)

    return None, None


def _procedural_knock(sr: int) -> np.ndarray:
    dur = 0.12
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    body = np.sin(2.0 * np.pi * 180.0 * t) * np.exp(-32.0 * t)
    click = np.random.randn(t.size).astype(np.float32) * np.exp(-120.0 * t) * 0.18
    return _normalize_peak((body + click).astype(np.float32), peak=0.85)


def _procedural_thud(sr: int) -> np.ndarray:
    dur = 0.18
    t = np.linspace(0.0, dur, int(sr * dur), endpoint=False)
    body = np.sin(2.0 * np.pi * 90.0 * t) * np.exp(-18.0 * t)
    body += 0.4 * np.sin(2.0 * np.pi * 55.0 * t) * np.exp(-14.0 * t)
    noise = np.random.randn(t.size).astype(np.float32) * np.exp(-38.0 * t) * 0.08
    return _normalize_peak((body + noise).astype(np.float32), peak=0.9)


def _procedural_creak(sr: int) -> np.ndarray:
    dur = 0.34
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    freq = np.linspace(420.0, 160.0, n)
    phase = 2.0 * np.pi * np.cumsum(freq) / float(sr)
    tone = np.sin(phase).astype(np.float32)
    noise = _highpass(np.random.randn(n).astype(np.float32), sr, 1200.0) * 0.18
    env = (1.0 - np.exp(-22.0 * t)) * np.exp(-4.5 * t)
    audio = (tone * env * 0.75 + noise * env).astype(np.float32)
    return _normalize_peak(audio, peak=0.82)


def _procedural_swish(sr: int) -> np.ndarray:
    dur = 0.16
    n = int(sr * dur)
    t = np.linspace(0.0, dur, n, endpoint=False)
    noise = _highpass(np.random.randn(n).astype(np.float32), sr, 900.0)
    env = np.sin(np.pi * np.clip(t / dur, 0.0, 1.0)) ** 2
    audio = (noise * env * 0.45).astype(np.float32)
    return _normalize_peak(audio, peak=0.75)


def _procedural_rain(sr: int, duration_s: float) -> np.ndarray:
    n = max(int(sr * duration_s), int(sr * 0.35))
    t = np.linspace(0.0, n / sr, n, endpoint=False)
    noise = np.random.randn(n).astype(np.float32)
    hiss = _highpass(noise, sr, 1500.0) * 0.35
    lfo = 0.78 + 0.22 * np.sin(2.0 * np.pi * 0.18 * t + 0.7)
    audio = (hiss * lfo).astype(np.float32)
    return _normalize_peak(audio, peak=0.28)


def _procedural_wind(sr: int, duration_s: float) -> np.ndarray:
    n = max(int(sr * duration_s), int(sr * 0.5))
    t = np.linspace(0.0, n / sr, n, endpoint=False)
    brown = np.cumsum(np.random.randn(n).astype(np.float32))
    brown = _normalize_peak(brown, peak=1.0)
    whoosh = _lowpass(brown, sr, 180.0)
    lfo = 0.45 + 0.55 * np.sin(2.0 * np.pi * 0.07 * t + 1.2)
    audio = (whoosh * lfo * 0.38).astype(np.float32)
    return _normalize_peak(audio, peak=0.24)


def _procedural_fire(sr: int, duration_s: float) -> np.ndarray:
    n = max(int(sr * duration_s), int(sr * 0.5))
    base = _lowpass(np.random.randn(n).astype(np.float32), sr, 900.0) * 0.12
    clicks = (np.random.rand(n) > 0.996).astype(np.float32) * (np.random.rand(n).astype(np.float32) * 2.0 - 1.0)
    crackle = _moving_average(clicks, max(1, int(sr * 0.002))) * 1.8
    audio = (base + crackle).astype(np.float32)
    return _normalize_peak(audio, peak=0.22)


def _procedural_crowd(sr: int, duration_s: float) -> np.ndarray:
    n = max(int(sr * duration_s), int(sr * 0.5))
    t = np.linspace(0.0, n / sr, n, endpoint=False)
    hum = 0.03 * np.sin(2.0 * np.pi * 160.0 * t + 0.4)
    murmur = _lowpass(np.random.randn(n).astype(np.float32), sr, 280.0) * 0.10
    audio = (hum + murmur).astype(np.float32)
    return _normalize_peak(audio, peak=0.18)


def _foley_from_cue(cue: str, sr: int) -> np.ndarray:
    text = cue or ""
    if any(k in text for k in ["ぎし", "きし", "ミシ", "軋"]):
        return _procedural_creak(sr)
    if any(k in text for k in ["こと", "こつ", "コト", "コツ", "とん", "トン"]):
        return _procedural_knock(sr)
    if any(k in text for k in ["どさ", "どん", "ドサ", "ドン", "ばた", "バタ"]):
        return _procedural_thud(sr)
    if any(k in text for k in ["さら", "しゃ", "しゅ", "衣擦", "布"]):
        return _procedural_swish(sr)
    return _procedural_knock(sr)


def _ambience_from_cue(cue: str, sr: int, duration_s: float) -> np.ndarray:
    text = (cue or "").lower()
    if "rain" in text or any(k in text for k in ["雨", "ざぁ", "ザー"]):
        return _procedural_rain(sr, duration_s)
    if "wind" in text or any(k in text for k in ["風", "びゅ", "ヒュ"]):
        return _procedural_wind(sr, duration_s)
    if "fire" in text or any(k in text for k in ["焚き火", "暖炉", "炎", "薪"]):
        return _procedural_fire(sr, duration_s)
    if "crowd" in text or any(k in text for k in ["雑踏", "群衆", "人混み", "市場"]):
        return _procedural_crowd(sr, duration_s)
    if "forest" in text or any(k in text for k in ["森", "林", "木々", "虫の音"]):
        return _procedural_wind(sr, duration_s) * 0.75
    if "waves" in text or any(k in text for k in ["波", "潮騒", "海鳴り"]):
        return _procedural_wind(sr, duration_s) * 0.9
    if "thunder" in text or any(k in text for k in ["雷", "ごろごろ"]):
        base = _procedural_rain(sr, duration_s) * 0.5
        return _normalize_peak(base, peak=0.18)
    return _procedural_wind(sr, duration_s) * 0.65


def _catalog_or_procedural_foley(seg: dict[str, Any], sr: int) -> tuple[np.ndarray, float | None]:
    cue = str(seg.get("cue") or seg.get("text") or "").strip()
    path, catalog_gain = _catalog_entry("foley", cue)
    if path and os.path.exists(path):
        audio, src_sr = _load_wav_mono_f32(path)
        if src_sr != sr:
            audio = _resample_linear(audio, src_sr, sr)
        return audio.astype(np.float32), catalog_gain
    return _foley_from_cue(cue, sr), catalog_gain


def _loop_to_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if audio.size >= target_len:
        return audio[:target_len].astype(np.float32)
    reps = int(math.ceil(target_len / float(audio.size)))
    tiled = np.tile(audio, reps)
    return tiled[:target_len].astype(np.float32)


def _catalog_or_procedural_ambience(seg: dict[str, Any], sr: int, target_len: int) -> tuple[np.ndarray, float | None]:
    cue = str(seg.get("cue") or seg.get("text") or "").strip()
    path, catalog_gain = _catalog_entry("ambience", cue)
    if path and os.path.exists(path):
        audio, src_sr = _load_wav_mono_f32(path)
        if src_sr != sr:
            audio = _resample_linear(audio, src_sr, sr)
        audio = _loop_to_length(audio.astype(np.float32), target_len)
        return audio, catalog_gain

    duration_s = max(target_len / float(sr), 0.35)
    audio = _ambience_from_cue(cue, sr, duration_s)
    audio = _loop_to_length(audio.astype(np.float32), target_len)
    return audio, catalog_gain


async def _tts_synthesize_async(tts, text: str, **kwargs) -> tuple[np.ndarray, int]:
    if hasattr(tts, "synthesize_async"):
        audio, sr = await tts.synthesize_async(text, **kwargs)
        return audio.astype(np.float32), int(sr)
    audio, sr = tts.synthesize(text, **kwargs)
    return audio.astype(np.float32), int(sr)


async def render_scene_audio_async(
    *,
    tts,
    render_plan: dict[str, Any] | None,
    fallback_text: str,
    tts_kwargs: dict[str, Any],
) -> tuple[np.ndarray, int]:
    plan = render_plan if isinstance(render_plan, dict) else {}
    segments = [s for s in (plan.get("segments") or []) if isinstance(s, dict)]

    speech_segments = [
        s for s in segments
        if str(s.get("kind") or "").lower() == "speech" and bool(s.get("audible", True))
    ]

    if not speech_segments:
        text = str(plan.get("speech_text") or fallback_text or "").strip()
        if text:
            speech_segments = [{"kind": "speech", "text": text, "audible": True}]

    sr: int | None = None
    speech_chunks: list[np.ndarray] = []

    for seg in speech_segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        audio, seg_sr = await _tts_synthesize_async(tts, text, **tts_kwargs)
        if sr is None:
            sr = seg_sr
        elif seg_sr != sr:
            audio = _resample_linear(audio, seg_sr, sr)
        speech_chunks.append(audio.astype(np.float32))

    if sr is None:
        sr = DEFAULT_SCENE_SR

    lead_in_chunks: list[np.ndarray] = []
    for seg in segments:
        kind = str(seg.get("kind") or "").lower()
        placement = str(seg.get("placement") or "lead_in").lower()
        audible = bool(seg.get("audible", True))
        if not audible or kind != "foley" or placement != "lead_in":
            continue
        audio, catalog_gain = _catalog_or_procedural_foley(seg, sr)
        gain_db = float(seg.get("level_db", -10.0))
        if catalog_gain is not None:
            gain_db += float(catalog_gain)
        lead_in_chunks.append(audio.astype(np.float32) * _db_to_gain(gain_db))

    lead_in = _concat_chunks(lead_in_chunks, sr, gap_ms=60)
    speech_audio = _concat_chunks(
        speech_chunks,
        sr,
        gap_ms=int(os.getenv("VOICE_BRIDGE_SPEECH_GAP_MS", "120")),
    )

    if lead_in.size > 0 and speech_audio.size > 0:
        base = np.concatenate([lead_in, speech_audio]).astype(np.float32)
    elif lead_in.size > 0:
        base = lead_in.astype(np.float32)
    elif speech_audio.size > 0:
        base = speech_audio.astype(np.float32)
    else:
        base = _silence(sr, 250)

    total = base.astype(np.float32)

    for seg in segments:
        kind = str(seg.get("kind") or "").lower()
        placement = str(seg.get("placement") or "").lower()
        audible = bool(seg.get("audible", True))
        if not audible or kind != "ambience" or placement != "underlay":
            continue

        bed, catalog_gain = _catalog_or_procedural_ambience(seg, sr, total.size)
        gain_db = float(seg.get("level_db", -26.0))
        if catalog_gain is not None:
            gain_db += float(catalog_gain)
        total = _mix(total, bed, offset=0, gain_db=gain_db)

    total = total.astype(np.float32)
    peak = float(np.max(np.abs(total))) if total.size else 0.0
    if peak > 0.99:
        total = (total / peak * 0.97).astype(np.float32)

    return total.astype(np.float32), int(sr)
