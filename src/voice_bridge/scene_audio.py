"""Scene-audio renderer for voice-bridge.

Purpose:
- keep ordinary TTS path working as-is
- when render_plan.segments exists, render:
    * speech   -> existing TTS backend
    * foley    -> catalog asset or procedural fallback
    * ambience -> catalog asset or procedural fallback
    * music    -> catalog asset or simple procedural bed fallback
- return one mono float32 waveform
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import subprocess
import wave
from functools import lru_cache
from typing import Any

import numpy as np
from voice_bridge.audio_arranger import build_scene_timeline, render_timeline


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
    """
    Load a catalog audio asset as mono float32.

    WAV is read directly.
    Non-WAV files such as mp3 are decoded through ffmpeg into 24kHz mono PCM.

    The function name is kept for compatibility with existing call sites.
    """
    lower = str(path).lower()

    if lower.endswith(".wav"):
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

    ffmpeg_bin = os.getenv("VOICE_BRIDGE_FFMPEG", "ffmpeg")
    cmd = [
        ffmpeg_bin,
        "-v", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(DEFAULT_SCENE_SR),
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-",
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"ffmpeg not found. Install ffmpeg or set VOICE_BRIDGE_FFMPEG: {path}"
        ) from e
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"ffmpeg decode failed: {path}: {err}") from e

    audio = np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    return audio.astype(np.float32), int(DEFAULT_SCENE_SR)


def _normalize_cue(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"[【】\[\](){}<>「」『』:：\s]+", "", value)
    return value.lower()


def _segment_type(seg: dict[str, Any]) -> str:
    return str(seg.get("type") or seg.get("kind") or "").lower()


def _segment_display_text(seg: dict[str, Any]) -> str:
    return str(seg.get("display_text") or seg.get("text") or "").strip()


def _segment_spoken_text(seg: dict[str, Any]) -> str:
    return str(seg.get("spoken_text") or seg.get("text") or seg.get("display_text") or "").strip()


def _segment_cue(seg: dict[str, Any]) -> str:
    return str(seg.get("cue") or seg.get("prompt") or seg.get("display_text") or seg.get("text") or "").strip()


def _segment_audio_query(seg: dict[str, Any]) -> str:
    """Build resolver query from cue + physical description fields."""
    parts: list[str] = []

    for key in (
        "cue",
        "prompt",
        "description",
        "material",
        "action",
        "count",
        "scene",
        "surface",
        "source",
    ):
        value = seg.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in parts:
            parts.append(text)

    return " ".join(parts).strip()


def _env_float(name: str, default: float = 0.0) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _segment_gain_db(seg: dict[str, Any], kind: str) -> float:
    """Return final gain relative to raw source 0 dB.

    Default behavior:
      - ignore render_plan level_db
      - use environment variable per audio kind
      - if env is missing, use 0 dB

    Optional:
      - VOICE_BRIDGE_USE_RENDER_PLAN_LEVEL_DB=1 makes explicit segment level_db win.
    """
    if _env_bool("VOICE_BRIDGE_USE_RENDER_PLAN_LEVEL_DB", False):
        value = seg.get("level_db")
        if value is not None:
            try:
                return float(value)
            except Exception:
                pass

    env_by_kind = {
        "foley": "VOICE_BRIDGE_FOLEY_GAIN_DB",
        "ambience": "VOICE_BRIDGE_AMBIENCE_GAIN_DB",
        "music": "VOICE_BRIDGE_MUSIC_GAIN_DB",
        "speech": "VOICE_BRIDGE_SPEECH_GAIN_DB",
    }

    return _env_float(env_by_kind.get(kind, "VOICE_BRIDGE_SFX_GAIN_DB"), 0.0)


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, dict[str, Any]]:
    path = os.getenv("VOICE_BRIDGE_SFX_CATALOG", "").strip()
    if not path:
        return {"foley": {}, "ambience": {}, "music": {}}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"foley": {}, "ambience": {}, "music": {}}
        out = {"foley": {}, "ambience": {}, "music": {}}
        for kind in ("foley", "ambience", "music"):
            src = data.get(kind, {})
            if isinstance(src, dict):
                out[kind] = src
        return out
    except Exception as e:
        print(f"[voice-bridge] WARN: failed to load VOICE_BRIDGE_SFX_CATALOG '{path}': {e}")
        return {"foley": {}, "ambience": {}, "music": {}}


@lru_cache(maxsize=1)
def _load_sfx_resolver_records() -> list[dict[str, Any]]:
    records_env = os.getenv("VOICE_BRIDGE_SFX_RECORDS", "").strip()
    if not records_env:
        return []

    try:
        from voice_bridge.sfx_asset_resolver import load_records

        records = load_records()
        if not isinstance(records, list):
            return []
        return records
    except Exception as e:
        print(f"[voice-bridge] WARN: failed to load SFX records: {type(e).__name__}: {e}")
        return []


def _resolver_entry(kind: str, cue: str) -> tuple[str | None, float | None]:
    cue = (cue or "").strip()
    if not cue:
        return None, None

    records = _load_sfx_resolver_records()
    if not records:
        return None, None

    def _env_float_local(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, str(default)))
        except Exception:
            return float(default)

    def _env_bool_local(name: str, default: bool) -> bool:
        raw = os.getenv(name)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    base_min_score = _env_float_local(
        f"VOICE_BRIDGE_{kind.upper()}_MIN_SCORE",
        _env_float_local("VOICE_BRIDGE_SFX_MIN_SCORE", 0.0),
    )
    mixed_foley_min_score = _env_float_local("VOICE_BRIDGE_MIXED_FOLEY_MIN_SCORE", 140.0)
    mixed_foley_max_sec = _env_float_local("VOICE_BRIDGE_MIXED_FOLEY_MAX_SEC", 5.0)
    allow_mixed_foley = _env_bool_local("VOICE_BRIDGE_ALLOW_MIXED_FOLEY", True)

    constraints: dict[str, Any] = {}

    if kind == "ambience":
        constraints["usages"] = ["ambience"]
        constraints["prefer_continuity"] = "continuous"
        constraints["allow_mixed"] = True
    elif kind == "foley":
        constraints["usages"] = ["foley"]
        constraints["prefer_composition"] = "isolated"
        constraints["allow_mixed"] = True
    elif kind == "music":
        constraints["usages"] = ["music"]

    try:
        from voice_bridge.sfx_asset_resolver import resolve_sfx

        candidates = resolve_sfx(
            cue,
            top=5,
            records=records,
            constraints=constraints,
            min_score=base_min_score,
        )
    except Exception as e:
        print(f"[voice-bridge] WARN: SFX resolver failed for cue={cue!r}: {type(e).__name__}: {e}")
        return None, None

    if not candidates:
        if os.getenv("VOICE_BRIDGE_SFX_DEBUG", "").strip():
            print(
                "[voice-bridge] SFX unresolved:",
                f"kind={kind}",
                f"cue={cue!r}",
                f"min_score={base_min_score:.1f}",
            )
        return None, None

    def _duration(rec: dict[str, Any]) -> float | None:
        for key in ("asset_duration_sec", "duration_sec"):
            value = rec.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except Exception:
                pass
        return None

    def _looks_mixed(rec: dict[str, Any]) -> bool:
        composition = str(rec.get("composition") or "").lower()
        if composition == "mixed":
            return True

        text = " ".join(
            str(rec.get(k) or "")
            for k in ("id", "title", "filename", "path", "note", "raw_text", "embedding_text")
        )
        text_l = text.lower()

        if "mix音源" in text:
            return True
        if "mixed" in text_l:
            return True
        return False

    chosen = None

    for cand in candidates:
        rec = cand.record
        path = rec.get("path")
        if not path:
            continue

        if kind == "foley" and _looks_mixed(rec):
            dur = _duration(rec)

            if not allow_mixed_foley:
                if os.getenv("VOICE_BRIDGE_SFX_DEBUG", "").strip():
                    print(
                        "[voice-bridge] SFX rejected:",
                        f"kind={kind}",
                        f"cue={cue!r}",
                        f"id={rec.get('id')}",
                        f"score={cand.score:.1f}",
                        "reason=mixed_not_allowed",
                    )
                continue

            if cand.score < mixed_foley_min_score:
                if os.getenv("VOICE_BRIDGE_SFX_DEBUG", "").strip():
                    print(
                        "[voice-bridge] SFX rejected:",
                        f"kind={kind}",
                        f"cue={cue!r}",
                        f"id={rec.get('id')}",
                        f"score={cand.score:.1f}",
                        f"required={mixed_foley_min_score:.1f}",
                        "reason=mixed_low_score",
                    )
                continue

            if dur is not None and mixed_foley_max_sec > 0.0 and dur > mixed_foley_max_sec:
                if os.getenv("VOICE_BRIDGE_SFX_DEBUG", "").strip():
                    print(
                        "[voice-bridge] SFX rejected:",
                        f"kind={kind}",
                        f"cue={cue!r}",
                        f"id={rec.get('id')}",
                        f"duration={dur:.2f}s",
                        f"max={mixed_foley_max_sec:.2f}s",
                        "reason=mixed_too_long",
                    )
                continue

        chosen = cand
        break

    if chosen is None:
        if os.getenv("VOICE_BRIDGE_SFX_DEBUG", "").strip():
            print(
                "[voice-bridge] SFX unresolved after filtering:",
                f"kind={kind}",
                f"cue={cue!r}",
            )
        return None, None

    rec = chosen.record
    path = rec.get("path")
    if not path:
        return None, None

    if os.getenv("VOICE_BRIDGE_SFX_DEBUG", "").strip():
        print(
            "[voice-bridge] SFX resolved:",
            f"kind={kind}",
            f"cue={cue!r}",
            f"id={rec.get('id')}",
            f"score={chosen.score:.1f}",
            f"path={path}",
        )

    return str(path), None

def _catalog_entry(kind: str, cue: str) -> tuple[str | None, float | None]:
    table = _load_catalog().get(kind, {})

    raw_key = (cue or "").strip()
    norm_key = _normalize_cue(raw_key)

    if isinstance(table, dict) and table:
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

    # Fallback: provider-independent records resolver.
    return _resolver_entry(kind, raw_key)


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


def _procedural_glass_break(sr: int) -> np.ndarray:
    hit = _procedural_thud(sr) * 0.55
    shatter = _procedural_creak(sr) * 0.35
    tail = _procedural_swish(sr) * 0.28
    gap = _silence(sr, 35)
    return _normalize_peak(np.concatenate([hit, gap, shatter, tail]).astype(np.float32), peak=0.92)


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


def _procedural_music_pad(sr: int, duration_s: float) -> np.ndarray:
    n = max(int(sr * duration_s), int(sr * 1.0))
    t = np.linspace(0.0, n / sr, n, endpoint=False)
    chord = (
        0.22 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.18 * np.sin(2.0 * np.pi * 277.18 * t)
        + 0.16 * np.sin(2.0 * np.pi * 329.63 * t)
    )
    lfo = 0.55 + 0.45 * np.sin(2.0 * np.pi * 0.09 * t + 0.3)
    shimmer = _highpass(np.random.randn(n).astype(np.float32), sr, 2500.0) * 0.02
    audio = (chord * lfo + shimmer).astype(np.float32)
    return _normalize_peak(audio, peak=0.16)


def _foley_from_cue(cue: str, sr: int) -> np.ndarray:
    text = cue or ""
    if any(k in text for k in ["カシャン", "ガシャン", "パリン", "割れる", "割れ", "砕け", "破片", "ガラス", "コップ"]):
        return _procedural_glass_break(sr)
    if any(k in text for k in ["ぎし", "きし", "ミシ", "軋"]):
        return _procedural_creak(sr)
    if any(k in text for k in ["こと", "こつ", "コト", "コツ", "とん", "トン", "コン", "コンッ"]):
        return _procedural_knock(sr)
    if any(k in text for k in ["どさ", "どん", "ドサ", "ドン", "ばた", "バタ", "落ちる", "落下"]):
        return _procedural_thud(sr)
    if any(k in text for k in ["さら", "しゃ", "しゅ", "衣擦", "布", "カサカサ", "ざらり"]):
        return _procedural_swish(sr)
    return _procedural_knock(sr)


def _ambience_from_cue(cue: str, sr: int, duration_s: float) -> np.ndarray:
    text = cue or ""
    if "rain" in text.lower() or any(k in text for k in ["雨", "ざぁ", "ザー"]):
        return _procedural_rain(sr, duration_s)
    if "wind" in text.lower() or any(k in text for k in ["風", "びゅ", "ヒュ"]):
        return _procedural_wind(sr, duration_s)
    if "fire" in text.lower() or any(k in text for k in ["焚き火", "暖炉", "炎", "薪"]):
        return _procedural_fire(sr, duration_s)
    if "crowd" in text.lower() or any(k in text for k in ["雑踏", "群衆", "人混み", "市場"]):
        return _procedural_crowd(sr, duration_s)
    if "forest" in text.lower() or any(k in text for k in ["森", "林", "木々", "虫の音"]):
        return _procedural_wind(sr, duration_s) * 0.75
    if "waves" in text.lower() or any(k in text for k in ["波", "潮騒", "海鳴り"]):
        return _procedural_wind(sr, duration_s) * 0.9
    if "thunder" in text.lower() or any(k in text for k in ["雷", "ごろごろ"]):
        base = _procedural_rain(sr, duration_s) * 0.5
        return _normalize_peak(base, peak=0.18)
    return _procedural_wind(sr, duration_s) * 0.65


def _music_from_cue(cue: str, sr: int, duration_s: float) -> np.ndarray:
    text = cue or ""
    bed = _procedural_music_pad(sr, duration_s)
    if any(k in text.lower() for k in ["cafe", "jazz", "lounge", "bossa", "piano"]) or any(k in text for k in ["カフェ", "喫茶", "店内BGM"]):
        crowd = _procedural_crowd(sr, duration_s) * 0.35
        mixed = bed + crowd
        return _normalize_peak(mixed.astype(np.float32), peak=0.15)
    return bed


def _catalog_or_procedural_foley(seg: dict[str, Any], sr: int) -> tuple[np.ndarray, float | None]:
    cue = _segment_cue(seg)
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
    cue = _segment_cue(seg)
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


def _catalog_or_procedural_music(seg: dict[str, Any], sr: int, target_len: int) -> tuple[np.ndarray, float | None]:
    cue = _segment_cue(seg)
    path, catalog_gain = _catalog_entry("music", cue)
    if path and os.path.exists(path):
        audio, src_sr = _load_wav_mono_f32(path)
        if src_sr != sr:
            audio = _resample_linear(audio, src_sr, sr)
        audio = _loop_to_length(audio.astype(np.float32), target_len)
        return audio, catalog_gain

    duration_s = max(target_len / float(sr), 1.0)
    audio = _music_from_cue(cue, sr, duration_s)
    audio = _loop_to_length(audio.astype(np.float32), target_len)
    return audio, catalog_gain


async def _tts_synthesize_async(tts, text: str, **kwargs) -> tuple[np.ndarray, int]:
    if hasattr(tts, "synthesize_async"):
        audio, sr = await tts.synthesize_async(text, **kwargs)
        return audio.astype(np.float32), int(sr)
    audio, sr = await asyncio.to_thread(tts.synthesize, text, **kwargs)
    return audio.astype(np.float32), int(sr)


async def render_scene_audio_async(
    *,
    tts,
    render_plan: dict[str, Any] | None,
    fallback_text: str,
    tts_kwargs: dict[str, Any],
) -> tuple[np.ndarray, int]:
    """Render speech + scene SFX.

    New arranger behavior:
    - ambience/music starts at 0.0 sec and covers the whole scene
    - before_speech foley plays after a short lead-in
    - speech starts after foley + gap
    - after_speech foley plays after the spoken line
    - ambience/music is looped/trimmed, faded, and ducked during speech

    Safety:
    - If VOICE_BRIDGE_SCENE_ARRANGER=0, this still falls back to a simple
      speech-first mix path.
    - If no render_plan/segments are present, ordinary TTS remains unchanged.
    """

    text = (fallback_text or "").strip()
    plan = render_plan if isinstance(render_plan, dict) else None
    segments = plan.get("segments") if isinstance(plan, dict) else None

    if not isinstance(segments, list) or not segments:
        audio, sr = await _tts_synthesize_async(tts, text, **tts_kwargs)
        return audio.astype(np.float32), int(sr)

    speech_texts: list[str] = []
    foley_segments: list[dict[str, Any]] = []
    ambience_segments: list[dict[str, Any]] = []
    music_segments: list[dict[str, Any]] = []

    for seg in segments:
        if not isinstance(seg, dict):
            continue

        typ = _segment_type(seg)

        if typ == "speech" and bool(seg.get("audible", True)):
            st = _segment_spoken_text(seg)
            if st:
                speech_texts.append(st)
            continue

        if typ == "foley" and bool(seg.get("audible", True)):
            foley_segments.append(seg)
            continue

        if typ == "ambience" and bool(seg.get("audible", True)):
            ambience_segments.append(seg)
            continue

        if typ == "music" and bool(seg.get("audible", True)):
            music_segments.append(seg)
            continue

    speech_text = "\n".join(speech_texts).strip() or text

    speech_audio, speech_sr = await _tts_synthesize_async(tts, speech_text, **tts_kwargs)
    sr = int(speech_sr)
    speech_audio = speech_audio.astype(np.float32)

    def _load_segment_audio(kind: str, seg: dict[str, Any], duration_hint_sec: float = 4.0) -> np.ndarray:
        cue = _segment_cue(seg)
        resolver_query = _segment_audio_query(seg) or cue
        path, catalog_gain = _catalog_entry(kind, resolver_query)

        if path:
            try:
                audio, asset_sr = _load_wav_mono_f32(path)
                if int(asset_sr) != sr:
                    audio = _resample_linear(audio, int(asset_sr), sr)
                return audio.astype(np.float32)
            except Exception as e:
                print(f"[voice-bridge] WARN: failed to load SFX asset kind={kind} cue={cue!r}: {e}")

        # Keep a small procedural safety net for common cases.
        q = (cue + " " + resolver_query).lower()

        if kind == "ambience":
            if "雨" in cue or "rain" in q:
                return _procedural_rain(sr, duration_hint_sec).astype(np.float32)
            if "風" in cue or "wind" in q:
                return _procedural_wind(sr, duration_hint_sec).astype(np.float32)

        if kind == "foley":
            if "ノック" in cue or "knock" in q or "叩" in cue:
                one = _procedural_knock(sr)
                gap = _silence(sr, 140)
                return np.concatenate([one, gap, one, gap, one]).astype(np.float32)
            if "ドア" in cue or "扉" in cue:
                return _procedural_creak(sr).astype(np.float32)
            if "ガラス" in cue:
                return _procedural_glass_break(sr).astype(np.float32)

        return np.zeros(0, dtype=np.float32)

    foley_items: list[dict[str, Any]] = []
    for seg in foley_segments:
        audio = _load_segment_audio("foley", seg, duration_hint_sec=1.0)
        if audio.size == 0:
            continue
        level = seg.get("level_db")
        foley_items.append(
            {
                "segment": seg,
                "cue": _segment_cue(seg),
                "audio": audio,
                "gain_db": _segment_gain_db(seg, "foley"),
            }
        )

    ambience_items: list[dict[str, Any]] = []
    for seg in ambience_segments:
        audio = _load_segment_audio("ambience", seg, duration_hint_sec=8.0)
        if audio.size == 0:
            continue
        level = seg.get("level_db")
        ambience_items.append(
            {
                "segment": seg,
                "cue": _segment_cue(seg),
                "audio": audio,
                "gain_db": _segment_gain_db(seg, "ambience"),
            }
        )

    music_items: list[dict[str, Any]] = []
    for seg in music_segments:
        audio = _load_segment_audio("music", seg, duration_hint_sec=8.0)
        if audio.size == 0:
            continue
        level = seg.get("level_db")
        music_items.append(
            {
                "segment": seg,
                "cue": _segment_cue(seg),
                "audio": audio,
                "gain_db": _segment_gain_db(seg, "music"),
            }
        )

    use_arranger = os.getenv("VOICE_BRIDGE_SCENE_ARRANGER", "1").strip().lower() not in {"0", "false", "off", "no"}

    if use_arranger:
        timeline = build_scene_timeline(
            speech_audio=speech_audio,
            speech_text=speech_text,
            foley_items=foley_items,
            ambience_items=ambience_items,
            music_items=music_items,
            sr=sr,
        )
        return render_timeline(timeline, sr), sr

    # Legacy-ish fallback: speech first, then simple overlays at 0 sec.
    out = speech_audio.astype(np.float32)
    for item in ambience_items + music_items + foley_items:
        out = _mix(out, item["audio"], offset=0, gain_db=float(item.get("gain_db", 0.0)))

    return _normalize_peak(out), sr
