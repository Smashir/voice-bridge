"""Deterministic scene audio arranger for voice-bridge.

This module converts resolved speech / foley / ambience items into a simple
timeline. It intentionally avoids LLM-based decisions at this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def db_to_gain(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def sec_to_samples(sec: float, sr: int) -> int:
    return max(0, int(round(float(sec) * int(sr))))


def duration_sec(audio: np.ndarray, sr: int) -> float:
    if not isinstance(audio, np.ndarray) or audio.size == 0:
        return 0.0
    return float(audio.size) / float(sr)


@dataclass
class ArrangedEvent:
    type: str
    cue: str
    start_sec: float
    gain_db: float
    audio: np.ndarray
    fade_in_sec: float = 0.0
    fade_out_sec: float = 0.0
    loop: bool = False
    target_duration_sec: float | None = None
    duck_during_speech_db: float = 0.0
    source_segment: dict[str, Any] | None = None


def normalize_placement(value: str | None) -> str:
    v = (value or "").strip().lower()

    if v in {"lead_in", "pre", "before", "before_speech"}:
        return "before_speech"

    if v in {"post", "outro", "after", "after_speech"}:
        return "after_speech"

    if v in {"underlay", "background", "during", "during_speech"}:
        return "during_speech"

    return "before_speech"


def fade_in_out(audio: np.ndarray, sr: int, fade_in_sec: float, fade_out_sec: float) -> np.ndarray:
    x = audio.astype(np.float32, copy=True)
    n = x.size
    if n <= 0:
        return x

    n_in = min(n, sec_to_samples(fade_in_sec, sr))
    if n_in > 1:
        x[:n_in] *= np.linspace(0.0, 1.0, n_in, dtype=np.float32)

    n_out = min(n, sec_to_samples(fade_out_sec, sr))
    if n_out > 1:
        x[-n_out:] *= np.linspace(1.0, 0.0, n_out, dtype=np.float32)

    return x.astype(np.float32)


def loop_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    target_samples = max(0, int(target_samples))
    if target_samples == 0:
        return np.zeros(0, dtype=np.float32)

    x = audio.astype(np.float32)
    if x.size == 0:
        return np.zeros(target_samples, dtype=np.float32)

    if x.size >= target_samples:
        return x[:target_samples].astype(np.float32)

    reps = int(np.ceil(target_samples / x.size))
    return np.tile(x, reps)[:target_samples].astype(np.float32)


def mix_at(base: np.ndarray, overlay: np.ndarray, sr: int, start_sec: float, gain_db: float = 0.0) -> np.ndarray:
    src = overlay.astype(np.float32) * db_to_gain(gain_db)
    start = sec_to_samples(start_sec, sr)
    n_out = max(base.size, start + src.size)

    out = np.zeros(n_out, dtype=np.float32)
    if base.size:
        out[:base.size] += base.astype(np.float32)
    if src.size:
        out[start:start + src.size] += src

    return out.astype(np.float32)


def apply_ducking(
    audio: np.ndarray,
    sr: int,
    ranges_sec: list[tuple[float, float]],
    duck_db: float,
    ramp_sec: float = 0.12,
) -> np.ndarray:
    if audio.size == 0 or not ranges_sec or duck_db >= 0.0:
        return audio.astype(np.float32)

    x = audio.astype(np.float32, copy=True)
    duck_gain = db_to_gain(duck_db)
    ramp = max(1, sec_to_samples(ramp_sec, sr))
    envelope = np.ones(x.size, dtype=np.float32)

    for start_sec, end_sec in ranges_sec:
        a = min(x.size, sec_to_samples(start_sec, sr))
        b = min(x.size, sec_to_samples(end_sec, sr))
        if b <= a:
            continue

        envelope[a:b] = np.minimum(envelope[a:b], duck_gain)

        left = max(0, a - ramp)
        if a > left:
            envelope[left:a] = np.minimum(
                envelope[left:a],
                np.linspace(1.0, duck_gain, a - left, dtype=np.float32),
            )

        right = min(x.size, b + ramp)
        if right > b:
            envelope[b:right] = np.minimum(
                envelope[b:right],
                np.linspace(duck_gain, 1.0, right - b, dtype=np.float32),
            )

    return (x * envelope).astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    m = float(np.max(np.abs(audio)))
    if m < 1e-8:
        return audio.astype(np.float32)
    if m <= peak:
        return audio.astype(np.float32)
    return (audio / m * peak).astype(np.float32)


def build_scene_timeline(
    *,
    speech_audio: np.ndarray,
    speech_text: str,
    foley_items: list[dict[str, Any]],
    ambience_items: list[dict[str, Any]],
    music_items: list[dict[str, Any]] | None = None,
    sr: int,
) -> dict[str, Any]:
    """Build deterministic event timing for scene audio."""

    music_items = music_items or []

    foley_lead_delay = env_float("VOICE_BRIDGE_FOLEY_LEAD_DELAY_SEC", 1.2)
    gap_after_foley = env_float("VOICE_BRIDGE_GAP_AFTER_FOLEY_SEC", 0.55)
    speech_pre_delay = env_float("VOICE_BRIDGE_SPEECH_PRE_DELAY_SEC", 0.6)
    gap_after_speech = env_float("VOICE_BRIDGE_GAP_AFTER_SPEECH_SEC", 0.8)
    tail_sec = env_float("VOICE_BRIDGE_SCENE_TAIL_SEC", 1.0)

    ambience_gain_default = env_float("VOICE_BRIDGE_AMBIENCE_GAIN_DB", 0.0)
    foley_gain_default = env_float("VOICE_BRIDGE_FOLEY_GAIN_DB", 0.0)
    music_gain_default = env_float("VOICE_BRIDGE_MUSIC_GAIN_DB", 0.0)

    ambience_fade_in = env_float("VOICE_BRIDGE_AMBIENCE_FADE_IN_SEC", 0.6)
    ambience_fade_out = env_float("VOICE_BRIDGE_AMBIENCE_FADE_OUT_SEC", 1.0)
    ambience_duck_db = env_float("VOICE_BRIDGE_AMBIENCE_DUCK_DB", 0.0)
    music_duck_db = env_float("VOICE_BRIDGE_MUSIC_DUCK_DB", 0.0)
    during_foley_max_gain_db = env_float("VOICE_BRIDGE_DURING_FOLEY_MAX_GAIN_DB", 0.0)
    speech_gain_db = env_float("VOICE_BRIDGE_SPEECH_GAIN_DB", 0.0)
    foley_max_sec = env_float("VOICE_BRIDGE_FOLEY_MAX_SEC", 0.0)

    before_foley: list[ArrangedEvent] = []
    during_foley: list[ArrangedEvent] = []
    after_foley: list[ArrangedEvent] = []

    t_cursor = foley_lead_delay

    for item in foley_items:
        seg = item.get("segment") or {}
        placement = normalize_placement(seg.get("placement"))

        audio = item.get("audio")
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            continue

        if foley_max_sec > 0.0:
            max_samples = sec_to_samples(foley_max_sec, sr)
            if audio.size > max_samples:
                audio = audio[:max_samples].astype(np.float32)

        gain_db = float(item.get("gain_db", foley_gain_default))
        cue = str(item.get("cue") or seg.get("cue") or "")

        ev = ArrangedEvent(
            type="foley",
            cue=cue,
            start_sec=t_cursor,
            gain_db=gain_db,
            audio=audio.astype(np.float32),
            source_segment=seg,
        )

        if placement == "after_speech":
            after_foley.append(ev)
        elif placement == "during_speech":
            during_foley.append(ev)
        else:
            before_foley.append(ev)
            t_cursor += duration_sec(audio, sr) + gap_after_foley

    if before_foley:
        speech_start = max(speech_pre_delay, t_cursor)
    else:
        speech_start = speech_pre_delay

    speech_dur = duration_sec(speech_audio, sr)
    speech_end = speech_start + speech_dur

    during_cursor = speech_start + 0.25
    for ev in during_foley:
        ev.start_sec = during_cursor
        ev.gain_db = min(ev.gain_db, during_foley_max_gain_db)
        during_cursor += min(duration_sec(ev.audio, sr) + 0.25, max(0.35, speech_dur / 3.0))

    after_cursor = speech_end + gap_after_speech
    for ev in after_foley:
        ev.start_sec = after_cursor
        after_cursor += duration_sec(ev.audio, sr) + 0.35

    scene_end = max(speech_end, after_cursor) + tail_sec

    events: list[ArrangedEvent] = []

    for item in ambience_items:
        seg = item.get("segment") or {}
        audio = item.get("audio")
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            continue

        gain_db = float(item.get("gain_db", ambience_gain_default))
        cue = str(item.get("cue") or seg.get("cue") or "")

        events.append(
            ArrangedEvent(
                type="ambience",
                cue=cue,
                start_sec=0.0,
                gain_db=gain_db,
                audio=audio.astype(np.float32),
                fade_in_sec=ambience_fade_in,
                fade_out_sec=ambience_fade_out,
                loop=True,
                target_duration_sec=scene_end,
                duck_during_speech_db=ambience_duck_db,
                source_segment=seg,
            )
        )

    for item in music_items:
        seg = item.get("segment") or {}
        audio = item.get("audio")
        if not isinstance(audio, np.ndarray) or audio.size == 0:
            continue

        gain_db = float(item.get("gain_db", music_gain_default))
        cue = str(item.get("cue") or seg.get("cue") or "")

        events.append(
            ArrangedEvent(
                type="music",
                cue=cue,
                start_sec=0.0,
                gain_db=gain_db,
                audio=audio.astype(np.float32),
                fade_in_sec=0.8,
                fade_out_sec=1.2,
                loop=True,
                target_duration_sec=scene_end,
                duck_during_speech_db=music_duck_db,
                source_segment=seg,
            )
        )

    events.extend(before_foley)
    events.extend(during_foley)

    events.append(
        ArrangedEvent(
            type="speech",
            cue=speech_text,
            start_sec=speech_start,
            gain_db=speech_gain_db,
            audio=speech_audio.astype(np.float32),
        )
    )

    events.extend(after_foley)

    return {
        "version": "1.0",
        "sample_rate": sr,
        "speech_range_sec": (speech_start, speech_end),
        "scene_end_sec": scene_end,
        "events": events,
    }


def render_timeline(timeline: dict[str, Any], sr: int) -> np.ndarray:
    events: list[ArrangedEvent] = list(timeline.get("events") or [])
    scene_end_sec = float(timeline.get("scene_end_sec") or 0.0)
    total_samples = sec_to_samples(scene_end_sec, sr)

    out = np.zeros(total_samples, dtype=np.float32)

    speech_range = timeline.get("speech_range_sec")
    speech_ranges: list[tuple[float, float]] = []
    if isinstance(speech_range, tuple) and len(speech_range) == 2:
        speech_ranges.append((float(speech_range[0]), float(speech_range[1])))

    for ev in events:
        audio = ev.audio.astype(np.float32)

        if ev.target_duration_sec is not None:
            audio = loop_or_trim(audio, sec_to_samples(ev.target_duration_sec, sr))

        if ev.fade_in_sec > 0.0 or ev.fade_out_sec > 0.0:
            audio = fade_in_out(audio, sr, ev.fade_in_sec, ev.fade_out_sec)

        if ev.duck_during_speech_db < 0.0 and speech_ranges:
            audio = apply_ducking(audio, sr, speech_ranges, ev.duck_during_speech_db)

        out = mix_at(out, audio, sr, ev.start_sec, ev.gain_db)

    if os.getenv("VOICE_BRIDGE_AUDIO_PLAN_DEBUG", "").strip():
        print("[voice-bridge] audio_plan:")
        for ev in events:
            print(
                " ",
                ev.type,
                f"start={ev.start_sec:.2f}s",
                f"dur={duration_sec(ev.audio, sr):.2f}s",
                f"gain={ev.gain_db:.1f}dB",
                f"cue={ev.cue!r}",
            )
        print("  scene_end=", f"{scene_end_sec:.2f}s")

    return normalize_peak(out, peak=0.95)