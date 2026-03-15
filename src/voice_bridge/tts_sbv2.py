"""
Style-Bert-VITS2 TTS backend.

- Calls Style-Bert-VITS2 server /voice endpoint (returns audio/wav).
- Decodes WAV -> float32 waveform + sample_rate to match Voice Bridge TTS interface.
- Resolves model_name -> model_id via /models/info (cached).
- Supports "voice_id" format: "sbv2:<model_name>:<speaker_id>".
  (engine namespace is kept outside; we only care about the local key part here.)
"""

from __future__ import annotations

import io
import wave
import time
import re
import os
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import httpx

import math



@dataclass
class SBV2Config:
    base_url: str # e.g. http://192.168.0.75:5000
    timeout_s: int = 120
    models_timeout_s: int = 3   # ★追加：/models/info専用


class StyleBertVITS2TTS:
    def __init__(self, cfg: SBV2Config):
        self.cfg = cfg
        self._models_cache: dict[str, Any] | None = None
        self._models_cache_at: float = 0.0
        self._models_cache_ttl_s: float = 30.0  # refresh occasionally
        self._models_info_raw_cache = None
        self._style_map = self._load_style_map_from_env()

    def _resolve_model_name_by_speaker(self, speaker_name: str, models_info_raw: dict) -> str | None:
        # SBV2 /models/info の生JSONから、spk2idに speaker_name を含むモデルを探す
        for model_id_str, info in models_info_raw.items():
            spk2id = info.get("spk2id", {}) or {}
            if speaker_name in spk2id:
                # config_path: model_assets/<model_name>/config.json から model_name を抜く
                cfg_path = info.get("config_path", "")
                if isinstance(cfg_path, str):
                    m = re.search(r"model_assets/([^/]+)/", cfg_path)
                    if m:
                        return m.group(1)
        return None

    def _decode_wav_to_f32(self, wav_bytes: bytes) -> tuple[np.ndarray, int]:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            sampwidth = wf.getsampwidth()
            nch = wf.getnchannels()
            raw = wf.readframes(n)

        if nch != 1:
            if sampwidth != 2:
                raise RuntimeError(f"unsupported wav format: nch={nch} sampwidth={sampwidth}")
            audio_i16 = np.frombuffer(raw, dtype=np.int16).reshape(-1, nch).mean(axis=1).astype(np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            return audio_f32, sr

        if sampwidth == 2:
            audio_i16 = np.frombuffer(raw, dtype=np.int16)
            audio_f32 = audio_i16.astype(np.float32) / 32768.0
            return audio_f32, sr

        raise RuntimeError(f"unsupported wav sampwidth={sampwidth}")

    def _load_style_map_from_env(self) -> dict[str, dict[str, str]]:
        """
        Load external style map (model_name -> {GAR_STYLE -> SBV2_STYLE_NAME}).

        Path is taken from env VOICE_BRIDGE_STYLE_MAP. If missing or unreadable, returns {}.
        """
        path = os.environ.get("VOICE_BRIDGE_STYLE_MAP", "").strip()
        if not path:
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                out: dict[str, dict[str, str]] = {}
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, dict):
                        out[k] = {str(kk): str(vv) for kk, vv in v.items()}
                return out
        except Exception as e:
            print(f"[sbv2] WARN: failed to load VOICE_BRIDGE_STYLE_MAP '{path}': {e}")
        return {}

    def _apply_external_style_map(
        self,
        model_name: str,
        desired_style: str,
        style_names: set[str],
    ) -> str | None:
        """Map GAR style name to SBV2 style name for the given model, if defined and available."""
        if not desired_style:
            return None
        mp = self._style_map.get(model_name) if isinstance(getattr(self, "_style_map", None), dict) else None
        if not isinstance(mp, dict):
            return None
        mapped = mp.get(desired_style)
        if not mapped:
            return None
        return mapped if mapped in style_names else None

    def _fallback_style_from_id0(self, model_id: int, style_names: set[str]) -> str | None:
        """
        Prefer Neutral. If not present, try to find style key whose id is 0 from /models/info raw cache.
        """
        if "Neutral" in style_names:
            return "Neutral"
        for s in style_names:
            if s.lower() == "neutral":
                return s
        raw = getattr(self, "_models_info_raw_cache", None)
        if isinstance(raw, dict):
            info = raw.get(str(model_id)) or {}
            style2id = info.get("style2id", {}) or {}
            if isinstance(style2id, dict):
                for k, v in style2id.items():
                    try:
                        if int(v) == 0 and k in style_names:
                            return k
                    except Exception:
                        continue
        return next(iter(sorted(style_names))) if style_names else None

    async def _get_models_info_async(self) -> dict:
        now = time.time()
        if self._models_cache and (now - self._models_cache_at) < self._models_cache_ttl_s:
            return self._models_cache

        url = self.cfg.base_url.rstrip("/") + "/models/info"
        try:
            timeout = httpx.Timeout(self.cfg.models_timeout_s, connect=self.cfg.models_timeout_s)
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.get(url, headers={"accept": "application/json"})
                r.raise_for_status()
                data = r.json()
                self._models_info_raw_cache = data

            name_to = {}
            for model_id_str, info in data.items():
                model_id = int(model_id_str)
                cfg_path = info.get("config_path", "")
                m = None
                if isinstance(cfg_path, str):
                    m = re.search(r"model_assets/([^/]+)/", cfg_path)
                model_name = m.group(1) if m else f"model_{model_id}"

                style2id = info.get("style2id", {}) or {}
                name_to[model_name] = {
                    "model_id": model_id,
                    "style_names": set(style2id.keys()),
                }

            self._models_cache = name_to
            self._models_cache_at = now
            return name_to

        except Exception as e:
            # ★ここが肝：取り直しに失敗しても最後のキャッシュで続行
            if self._models_cache:
                print("[sbv2] /models/info failed, using cached models:", repr(e))
                self._models_cache_at = now  # 無限リトライ連打を防ぐ
                return self._models_cache
            raise


    def _split_voice_id(self, voice_id: str) -> tuple[str, int]:
        parts = voice_id.split(":")
        if len(parts) >= 3 and parts[0] == "sbv2":
            model_name = ":".join(parts[1:-1])
            speaker_id = int(parts[-1])
            return model_name, speaker_id
        if len(parts) >= 2:
            model_name = ":".join(parts[:-1])
            speaker_id = int(parts[-1])
            return model_name, speaker_id
        raise ValueError(f"invalid voice_id: {voice_id}")


    def _pick_style_from_available(
        self,
        desired_style: str | None,
        style_weight: float | None,
        style_names: set[str],
    ) -> str | None:
        """
        Choose the best SBV2 style from available style_names.

        - If desired_style exists, use it.
        - Else try loose synonyms (JP/EN).
        - Else if only numeric styles (01/02/...) exist, use style_weight as intensity to pick one.
        - Else fall back to Neutral if available, otherwise None.
        """
        if not style_names:
            return None

        # Normalize for matching
        avail = {s: s.lower() for s in style_names}

        # 1) exact match
        if desired_style and desired_style in style_names:
            return desired_style

        # 2) case-insensitive match
        if desired_style:
            ds = desired_style.lower()
            for s, sl in avail.items():
                if sl == ds:
                    return s

        # 3) synonym match (EN/JP, casual)
        syn = {
            "happy": ["happy", "joy", "cheer", "嬉", "喜", "楽", "るんるん", "ごきげん", "明る"],
            "angry": ["angry", "anger", "mad", "怒", "憤", "激", "苛"],
            "sad": ["sad", "sadness", "cry", "悲", "泣", "沈", "寂"],
            "fear": ["fear", "scared", "panic", "怖", "怯", "震"],
            "disgust": ["disgust", "嫌", "厭", "不快", "吐"],
            "surprise": ["surprise", "shock", "驚", "びっくり"],
            "neutral": ["neutral", "normal", "ノーマル", "通常", "標準"],
            "whisper": ["whisper", "囁", "ささやき", "小声"],
            "calm": ["calm", "落ち着", "穏", "静"],
        }

        key = (desired_style or "").lower()
        for canon, words in syn.items():
            if key == canon or key in words:
                # try find any style containing any synonym token
                for s in style_names:
                    for w in words:
                        if w.lower() in s.lower():
                            return s
                break  # matched canon but not found in available -> continue to numeric / fallback

        # 4) numeric-only styles like 01/02/03/04
        #    (common in some SBV2 models e.g. amitaro)
        numeric = []
        for s in style_names:
            if s.isdigit():
                numeric.append(int(s))
            else:
                # allow "01" form
                if len(s) == 2 and s[0].isdigit() and s[1].isdigit():
                    numeric.append(int(s))
        numeric = sorted(set(numeric))
        if numeric:
            STYLE_WEIGHT_MAX = 20.0

            w = 0.5 if style_weight is None else float(style_weight)
            if w < 0.0:
                w = 0.0
            if w > STYLE_WEIGHT_MAX:
                w = STYLE_WEIGHT_MAX

            # map 0..20 -> index
            idx = int(round((w / STYLE_WEIGHT_MAX) * (len(numeric) - 1)))
            idx = max(0, min(idx, len(numeric) - 1))
            chosen = f"{numeric[idx]:02d}" if any(len(s) == 2 and s.isdigit() for s in style_names) else str(numeric[idx])
            # if exact chosen not present (rare), pick first numeric
            if chosen in style_names:
                return chosen
            for s in style_names:
                if s == str(numeric[idx]) or s == f"{numeric[idx]:02d}":
                    return s
            return next(iter(sorted(style_names)))  # last resort

        # 5) fallback
        if "Neutral" in style_names:
            return "Neutral"
        for s in style_names:
            if s.lower() == "neutral":
                return s
        return None


    async def synthesize_async(
        self,
        text: str,
        *,
        voice_id: str | None = None,
        style: str | None = None,
        style_weight: float | None = None,
    ) -> tuple[np.ndarray, int]:
        if not voice_id:
            raise RuntimeError("StyleBertVITS2TTS requires voice_id like 'sbv2:<model_name>:<speaker_id>'")

        model_name, speaker_id = self._split_voice_id(voice_id)
        models = await self._get_models_info_async()

        if model_name not in models:
            # model_name が実は "話者名" の可能性がある（例: あみたろ）
            raw = getattr(self, "_models_info_raw_cache", None)
            if isinstance(raw, dict):
                resolved = self._resolve_model_name_by_speaker(model_name, raw)
                if resolved and resolved in models:
                    print(f"[sbv2] resolved speaker '{model_name}' -> model '{resolved}'")
                    model_name = resolved

        if model_name not in models:
            raise RuntimeError(f"SBV2 model_name not found: {model_name}. known={list(models.keys())[:10]}")

        model_id = int(models[model_name]["model_id"])
        style_names = models[model_name]["style_names"]

        # Resolve best style for this model:
        # 1) if desired style exists in model, use it
        # 2) else consult external per-model style map (VOICE_BRIDGE_STYLE_MAP)
        # 3) else use heuristic picker (synonyms / numeric intensity)
        # 4) else fall back to Neutral (or id=0)
        desired_style = style
        desired_weight = style_weight

        picked: str | None = None
        if desired_style and desired_style in style_names:
            picked = desired_style
        if picked is None and desired_style:
            picked = self._apply_external_style_map(model_name, desired_style, style_names)
        if picked is None:
            picked = self._pick_style_from_available(desired_style, desired_weight, style_names)
        if picked is None:
            picked = self._fallback_style_from_id0(model_id, style_names)

        print("[sbv2] model_name =", model_name, "model_id =", model_id, "speaker_id =", speaker_id)
        print("[sbv2] desired style/weight =", desired_style, desired_weight)
        print("[sbv2] available styles =", sorted(list(style_names))[:50], ("...(truncated)" if len(style_names) > 50 else ""))
        print("[sbv2] picked style =", picked)

        style = picked

        params = {"text": text, "model_id": model_id, "speaker_id": speaker_id}
        if style is not None:
            params["style"] = style
        if style_weight is not None:
            params["style_weight"] = float(style_weight)

        url = self.cfg.base_url.rstrip("/") + "/voice"
        async with httpx.AsyncClient(timeout=self.cfg.timeout_s) as client:
            r = await client.post(url, params=params)
            r.raise_for_status()
            wav_bytes = r.content

        audio_f32, sr = self._decode_wav_to_f32(wav_bytes)
        return audio_f32, sr

    def synthesize(self, text: str, **kwargs) -> tuple[np.ndarray, int]:
        import asyncio
        return asyncio.run(self.synthesize_async(text, **kwargs))
