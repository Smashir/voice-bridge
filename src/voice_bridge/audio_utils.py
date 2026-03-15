"""
Audio utilities:

- Convert float32 waveform to WAV bytes (16-bit PCM).
- Convert uploaded audio bytes to 16kHz mono float32.

Notes:
- If input isn't WAV PCM, we fall back to ffmpeg (if installed).
"""

import io
import wave
import numpy as np
import subprocess
import shutil


def f32_to_wav_bytes(audio_f32: np.ndarray, sr: int) -> bytes:
    """float32 [-1,1] -> WAV (PCM16) bytes."""
    audio_i16 = np.clip(audio_f32, -1.0, 1.0)
    audio_i16 = (audio_i16 * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16
        wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    return buf.getvalue()


def bytes_to_16k_mono_f32(audio_bytes: bytes) -> np.ndarray:
    """
    Try to decode audio bytes into 16kHz mono float32.

    1) Try wave module (WAV PCM).
    2) If not WAV, use ffmpeg to convert to WAV(16k mono) then decode.
    """
    # 1) Try WAV PCM
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            raw = wf.readframes(n)
            audio_i16 = np.frombuffer(raw, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32768.0
        if sr != 16000:
            audio_f32 = _resample_linear(audio_f32, sr, 16000)
        return audio_f32.astype(np.float32)
    except Exception:
        pass

    # 2) ffmpeg fallback
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("Non-WAV audio received. Install ffmpeg or send WAV PCM.")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", "pipe:0",
        "-ac", "1", "-ar", "16000",
        "-f", "wav", "pipe:1"
    ]
    p = subprocess.run(cmd, input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        err = p.stderr.decode("utf-8", errors="ignore")[:200]
        raise RuntimeError(f"ffmpeg decode failed: {err}")

    return bytes_to_16k_mono_f32(p.stdout)


def _resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Simple linear resampler (good enough for ASR input)."""
    if sr_in == sr_out or x.size == 0:
        return x
    ratio = sr_out / sr_in
    n_out = int(round(x.size * ratio))
    idx = np.linspace(0, x.size - 1, n_out)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, x.size - 1)
    w = idx - i0
    return (1 - w) * x[i0] + w * x[i1]
