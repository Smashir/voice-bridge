"""
ASR module using faster-whisper.

Input expected:
- 16kHz mono float32 waveform in [-1, 1]

Output:
- transcribed text (Japanese by default)
"""

import numpy as np
from faster_whisper import WhisperModel


class WhisperASR:
    def __init__(self, model_name: str = "small", device: str = "auto"):
        # compute_type=int8: speed/ram friendly; adjust if you want higher accuracy
        self.model = WhisperModel(model_name, device=device, compute_type="int8")

    def transcribe_16k_mono(self, audio_f32: np.ndarray, language: str = "ja") -> str:
        if audio_f32.size == 0:
            return ""
        segments, _info = self.model.transcribe(
            audio_f32,
            language=language,
            vad_filter=False,   # we handle VAD in standalone; server mode may receive already-trimmed audio
            beam_size=1,
        )
        return "".join(s.text for s in segments).strip()
