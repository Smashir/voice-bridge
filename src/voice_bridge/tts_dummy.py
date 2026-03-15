"""
Dummy TTS for pipeline testing:
- Returns short silence so you can verify "everything else works"
  even if your external TTS server is not running.
"""

import numpy as np


class DummyTTS:
    def synthesize(self, text: str, **kwargs):
        sr = 24000
        return np.zeros(int(sr * 0.2), dtype=np.float32), sr
