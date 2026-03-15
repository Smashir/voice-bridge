"""
Standalone mode (no OpenWebUI):

- This process holds chat history in-memory (messages list).
- Each turn sends full messages to LLM server (OpenAI-compatible).
- Audio input is recorded from microphone using VAD.
- TTS is external HTTP (or dummy) selected by env vars.

This is intentionally single-user / single-session.
"""

import time
import queue
import numpy as np
import sounddevice as sd
import webrtcvad

from voice_bridge.config import Settings
from voice_bridge.asr_whisper import WhisperASR
from voice_bridge.gar_client import chat_sync
from voice_bridge.tts_base import load_tts_config_from_env, build_tts

settings = Settings()


def record_utterance_vad(
    fs: int = 16000,
    frame_ms: int = 30,
    vad_level: int = 2,
    max_seconds: float = 20.0,
    end_silence_ms: int = 600,
) -> np.ndarray:
    """
    Record from microphone until end-of-speech is detected.

    Returns:
      float32 mono waveform in [-1, 1], sr=fs
    """
    vad = webrtcvad.Vad(vad_level)
    frame_len = int(fs * frame_ms / 1000)
    bytes_per_frame = frame_len * 2

    q: "queue.Queue[bytes]" = queue.Queue()
    voiced: list[bytes] = []
    silence_run_ms = 0
    started = False
    start_time = time.time()

    def callback(indata, frames, time_info, status):
        mono = indata[:, 0]
        i16 = np.clip(mono * 32767.0, -32768, 32767).astype(np.int16)
        q.put(i16.tobytes())

    with sd.InputStream(
        samplerate=fs,
        channels=1,
        dtype="float32",
        blocksize=frame_len,
        callback=callback,
    ):
        while True:
            if time.time() - start_time > max_seconds:
                break

            b = q.get()
            if len(b) != bytes_per_frame:
                continue

            is_speech = vad.is_speech(b, fs)
            if is_speech:
                started = True
                silence_run_ms = 0
                voiced.append(b)
            else:
                if started:
                    silence_run_ms += frame_ms
                    voiced.append(b)
                    if silence_run_ms >= end_silence_ms:
                        break

    if not voiced:
        return np.zeros((0,), dtype=np.float32)

    audio_i16 = np.frombuffer(b"".join(voiced), dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0).copy()


def play_audio(audio: np.ndarray, sr: int):
    """Play audio via default output device."""
    if audio.size == 0:
        return
    sd.play(audio, sr)
    sd.wait()


def run():
    asr = WhisperASR(settings.asr_model, settings.asr_device)
    tts = build_tts(load_tts_config_from_env())

    # In standalone mode, THIS list is the conversation memory.
    messages = [
        {"role": "system", "content": "あなたは自然な会話をするアシスタントです。短めに返答してください。"}
    ]

    print("standalone ready. Speak. (Ctrl+C to exit)")
    try:
        while True:
            audio = record_utterance_vad()
            user_text = asr.transcribe_16k_mono(audio, language="ja")
            if not user_text:
                continue

            print(f"\n[YOU] {user_text}")
            messages.append({"role": "user", "content": user_text})

            body = {
                "model": "local-llm",
                "messages": messages,
                "temperature": 0.7,
                "stream": False,
            }
            resp = chat_sync(settings.llm_chat_url, body)

            assistant_text = resp["choices"][0]["message"]["content"]
            print(f"[BOT] {assistant_text}")
            messages.append({"role": "assistant", "content": assistant_text})

            audio_f32, sr = tts.synthesize(assistant_text)
            play_audio(audio_f32, sr)

    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    run()
