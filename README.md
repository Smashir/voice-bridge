# Ghost Nexus

Formerly `voice-bridge`.

Ghost Nexus is a narrative and multimodal interaction layer for GAR-LLM.
It connects generated responses and render plans to speech, sound effects,
ambience, foley, UI clients, and future multimodal agent actions.

The internal Python package currently remains `voice_bridge` for compatibility.

# voice-bridge

STT / TTS bridge for **gar-llm**.

`voice-bridge` provides an OpenAI-compatible audio API and connects:

* gar-llm (LLM + persona / emotion engine)
* Style-Bert-VITS2 (TTS)
* OpenWebUI (optional voice UI)

The bridge converts gar-llm emotion signals into speech styles for SBV2.

---

# Architecture

```
User
  ↓
OpenWebUI (optional)
  ↓
voice-bridge
  ├─ STT
  ├─ TTS
  └─ emotion → voice style mapping
  ↓
Style-Bert-VITS2
  ↓
Audio
```

Typical setup:

```
gar-llm        : 8081
voice-bridge   : 8787
SBV2 server    : 5000
```

---

# Features

* OpenAI compatible audio endpoints
* STT / TTS bridge
* emotion → style mapping
* external `style_map.json`
* OpenWebUI compatible
* persona voice switching

---

# Requirements

Python 3.10+

Recommended environment:

```
venv
FastAPI
uvicorn
httpx
```

---

# Quick Start

Run Style-Bert-VITS2 first.

```
python server_fastapi.py
```

Then start voice-bridge:

```
python run_server.py
```

Default endpoint:

```
http://localhost:8787
```

---

# Configuration

Environment variables:

```
GAR_BASE_URL=http://localhost:8081
SBV2_BASE_URL=http://localhost:5000
VOICE_BRIDGE_STYLE_MAP=./style_map.json
```

---

# style_map.json

Optional mapping for models that use numeric styles.

Example:

```
{
  "amitaro": {
    "Angry": "03",
    "Happy": "04",
    "Sad": "01"
  }
}
```

---

# License

This project is licensed under the **Apache License 2.0**.

---

# Third-party Software

voice-bridge communicates with **Style-Bert-VITS2** through its HTTP API.

Style-Bert-VITS2 is licensed separately under **AGPL-3.0 / LGPL-3.0**.

See the upstream repository for details:

https://github.com/litagin02/Style-Bert-VITS2

---

# Notes

* SBV2 model files are **not included** in this repository.
* Install and run Style-Bert-VITS2 separately.

---

# Related Projects

gar-llm
https://github.com/Smashir/gar-llm
