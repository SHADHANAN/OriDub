# VoiceFusion AI

> Tamil video → Hindi dubbed audio pipeline with a minimal web frontend.

## Project layout

```
voicefusion_ai/
├── backend/
│   ├── server.py                    # 🚀 FastAPI entry-point (run this)
│   ├── requirements.txt
│   └── voicefusion/                 # Core pipeline package
│       ├── __init__.py
│       ├── config.py                # PipelineConfig + PipelineError
│       ├── tamil_slang_mappings.py  # Colloquial → formal Tamil dictionary
│       ├── audio_extractor.py       # AudioExtractor  (ffmpeg)
│       ├── transcriber.py           # Transcriber     (Whisper ASR)
│       ├── translator.py            # Translator      (NLLB-200)
│       ├── voice_cloner.py          # VoiceCloner     (XTTS-v2)
│       ├── video_merger.py          # VideoMerger     (MoviePy)
│       └── pipeline_orchestrator.py # PipelineOrchestrator (glue)
└── frontend/
    ├── index.html
    ├── style.css
    └── app.js
```

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| FFmpeg | any recent (must be on PATH) |
| CUDA (optional) | 11.8 / 12.x for GPU acceleration |

## Setup

```powershell
cd d:\VoiceFusion\voicefusion_ai\backend

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

## Run the server

```powershell
# from backend/
python server.py
```

Then open **http://localhost:8000** in your browser.

## How it works

| Step | Class | Detail |
|------|-------|--------|
| 1 | `AudioExtractor` | Pulls mono 22 kHz PCM WAV from video via FFmpeg; applies FFT denoising |
| 2 | `Transcriber` | Runs OpenAI Whisper (`large-v3` by default); normalises colloquial Tamil |
| 3 | `Translator` | NLLB-200 translates Tamil → Hindi in sentence chunks |
| 4 | `VoiceCloner` | Marian models (Tamil→EN→HI) + Coqui XTTS-v2 synthesise Hindi audio |
| 5 | `VideoMerger` | MoviePy replaces original audio with dubbed WAV |

`PipelineOrchestrator` drives each step and surfaces errors immediately.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/process` | Upload video → returns `{job_id}` |
| `GET` | `/api/status/{job_id}` | Poll for status |
| `GET` | `/api/download/{job_id}` | Download dubbed WAV |

## Frontend

Minimal dark-mode SPA:
- Drag-and-drop video upload (up to 2 GB)
- Advanced options panel (model selection, slang toggle, speaker reference)
- Animated 5-step progress tracker
- In-browser audio playback + WAV download
