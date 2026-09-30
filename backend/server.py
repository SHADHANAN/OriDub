"""
VoiceFusion AI – FastAPI Server  v3.0
======================================
Adds auto language-detection and multi-language dubbing support.

Endpoints
---------
POST /api/detect
    Upload a video (multipart).  Runs Whisper language-detection on the
    first 30 s of audio.  Returns a JSON payload with detected language,
    confidence, top-5 candidates, and the full list of supported target
    languages so the frontend can render a picker.

    Response:
    {
      "detected":        "ta",
      "label":           "Tamil",
      "confidence":      0.9873,
      "confidence_pct":  "98.7%",
      "top5":            [{"code":"ta","label":"Tamil","confidence":0.9873,"flag":"🇮🇳","supported":true}, …],
      "supported":       true,
      "target_languages":[{"code":"hi","label":"Hindi","nllb":"hin_Deva","tts":"hi","flag":"🇮🇳"}, …]
    }

POST /api/process
    Accepts multipart/form-data:
    - video            – video file (required)
    - speaker_wav      – speaker reference WAV (optional)
    - source_whisper   – Whisper ISO code of source language  (e.g. "ta")
    - source_nllb      – NLLB FLORES-200 code of source lang  (e.g. "tam_Taml")
    - target_nllb      – NLLB FLORES-200 code of target lang  (e.g. "hin_Deva")
    - target_tts       – BCP-47 TTS code of target lang       (e.g. "hi")
    - whisper_model    – Whisper model variant (default: large-v3)
    - nllb_size        – NLLB size suffix (default: distilled-600M)
    - preserve_slang   – "true"/"false" (default: true)
    - skip_lipsync     – "true"/"false" (default: true)

    Returns JSON { "job_id": "<uuid>", "status": "queued" }

GET /api/status/{job_id}
    Returns JSON { "status": "processing|done|error", "message": "…" }

GET /api/download/{job_id}
    Streams the dubbed WAV when status == "done".

GET /api/languages
    Returns the full list of supported source/target languages.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# GPU selection – MUST be the very first thing before any torch/CUDA import.
# On Windows systems with both AMD Radeon and NVIDIA GPUs, ROCm/DirectML can
# hijack cuda:0 before PyTorch initialises. We suppress AMD here and let
# select_torch_device() confirm the correct NVIDIA device at runtime.
# ---------------------------------------------------------------------------
import os as _os
import sys as _sys
from pathlib import Path as _Path

if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")   # Disable AMD ROCm/HIP

# Ensure local project binaries and virtualenv scripts are in PATH
_backend_dir = _Path(__file__).resolve().parent
for _p in [
    str(_backend_dir),
    str(_backend_dir / "bin"),
    str(_backend_dir / ".venv" / "Scripts"),
    str(_Path(_sys.prefix) / "Scripts"),
]:
    if _os.path.exists(_p) and _p not in _os.environ.get("PATH", ""):
        _os.environ["PATH"] = _p + _os.pathsep + _os.environ.get("PATH", "")

import asyncio
import shutil
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

try:
    import aiofiles
    _HAS_AIOFILES = True
except ImportError:
    _HAS_AIOFILES = False

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from voicefusion.config import DEFAULT_DETECTOR_MODEL, DEFAULT_WHISPER_MODEL, PipelineConfig
from voicefusion.language_detector import LanguageDetector
from voicefusion.languages import all_language_options
from voicefusion.pipeline_orchestrator import PipelineOrchestrator

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
FRONTEND_DIR = BASE_DIR.parent / "frontend"

JOBS_DIR.mkdir(exist_ok=True)

# In-memory job store
_JOBS: dict[str, dict[str, Any]] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=1)

# Singleton language detector (model loaded once, reused across requests)
_detector = LanguageDetector(whisper_model=DEFAULT_DETECTOR_MODEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load Whisper model once at server startup."""
    from starlette.concurrency import run_in_threadpool
    from voicefusion.whisper_loader import get_whisper_manager
    await run_in_threadpool(get_whisper_manager().get_model)
    yield


app = FastAPI(title="VoiceFusion AI API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Async file-save helper – avoids disk 100% during upload
# ---------------------------------------------------------------------------
_CHUNK_SIZE = 16 * 1024 * 1024  # 16 MB chunks – large enough to amortise syscall
                                  # overhead, small enough to let the OS breathe.

async def _save_upload(upload: "UploadFile", dest: Path) -> None:
    """Write *upload* to *dest* using async chunked I/O.

    Uses ``aiofiles`` when available for true non-blocking writes; falls back
    to chunked synchronous writes in a thread so the event loop is never
    blocked for more than one chunk at a time.
    """
    if _HAS_AIOFILES:
        import aiofiles
        async with aiofiles.open(dest, "wb") as fh:
            while True:
                chunk = await upload.read(_CHUNK_SIZE)
                if not chunk:
                    break
                await fh.write(chunk)
    else:
        # Fallback: chunked sync writes offloaded to the thread pool
        loop = asyncio.get_event_loop()
        def _write_sync():
            with dest.open("wb") as fh:
                while True:
                    chunk = upload.file.read(_CHUNK_SIZE)
                    if not chunk:
                        break
                    fh.write(chunk)
        await loop.run_in_executor(None, _write_sync)


# ---------------------------------------------------------------------------
# Background pipeline worker
# ---------------------------------------------------------------------------
def _run_pipeline(
    job_id: str,
    video_path: Path,
    speaker_path: Path | None,
    source_whisper: str,
    source_nllb: str,
    target_nllb: str,
    target_tts: str,
    whisper_model: str,
    nllb_size: str,
    preserve_slang: bool,
    skip_lipsync: bool,
) -> None:
    """Blocking pipeline execution – runs inside a thread pool."""
    job_dir = JOBS_DIR / job_id
    workdir  = job_dir / "work"
    out_dir  = job_dir / "output"

    _JOBS[job_id]["status"]  = "processing"
    _JOBS[job_id]["message"] = "Pipeline started…"

    try:
        config = PipelineConfig(
            input_video=video_path,
            workdir=workdir,
            output_dir=out_dir,
            speaker_wav=speaker_path,
            whisper_model="tiny",
            nllb_model_size=nllb_size,
            source_lang=source_nllb,
            target_lang=target_nllb,
            tts_language=target_tts,
            source_whisper_code=source_whisper,
            preserve_slang=preserve_slang,
            skip_lipsync=skip_lipsync,
        )
        orchestrator = PipelineOrchestrator(config)
        success = orchestrator.run()

        if success:
            _JOBS[job_id]["status"]  = "done"
            _JOBS[job_id]["message"] = "Pipeline completed successfully."
            _JOBS[job_id]["output"]  = config.dubbed_audio
        else:
            _JOBS[job_id]["status"]  = "error"
            _JOBS[job_id]["message"] = str(orchestrator.last_error)
    except Exception as exc:
        _JOBS[job_id]["status"]  = "error"
        _JOBS[job_id]["message"] = f"Internal error: {exc}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/languages")
async def get_languages() -> JSONResponse:
    """Return the full list of supported source/target languages."""
    return JSONResponse(all_language_options())


@app.post("/api/detect")
async def detect_language(
    video: UploadFile = File(...),
) -> JSONResponse:
    """Auto-detect the spoken language from the first 30 s of *video*.

    The client should call this immediately after a file is selected
    (before the user hits 'Start Dubbing').
    """
    from starlette.concurrency import run_in_threadpool

    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    # Save upload to a temp location
    tmp_dir  = JOBS_DIR / f"detect_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True)
    suffix   = Path(video.filename).suffix or ".mp4"
    tmp_path = tmp_dir / f"input{suffix}"

    try:
        await _save_upload(video, tmp_path)
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")

        result = await run_in_threadpool(_detector.detect_from_video, tmp_path)
        result["target_languages"] = all_language_options()
        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()          # ← full traceback in server terminal
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/process")
async def process_video(
    video: UploadFile = File(...),
    speaker_wav: UploadFile | None = File(default=None),
    source_whisper: str = Form(default="ta"),
    source_nllb: str    = Form(default="tam_Taml"),
    target_nllb: str    = Form(default="hin_Deva"),
    target_tts: str     = Form(default="hi"),
    whisper_model: str  = Form(default="tiny"),
    nllb_size: str      = Form(default="distilled-600M"),
    preserve_slang: str = Form(default="true"),
    skip_lipsync: str   = Form(default="true"),
) -> JSONResponse:
    """Accept a video upload and start the dubbing pipeline asynchronously."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    job_id  = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    # Save video – use async chunked I/O to avoid disk 100% saturation
    suffix     = Path(video.filename).suffix or ".mp4"
    video_path = job_dir / f"input{suffix}"
    await _save_upload(video, video_path)

    if not video_path.exists() or video_path.stat().st_size == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded video file is empty (0 bytes).")

    # Save optional speaker reference
    speaker_path: Path | None = None
    if speaker_wav and speaker_wav.filename:
        speaker_path = job_dir / "speaker.wav"
        await _save_upload(speaker_wav, speaker_path)

    # Step 2: Enforce Whisper 'tiny' model
    enforced_whisper_model = "tiny"

    _JOBS[job_id] = {"status": "queued", "message": "Job queued.", "output": None}

    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        _EXECUTOR,
        _run_pipeline,
        job_id,
        video_path,
        speaker_path,
        source_whisper,
        source_nllb,
        target_nllb,
        target_tts,
        enforced_whisper_model,
        nllb_size,
        preserve_slang.lower() == "true",
        skip_lipsync.lower() == "true",
    )

    return JSONResponse({"job_id": job_id, "status": "queued"})


@app.get("/api/status/{job_id}")
async def get_status(job_id: str) -> JSONResponse:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse({"status": job["status"], "message": job["message"]})


@app.get("/api/download/{job_id}")
async def download_audio(job_id: str) -> FileResponse:
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job not ready: {job['status']}")
    out: Path | None = job.get("output")
    if not out or not out.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")
    return FileResponse(path=str(out), media_type="audio/wav", filename="dubbed_audio.wav")


# ---------------------------------------------------------------------------
# Health check (useful for debugging)
# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "version": "3.0.0"})


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
# IMPORTANT: StaticFiles MUST be mounted at a sub-path (not "/") so that
# FastAPI's own route table is checked first for every /api/* request.
# Mounting at "/" causes StaticFiles to shadow all API routes → 405 errors.
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/")
async def root_redirect():
    """Redirect bare root to the frontend SPA."""
    return RedirectResponse(url="/app/index.html")


def _is_port_in_use(port: int, host: str = "0.0.0.0") -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def _free_stale_port(port: int = 8000) -> bool:
    """If port is occupied, automatically terminate the stale owning process."""
    import os
    import subprocess
    import time

    current_pid = os.getpid()
    killed = False
    try:
        cmd = "netstat -ano -p tcp"
        output = subprocess.check_output(cmd, shell=True, text=True, errors="replace")
        for line in output.splitlines():
            line = line.strip()
            if f":{port}" in line and "LISTENING" in line.upper():
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        pid = int(parts[-1])
                        if pid > 0 and pid != current_pid:
                            print(f"[VoiceFusion] Port {port} occupied by PID {pid}. Terminating stale process...")
                            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
                            killed = True
                    except ValueError:
                        pass
        if killed:
            time.sleep(1.0)
    except Exception as exc:
        print(f"[VoiceFusion] Could not auto-terminate process on port {port}: {exc}")
    return not _is_port_in_use(port)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from voicefusion.ffmpeg_utils import get_ffmpeg_path

    try:
        ffmpeg_bin = get_ffmpeg_path()
        print(f"[VoiceFusion] [OK] Found FFmpeg: {ffmpeg_bin}")
    except Exception as exc:
        print(f"[VoiceFusion] [!] Warning: {exc}")

    if _is_port_in_use(8000):
        print("[VoiceFusion] Port 8000 is currently occupied. Attempting to free it...")
        if not _free_stale_port(8000):
            print(
                "\n" + "=" * 70 + "\n"
                "[ERROR] Port 8000 is already in use and could not be freed automatically!\n"
                "Please terminate the process holding port 8000 using:\n"
                "  PowerShell: Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }\n"
                "  CMD:        for /f \"tokens=5\" %a in ('netstat -aon ^| findstr :8000') do taskkill /f /pid %a\n"
                + "=" * 70 + "\n"
            )
            import sys
            sys.exit(1)
        print("[VoiceFusion] [OK] Port 8000 freed successfully.")

    _detector._ensure_model_loaded()
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
