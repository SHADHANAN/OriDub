"""
VoiceFusion AI – FFmpeg Detection & Resolution Utility
======================================================
Robust cross-platform (specifically Windows-optimised) locator for FFmpeg.
Never uses hard-coded user/machine paths.

Resolution Order:
1. FFMPEG_PATH environment variable (explicit user override).
2. System PATH (via shutil.which).
3. Project-local directories (.venv/Scripts, bin/, backend root).
4. imageio-ffmpeg bundled binary (if package is installed).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from .config import PipelineError


def get_ffmpeg_path() -> str:
    """Locate and return the absolute path to a functional FFmpeg executable.

    Returns
    -------
    str:
        Resolved absolute path to ffmpeg.exe (or ffmpeg binary).

    Raises
    ------
    PipelineError:
        If FFmpeg cannot be located anywhere.
    """
    candidate_path: Optional[str] = None

    # 1. Environment variable FFMPEG_PATH (supports file path or directory)
    env_ffmpeg = os.environ.get("FFMPEG_PATH", "").strip().strip('"').strip("'")
    if env_ffmpeg:
        p = Path(env_ffmpeg).resolve()
        if p.is_file():
            candidate_path = str(p)
        elif p.is_dir():
            for name in ("ffmpeg.exe", "ffmpeg"):
                c = p / name
                if c.is_file():
                    candidate_path = str(c)
                    break

    # 2. System PATH (standard lookup)
    if not candidate_path:
        path_ffmpeg = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        if path_ffmpeg:
            candidate_path = str(Path(path_ffmpeg).resolve())

    # 3. Project-local directories relative to backend
    if not candidate_path:
        backend_dir = Path(__file__).resolve().parents[1]
        local_candidates = [
            backend_dir / "ffmpeg.exe",
            backend_dir / "bin" / "ffmpeg.exe",
            backend_dir / ".venv" / "Scripts" / "ffmpeg.exe",
            Path(sys.prefix) / "Scripts" / "ffmpeg.exe",
            Path(sys.prefix) / "bin" / "ffmpeg",
        ]
        for candidate in local_candidates:
            if candidate.is_file():
                candidate_path = str(candidate.resolve())
                break

    # 4. imageio-ffmpeg fallback
    if not candidate_path:
        try:
            import imageio_ffmpeg
            imageio_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if imageio_exe and Path(imageio_exe).is_file():
                candidate_path = str(Path(imageio_exe).resolve())
        except Exception:
            pass

    if candidate_path:
        # Ensure parent directory is in PATH so subprocess calls by other tools (whisper, etc.) find it
        p_dir = str(Path(candidate_path).parent)
        if p_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p_dir + os.pathsep + os.environ.get("PATH", "")
        return candidate_path

    # If we reached here, FFmpeg is completely missing
    raise PipelineError("FFmpeg is not installed or is not available in PATH.")

