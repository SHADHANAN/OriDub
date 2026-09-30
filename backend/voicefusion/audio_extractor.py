"""
VoiceFusion AI – AudioExtractor
================================
Responsible for pulling raw audio from a video file and optionally applying
FFmpeg-based noise reduction.  All logic is encapsulated in ``AudioExtractor``
so the orchestrator only calls ``extract()`` and ``denoise()``.

GPU acceleration
----------------
When an NVIDIA GPU is available, FFmpeg is invoked with::

    -hwaccel cuda -hwaccel_output_format cuda

This offloads video *decode* to the GPU (NVDEC / CUVID), which removes the
video-decode workload from the CPU entirely.  Audio extraction is a pure
demux-and-decode operation so it benefits most from reduced CPU pressure.
Denoising is purely an audio DSP filter that runs inside FFmpeg's CPU
filter-graph regardless of ``-hwaccel``; the noise filter itself is very
light and does NOT cause disk saturation.
"""
from __future__ import annotations

import subprocess
import shlex
from pathlib import Path

from .config import PipelineConfig, PipelineError
from .ffmpeg_utils import get_ffmpeg_path


def _ffmpeg_hwaccel_flags() -> list[str]:
    """Return FFmpeg input flags to enable NVIDIA GPU hardware-accelerated decode.

    Probes whether CUDA/NVDEC is available via a quick ``ffmpeg -hwaccels``
    check.  Falls back to an empty list (CPU-only) if NVIDIA is not found or
    ffmpeg was not compiled with NVDEC support.
    """
    try:
        ffmpeg_bin = get_ffmpeg_path()
        result = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        output = (result.stdout + result.stderr).lower()
        if "cuda" in output or "cuvid" in output:
            print("[AudioExtractor] GPU decode available – using -hwaccel cuda")
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    except Exception as exc:
        print(f"[AudioExtractor] hwaccel probe failed ({exc}), using CPU.")
    return []


class AudioExtractor:
    """Extracts audio from a video and optionally denoises it using FFmpeg.

    Parameters
    ----------
    config:
        Shared pipeline configuration object.
    rnnoise_model:
        Optional path to a pre-trained RNNoise ``.nn`` model file.
        When ``None`` the ``afftdn`` FFT-based denoiser is used instead.
    denoise_strength:
        Noise-reduction strength for ``afftdn`` (ignored when an RNNoise
        model is supplied).  Typical range: 5–20.
    """

    def __init__(
        self,
        config: PipelineConfig,
        rnnoise_model: str | None = None,
        denoise_strength: int = 10,
    ) -> None:
        self._config = config
        self._rnnoise_model = rnnoise_model
        self._denoise_strength = denoise_strength

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> Path:
        """Extract mono 22 050 Hz PCM audio from the input video.

        NVIDIA GPU hardware-accelerated decode (NVDEC/CUVID) is used
        automatically when available, which offloads video demuxing from
        the CPU and significantly reduces disk I/O pressure.

        Returns
        -------
        Path
            Path to the extracted ``.wav`` file.

        Raises
        ------
        PipelineError
            If the input video does not exist or FFmpeg fails.
        """
        if not self._config.input_video.exists():
            raise PipelineError(
                f"Input video not found: {self._config.input_video}"
            )
        if self._config.input_video.stat().st_size == 0:
            raise PipelineError("Input video file is empty (0 bytes).")

        output = self._config.extracted_audio
        hwaccel = _ffmpeg_hwaccel_flags()
        ffmpeg_bin = get_ffmpeg_path()
        cmd = [
            ffmpeg_bin, "-y",
            # GPU-accelerated decode when NVDEC is available
            *hwaccel,
            "-thread_queue_size", "512",  # cap read-ahead queue → lower disk pressure
            "-i", str(self._config.input_video.resolve()),
            "-vn",            # discard video stream
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",   # 16 kHz mono optimized for Whisper
            "-threads", "0",  # let FFmpeg pick optimal CPU thread count
            str(output.resolve()),
        ]
        self._run(cmd, "audio extraction")

        if not output.exists():
            raise PipelineError("Audio extraction produced no output file.")
        print(f"[AudioExtractor] Extracted → {output}")
        return output

    def denoise(self, input_audio: Path, output_audio: Path) -> Path:
        """Apply FFmpeg noise-reduction to *input_audio*.

        Parameters
        ----------
        input_audio:
            Path to the source WAV file.
        output_audio:
            Destination path for the denoised WAV file.

        Returns
        -------
        Path
            Path to the denoised output file.
        """
        if not input_audio.exists():
            raise PipelineError(f"Audio file not found for denoising: {input_audio}")

        if self._rnnoise_model and Path(self._rnnoise_model).exists():
            af_filter = f"arnndn=m='{self._rnnoise_model}':mix=0.9"
        else:
            af_filter = f"afftdn=nr={self._denoise_strength}:nt=w:om=o"

        ffmpeg_bin = get_ffmpeg_path()
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(input_audio.resolve()),
            "-ac", "1",
            "-ar", "16000",
            "-af", af_filter,
            str(output_audio.resolve()),
        ]
        self._run(cmd, "noise reduction")

        if not output_audio.exists():
            raise PipelineError("Noise reduction produced no output file.")
        print(f"[AudioExtractor] Denoised → {output_audio}")
        return output_audio

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(cmd: list[str], step: str) -> None:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise PipelineError("FFmpeg is not installed or is not available in PATH.")
        if result.returncode != 0:
            err = result.stderr.strip() or result.stdout.strip() or "no output"
            raise PipelineError(f"{step} failed (exit {result.returncode}): {err}")
