"""
VoiceFusion AI – Centralised Configuration
==========================================
All runtime paths, model names, and tunable parameters live here so that
each processing class stays dependency-free from each other's path choices.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Pipeline-level exception
# ---------------------------------------------------------------------------
class PipelineError(Exception):
    """Raised when any VoiceFusion pipeline step fails unrecoverably."""


def select_torch_device(require_nvidia: bool = False) -> str:
    """Return the preferred Torch device string for this machine.

    Actively searches for an NVIDIA GPU among all visible CUDA devices and
    forces it to be used. On Windows systems with both AMD Radeon and NVIDIA
    GPUs, this prevents AMD from being selected by default.

    Strategy
    --------
    1. Set ``HIP_VISIBLE_DEVICES=-1`` so ROCm/HIP (AMD) is suppressed.
    2. Enumerate all CUDA devices and filter for NVIDIA by device name.
    3. If an NVIDIA device is found, pin it via ``CUDA_VISIBLE_DEVICES`` and
       ``torch.cuda.set_device`` so every subsequent allocation lands there.
    4. Raise ``PipelineError`` if ``require_nvidia=True`` and no NVIDIA device
       is visible.
    """
    import os

    try:
        import torch
    except ImportError as exc:
        raise PipelineError("torch is not installed.") from exc

    # Suppress AMD ROCm/HIP so it cannot hijack the CUDA context on Windows.
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "-1")

    if not torch.cuda.is_available():
        print("[GPU] CUDA not available – falling back to CPU.")
        return "cpu"

    cuda_devices: list[tuple[int, str]] = []
    nvidia_devices: list[tuple[int, str]] = []
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        cuda_devices.append((index, name))
        if "nvidia" in name.lower() or "geforce" in name.lower() or "gtx" in name.lower() or "rtx" in name.lower() or "quadro" in name.lower():
            nvidia_devices.append((index, name))

    if nvidia_devices:
        index, name = nvidia_devices[0]
        # Pin the process to only this GPU so AMD cannot appear as cuda:0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(index)
        torch.cuda.set_device(0)  # After remapping, it becomes device 0
        print(f"[GPU] ✓ Pinned to NVIDIA GPU (physical index {index}): {name}")
        return "cuda:0"

    # No NVIDIA device found — list what was found for diagnostics
    all_names = ", ".join(f"{i}:{n}" for i, n in cuda_devices) or "none"
    print(f"[GPU] Warning: No NVIDIA GPU found. Visible CUDA devices: {all_names}")

    if require_nvidia:
        raise PipelineError(
            "No NVIDIA GPU detected. VoiceFusion requires an NVIDIA CUDA-capable GPU.\n"
            f"Visible CUDA device(s): {all_names}\n"
            "Ensure NVIDIA drivers and CUDA toolkit are installed, then retry."
        )

    if cuda_devices:
        index, name = cuda_devices[0]
        print(f"[GPU] Falling back to CUDA device {index}: {name}")
        torch.cuda.set_device(index)
        return f"cuda:{index}"

    return "cpu"


# ---------------------------------------------------------------------------
# Immutable defaults
# ---------------------------------------------------------------------------
# Fast & lightweight default for language detection (runs reliably on CPU & GPU)
DEFAULT_DETECTOR_MODEL: str = "tiny"

# Model VRAM budgets:
#   Whisper tiny   → ~75 MB  ← safe, instant default for CPU & low-VRAM GPUs
#   Whisper small  → ~2 GB
#   NLLB 600M      → ~1.5 GB (loaded after Whisper is unloaded)
#   XTTS-v2        → ~2 GB   (loaded after NLLB is unloaded)
DEFAULT_WHISPER_MODEL: str = "tiny"
DEFAULT_NLLB_MODEL_SIZE: str = "distilled-600M"   # options: distilled-600M | 1.3B | 3.3B

# NLLB FLORES-200 codes — overridden at runtime by LanguageDetector + user selection
DEFAULT_SOURCE_LANG: str = "tam_Taml"   # Tamil (fallback)
DEFAULT_TARGET_LANG: str = "hin_Deva"   # Hindi (fallback)
DEFAULT_TTS_LANGUAGE: str = "hi"        # BCP-47 for XTTS

WAV2LIP_DIR: str = r"D:\VoiceFusion\Wav2Lip"
VOSK_MODEL_PATH: str = r"C:\Python312\vosk-model-small-en-us-0.15"


# ---------------------------------------------------------------------------
# PipelineConfig dataclass – created once per request and passed everywhere
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Holds all paths and options for a single pipeline run.

    Attributes
    ----------
    input_video:
        Absolute path to the uploaded input video.
    workdir:
        Directory for intermediate artefacts (audio, transcripts, etc.).
    output_dir:
        Directory for the final dubbed audio / video output.
    speaker_wav:
        Optional path to a separate speaker-reference WAV. Falls back to
        the extracted audio when ``None``.
    whisper_model:
        Whisper model variant to use for ASR.
    nllb_model_size:
        NLLB-200 model size suffix (e.g. ``"distilled-600M"``).
    source_lang / target_lang:
        FLORES-200 language codes for NLLB translation.
    tts_language:
        BCP-47 language code passed to XTTS for synthesis.
    preserve_slang:
        When ``True`` the Tamil colloquial tone is preserved in translation.
    skip_lipsync:
        When ``True`` the Wav2Lip lipsync step is skipped.
    """

    input_video: Path
    workdir: Path
    output_dir: Path

    speaker_wav: Optional[Path] = None
    whisper_model: str = DEFAULT_WHISPER_MODEL
    nllb_model_size: str = DEFAULT_NLLB_MODEL_SIZE
    source_lang: str = DEFAULT_SOURCE_LANG    # NLLB FLORES-200 code
    target_lang: str = DEFAULT_TARGET_LANG    # NLLB FLORES-200 code
    tts_language: str = DEFAULT_TTS_LANGUAGE  # BCP-47 for XTTS
    source_whisper_code: str = "ta"           # ISO 639-1 code used by Whisper ASR
    preserve_slang: bool = True
    skip_lipsync: bool = True

    # -- Derived paths (set in __post_init__) --
    extracted_audio: Optional[Path] = field(default=None, init=False)
    denoised_audio: Optional[Path] = field(default=None, init=False)
    transcript_file: Optional[Path] = field(default=None, init=False)
    translated_file: Optional[Path] = field(default=None, init=False)
    dubbed_audio: Optional[Path] = field(default=None, init=False)
    final_video: Optional[Path] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.input_video = Path(self.input_video).resolve()
        self.workdir = Path(self.workdir).resolve()
        self.output_dir = Path(self.output_dir).resolve()
        if self.speaker_wav is not None:
            self.speaker_wav = Path(self.speaker_wav).resolve()

        stem = self.input_video.stem
        self.extracted_audio = self.workdir / f"{stem}_raw.wav"
        self.denoised_audio = self.workdir / f"{stem}_denoised.wav"
        self.transcript_file = self.workdir / f"{stem}_transcript.txt"
        self.translated_file = self.workdir / f"{stem}_translated.txt"
        self.dubbed_audio = self.output_dir / "final_output_combined.wav"
        self.final_video = self.output_dir / f"{stem}_dubbed.mp4"
