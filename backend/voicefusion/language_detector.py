"""
VoiceFusion AI – LanguageDetector
===================================
Uses OpenAI Whisper's built-in language-detection capability to identify the
spoken language in a short audio clip *without* running a full transcription.

Whisper runs ~30 s of audio through a log-mel spectrogram and returns a
probability distribution over 99 languages.  We surface the top-5 candidates
with their confidence scores so the frontend can display them to the user.

Design notes
------------
* This class shares the Whisper model loading helper from ``Transcriber``
  but only calls ``model.detect_language()`` – much faster than a full
  ``model.transcribe()``.
* The model is loaded *lazily* to avoid memory allocation on import.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .config import DEFAULT_DETECTOR_MODEL, PipelineConfig, PipelineError, select_torch_device
from .ffmpeg_utils import get_ffmpeg_path
from .languages import LANGUAGES, label_for


class LanguageDetector:
    """Detects the spoken language of a video/audio file using Whisper.

    Parameters
    ----------
    whisper_model:
        Whisper model variant.  ``"tiny"`` is fast and lightweight for detection
        and is used as the default even when the pipeline uses larger models.
    """

    DETECT_MODEL = DEFAULT_DETECTOR_MODEL   # "tiny" – fast detection on CPU / GPU

    def __init__(self, whisper_model: str = DETECT_MODEL) -> None:
        self._whisper_model = whisper_model
        self._model = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_from_video(self, video_path: Path) -> dict:
        """Extract a short audio clip from *video_path* and detect language."""
        if not video_path.exists():
            raise PipelineError(f"Video file not found: {video_path}")
        if video_path.stat().st_size == 0:
            raise PipelineError("Uploaded video file is empty (0 bytes).")

        tmp_audio = video_path.parent / f"_detect_clip_{video_path.stem}.wav"
        try:
            self._extract_clip(video_path, tmp_audio)
            return self.detect_from_audio(tmp_audio)
        finally:
            if tmp_audio.exists():
                try:
                    tmp_audio.unlink()
                except Exception:
                    pass

    def detect_from_audio(self, audio_path: Path) -> dict:
        """Run language detection on a WAV/MP3 audio file."""
        if not audio_path.exists():
            raise PipelineError(f"Audio file not found: {audio_path}")
        if audio_path.stat().st_size == 0:
            raise PipelineError("Audio clip is empty (0 bytes).")

        from .whisper_loader import get_whisper_manager
        mgr = get_whisper_manager()
        model = mgr.get_model()
        device = mgr.get_device()

        import torch
        import whisper

        with mgr.inference_lock:
            # Load audio (Whisper pads/trims to 30 s internally)
            audio = whisper.load_audio(str(audio_path.resolve()))
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(device)

            with torch.no_grad():
                _, probs = model.detect_language(mel)

        # Sort by probability
        sorted_langs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        detected_code, detected_prob = sorted_langs[0]

        top5 = [
            {
                "code": code,
                "label": label_for(code),
                "confidence": round(float(prob), 4),
                "flag": LANGUAGES.get(code, {}).get("flag", "🌐"),
                "supported": code in LANGUAGES,
            }
            for code, prob in sorted_langs[:5]
        ]

        return {
            "detected": detected_code,
            "label": label_for(detected_code),
            "confidence": round(float(detected_prob), 4),
            "confidence_pct": f"{detected_prob * 100:.1f}%",
            "top5": top5,
            "supported": detected_code in LANGUAGES,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model_loaded(self) -> None:
        """Pre-load Whisper model singleton."""
        from .whisper_loader import get_whisper_manager
        get_whisper_manager().get_model()

    @staticmethod
    def _extract_clip(video_path: Path, out_wav: Path, duration_s: int = 30) -> None:
        """Extract the first *duration_s* seconds of audio from a video."""
        ffmpeg_bin = get_ffmpeg_path()
        cmd = [
            ffmpeg_bin, "-y",
            "-i", str(video_path.resolve()),
            "-t", str(duration_s),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            str(out_wav.resolve()),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
        except FileNotFoundError:
            raise PipelineError("FFmpeg is not installed or is not available in PATH.")

        if result.returncode != 0:
            err_msg = result.stderr.strip() or "Audio extraction failed"
            raise PipelineError(f"FFmpeg clip extraction failed: {err_msg}")

        if not out_wav.exists() or out_wav.stat().st_size == 0:
            raise PipelineError("Failed to extract audio from video: output is empty.")

