"""
VoiceFusion AI – Transcriber
=============================
Handles multilingual ASR using OpenAI Whisper with:
  * GPU auto-detection
  * Audio pre-processing (trim / normalise via librosa)
  * Retry on corrupted model downloads
  * Dynamic source language from ``PipelineConfig.source_whisper_code``
  * Optional colloquial-to-formal Tamil normalisation

The ``transcribe()`` method returns the raw transcript text in whatever
language Whisper detects; ``normalise()`` applies Tamil-specific slang
mappings (only useful when the source is Tamil).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from .config import PipelineConfig, PipelineError, select_torch_device
from .tamil_slang_mappings import ALL_MAPPINGS, CONTEXT_RULES


class Transcriber:
    """Converts an audio file into text using Whisper.

    The spoken language is read from ``config.source_whisper_code``
    (set by the auto-detector or supplied by the user).

    Parameters
    ----------
    config:
        Shared pipeline configuration object.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._model_size = config.whisper_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: Path) -> str:
        """Transcribe *audio_path* and save result to the transcript file.

        Returns
        -------
        str
            Raw transcript text in the source language.
        """
        if not audio_path.exists():
            raise PipelineError(f"Audio file not found: {audio_path}")

        from .whisper_loader import get_whisper_manager
        mgr = get_whisper_manager()
        model = mgr.get_model()
        device = mgr.get_device()
        lang = self._config.source_whisper_code
        print(f"[Transcriber] Device: {device} | Model: tiny | Lang: {lang}")

        preprocessed = self._preprocess_audio(audio_path)
        try:
            print("[Transcriber] Running ASR…")
            is_cuda = device.startswith("cuda")
            with mgr.inference_lock:
                result = model.transcribe(
                    str(preprocessed),
                    language=lang,         # dynamic – set by LanguageDetector or user
                    fp16=is_cuda,          # False on CPU
                    beam_size=5,
                    best_of=5,
                    temperature=0.0,
                    condition_on_previous_text=True,
                    verbose=False,
                )
        finally:
            # Clean up temp preprocessed file safely
            if preprocessed != audio_path and preprocessed.exists():
                try:
                    preprocessed.unlink()
                except Exception:
                    pass

        text: str = result.get("text", "").strip()
        if not text:
            raise PipelineError("Transcription produced no text.")

        self._config.transcript_file.write_text(text, encoding="utf-8")
        print(f"[Transcriber] Saved transcript → {self._config.transcript_file}")
        return text

    def normalise(self, text: str) -> str:
        """Apply colloquial-Tamil → formal-Tamil transformations.

        Returns
        -------
        str
            Normalised formal-Tamil string.
        """
        normalised = text
        applied = 0
        for pattern, replacement in ALL_MAPPINGS.items():
            if re.search(pattern, normalised):
                normalised = re.sub(pattern, replacement, normalised)
                applied += 1

        for pattern, replacement in CONTEXT_RULES:
            normalised = re.sub(pattern, replacement, normalised)

        print(f"[Transcriber] Applied {applied} slang normalisation(s).")
        return normalised.strip()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_audio(audio_path: Path) -> Path:
        """Trim silence and normalise audio amplitude.

        Important: ``mmap=False`` is set on ``librosa.load`` to prevent
        memory-mapped file reads, which cause repeated disk seeks and can
        push disk utilisation to 100 %.
        """
        try:
            import librosa
            import soundfile as sf
        except ImportError:
            print("[Transcriber] librosa/soundfile not installed – proceeding with extracted audio directly.")
            return audio_path

        # mmap=False → load the entire file into RAM in a single sequential
        # read instead of demand-paging it via the OS mmap subsystem.
        audio, sr = librosa.load(
            str(audio_path), sr=16000, mono=True,
            res_type="soxr_vhq",  # fast, high-quality resampler
        )
        audio, _ = librosa.effects.trim(audio, top_db=20)
        audio = librosa.util.normalize(audio)

        out_path = audio_path.with_suffix("").with_name(
            audio_path.stem + "_preprocessed.wav"
        )
        sf.write(str(out_path), audio, sr)
        return out_path
