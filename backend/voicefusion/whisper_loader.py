"""
VoiceFusion AI – Whisper Model Loader & Singleton Manager
=========================================================
Ensures Whisper model is loaded ONCE, verified via SHA256 checksum,
and safely reused across language detection and transcription.
Optimised for CPU-only execution (Intel i5-1334U + 16 GB RAM).
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Optional

from .config import PipelineError


class WhisperManager:
    """Thread-safe singleton manager for Whisper ASR model."""

    _instance: Optional[WhisperManager] = None
    _init_lock = threading.Lock()

    def __init__(self) -> None:
        self._model = None
        self._device: str = "cpu"
        self._load_lock = threading.Lock()
        self.inference_lock = threading.Lock()  # Serialize CPU Whisper inference

    @classmethod
    def get_instance(cls) -> WhisperManager:
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_device(self) -> str:
        return self._device

    def get_model(self):
        """Return the pre-loaded Whisper model (loads once on first call)."""
        if self._model is not None:
            return self._model

        with self._load_lock:
            if self._model is not None:
                return self._model

            try:
                import torch
                import whisper
            except ImportError as exc:
                raise PipelineError(f"Whisper or PyTorch is not installed: {exc}") from exc

            # Step 3: Device detection - CUDA if available, otherwise CPU
            if torch.cuda.is_available():
                self._device = "cuda"
            else:
                self._device = "cpu"

            cache_dir = Path(os.path.expanduser("~")) / ".cache" / "whisper"
            model_name = "tiny"
            model_path = cache_dir / f"{model_name}.pt"

            # Check if model is registered in Whisper
            if model_name not in whisper._MODELS:
                raise PipelineError(f"Whisper model '{model_name}' not found in registry.")

            url = whisper._MODELS[model_name]
            expected_sha = url.split("/")[-2].lower()

            # Check existence and verify SHA256 checksum
            if not model_path.is_file():
                raise PipelineError(
                    f"Whisper '{model_name}' model file not found at {model_path}.\n"
                    "Please verify that tiny.pt exists in ~/.cache/whisper/."
                )

            # Strict SHA256 verification (Do NOT bypass checksum)
            file_hash = hashlib.sha256()
            with model_path.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    file_hash.update(chunk)
            actual_sha = file_hash.hexdigest().lower()

            if actual_sha != expected_sha:
                raise PipelineError(
                    f"Whisper model '{model_name}' failed SHA256 checksum verification.\n"
                    f"Expected: {expected_sha}\n"
                    f"Actual:   {actual_sha}\n"
                    "The model file appears to be corrupted. Do not use corrupted models."
                )

            print(f"[Whisper] Loading {model_name} model...")
            try:
                # Load once, specifying CPU device and cache directory
                loaded = whisper.load_model(
                    model_name,
                    device=self._device,
                    download_root=str(cache_dir),
                )
                self._model = loaded
                print("[Whisper] Model loaded successfully")
                print(f"[Whisper] Device: {self._device}")
            except Exception as exc:
                raise PipelineError(f"Failed to load Whisper model '{model_name}': {exc}") from exc

            return self._model


def get_whisper_manager() -> WhisperManager:
    """Convenience accessor for the WhisperManager singleton."""
    return WhisperManager.get_instance()
