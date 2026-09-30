"""
VoiceFusion AI – PipelineOrchestrator
=======================================
The central coordinator that drives every step of the dubbing pipeline:

  1. Extract audio from video (AudioExtractor)
  2. Denoise the extracted audio (AudioExtractor)
  3. Transcribe Tamil audio → Tamil text (Transcriber)
  4. (Optionally) normalise colloquial Tamil (Transcriber)
  5. Translate Tamil text → Hindi text (Translator)
  6. Clone / synthesise Hindi dubbed audio (VoiceCloner)
  7. Merge dubbed audio back into the video (VideoMerger)

Each step is wrapped in error handling; if any step raises a
``PipelineError`` the orchestrator stops immediately and records the failure.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .audio_extractor import AudioExtractor
from .config import PipelineConfig, PipelineError
from .transcriber import Transcriber
from .translator import Translator
from .video_merger import VideoMerger
from .voice_cloner import VoiceCloner


class PipelineOrchestrator:
    """Runs the full Tamil-dubbing pipeline end-to-end.

    Parameters
    ----------
    config:
        Fully-populated ``PipelineConfig`` for this job.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self.last_error: Optional[PipelineError] = None

        # Instantiate collaborators
        self._extractor = AudioExtractor(config)
        self._transcriber = Transcriber(config)
        self._translator = Translator(config)
        self._cloner = VoiceCloner(config)
        self._merger = VideoMerger(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """Execute every pipeline step in order.

        Returns
        -------
        bool
            ``True`` when the full pipeline completes successfully;
            ``False`` when any step fails (see ``self.last_error``).
        """
        self._prepare_directories()
        steps = [
            ("extract_audio", self._step_extract_audio),
            ("transcribe", self._step_transcribe),
            ("translate", self._step_translate),
            ("voice_clone", self._step_voice_clone),
            ("merge_video", self._step_merge_video),
        ]
        for name, handler in steps:
            print(f"\n[Orchestrator] ── {name} ──")
            try:
                handler()
            except PipelineError as exc:
                self.last_error = exc
                print(f"[Orchestrator] ✗ {name}: {exc}")
                return False
            except Exception as exc:
                self.last_error = PipelineError(
                    f"Unexpected error in {name}: {exc}"
                )
                print(f"[Orchestrator] ✗ {name} (unexpected): {exc}")
                return False
            print(f"[Orchestrator] ✓ {name} complete.")

        print("\n[Orchestrator] ✅ Pipeline completed successfully.")
        return True

    # ------------------------------------------------------------------
    # Step implementations
    # ------------------------------------------------------------------

    def _prepare_directories(self) -> None:
        self._config.workdir.mkdir(parents=True, exist_ok=True)
        self._config.output_dir.mkdir(parents=True, exist_ok=True)

    def _step_extract_audio(self) -> None:
        self._extractor.extract()
        self._extractor.denoise(
            self._config.extracted_audio,
            self._config.denoised_audio,
        )

    def _step_transcribe(self) -> None:
        audio = self._config.denoised_audio
        if not audio.exists():
            # Fallback to raw extract if denoising was skipped / failed
            audio = self._config.extracted_audio
        text = self._transcriber.transcribe(audio)
        # Optionally normalise colloquial Tamil before translation
        if not self._config.preserve_slang:
            text = self._transcriber.normalise(text)
            self._config.transcript_file.write_text(text, encoding="utf-8")

    def _step_translate(self) -> None:
        transcript = self._config.transcript_file
        if not transcript or not transcript.exists():
            raise PipelineError("No transcript available for translation.")
        text = transcript.read_text(encoding="utf-8")
        self._translator.translate(text)

    def _step_voice_clone(self) -> None:
        translated = self._config.translated_file
        if not translated or not translated.exists():
            raise PipelineError("No translated text available for voice cloning.")
        speaker = self._config.speaker_wav or self._config.denoised_audio
        self._cloner.clone(
            translated_text_path=translated,
            original_audio_path=self._config.denoised_audio,
            speaker_wav=speaker,
        )

    def _step_merge_video(self) -> None:
        self._merger.merge(
            video_path=self._config.input_video,
            audio_path=self._config.dubbed_audio,
        )
