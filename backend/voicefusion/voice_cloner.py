"""
VoiceFusion AI – VoiceCloner
==============================
Synthesises dubbed audio that preserves the timing and silence structure of
the original using Coqui XTTS-v2 for Text-to-Speech and pydub for assembly.

Key improvements over original code
------------------------------------
* All module-level side-effects (model loading, argparse, file I/O) are
  removed – everything is now inside methods called by the orchestrator.
* Translation (Tamil→English→Hindi) is done inside this class via Marian
  models that are loaded lazily once and cached on the instance.
* No global variable pollution.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, PipelineError, select_torch_device

# Accept Coqui TTS terms non-interactively
os.environ["COQUI_TOS_AGREED"] = "1"

# Compatibility patch: transformers 5+ removed isin_mps_friendly which XTTS imports
try:
    import transformers.pytorch_utils as _pt_utils
    if not hasattr(_pt_utils, "isin_mps_friendly"):
        import torch
        def _isin_mps_friendly(elements, test_elements):
            return torch.isin(elements, test_elements)
        _pt_utils.isin_mps_friendly = _isin_mps_friendly
except Exception:
    pass

# Compatibility patch: torchaudio 2.9+ on Windows delegates to torchcodec which can fail
# with [WinError 127] libtorchcodec_image.dll. Patch torchaudio.load to safely use soundfile.
try:
    import torchaudio
    import soundfile as _sf
    import torch as _torch

    _orig_ta_load = torchaudio.load

    def _safe_torchaudio_load(filepath, *args, **kwargs):
        try:
            data, sr = _sf.read(str(filepath), dtype="float32")
            if data.ndim == 1:
                tensor = _torch.from_numpy(data).unsqueeze(0)
            else:
                tensor = _torch.from_numpy(data.T)
            return tensor, sr
        except Exception:
            return _orig_ta_load(filepath, *args, **kwargs)

    torchaudio.load = _safe_torchaudio_load
except Exception:
    pass

# Configure FFmpeg and pydub for WAV operations without requiring ffprobe
try:
    from .ffmpeg_utils import get_ffmpeg_path
    _ffmpeg_exe = get_ffmpeg_path()
    from pydub import AudioSegment
    AudioSegment.converter = _ffmpeg_exe
    _orig_from_file = AudioSegment.from_file

    def _pydub_from_file(file, *args, **kwargs):
        if "codec" not in kwargs:
            is_wav = kwargs.get("format") == "wav" or (
                isinstance(file, (str, Path)) and str(file).lower().endswith(".wav")
            )
            if is_wav:
                kwargs["codec"] = "pcm_s16le"
        return _orig_from_file(file, *args, **kwargs)

    AudioSegment.from_file = _pydub_from_file
except Exception:
    pass


class VoiceCloner:
    """Generates a dubbed audio track that mirrors the original's timing.

    Parameters
    ----------
    config:
        Shared pipeline configuration object.
    """

    _MAX_TOKENS = 350  # conservative XTTS token budget per chunk

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._tts = None
        self._device: str | None = None
        # Marian translation models (lazy-loaded)
        self._tok_ta_en = None
        self._mdl_ta_en = None
        self._tok_en_hi = None
        self._mdl_en_hi = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone(
        self,
        translated_text_path: Path,
        original_audio_path: Path,
        speaker_wav: Path,
    ) -> Path:
        """Generate final dubbed audio and save to ``config.dubbed_audio``.

        Parameters
        ----------
        translated_text_path:
            Path to the translated text file (Hindi lines, one per sentence).
        original_audio_path:
            Original extracted/denoised audio used for timing reference.
        speaker_wav:
            Speaker-reference WAV for XTTS voice cloning.

        Returns
        -------
        Path
            Path to the final combined output WAV.
        """
        for p, label in [
            (translated_text_path, "translated text"),
            (original_audio_path, "original audio"),
            (speaker_wav, "speaker WAV"),
        ]:
            if not Path(p).exists():
                raise PipelineError(f"VoiceCloner: {label} not found: {p}")

        self._ensure_tts_loaded()
        # NOTE: Translation is already done by the NLLB Translator step.
        # Do NOT load or call Marian translation models here – the lines in
        # translated_text_path are already in the target language.

        # Pre-process speaker audio
        processed_speaker = self._config.output_dir / "speaker_processed.wav"
        speaker_wav = self._preprocess_speaker(speaker_wav, processed_speaker)

        # Load text lines
        lines = [
            ln.strip()
            for ln in translated_text_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        print(f"[VoiceCloner] {len(lines)} text line(s) loaded.")

        # Build segments timeline from original audio
        from pydub import AudioSegment
        orig_audio = AudioSegment.from_file(str(original_audio_path))
        timeline = self._detect_segments(original_audio_path)

        # Generate dubbed segments
        output_segments, _ = self._generate_segments(
            lines, timeline, speaker_wav, orig_audio
        )

        # Assemble final audio
        final_audio = AudioSegment.silent(duration=0)
        seg_idx = 0
        for start_ms, end_ms, is_speech in timeline:
            if is_speech:
                if seg_idx in output_segments:
                    final_audio += output_segments[seg_idx]
                else:
                    final_audio += AudioSegment.silent(duration=end_ms - start_ms)
                seg_idx += 1
            else:
                final_audio += AudioSegment.silent(duration=end_ms - start_ms)

        out_path = self._config.dubbed_audio
        final_audio.export(str(out_path), format="wav")
        print(f"[VoiceCloner] Final audio → {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Private helpers – TTS / models
    # ------------------------------------------------------------------

    # XTTS-v2 supported language codes
    _XTTS_LANGS: frozenset[str] = frozenset([
        "en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru",
        "nl", "cs", "ar", "zh-cn", "hu", "ko", "ja", "hi",
    ])

    def _resolve_tts_lang(self) -> str:
        """Return a valid XTTS language code for the configured target language.

        Maps common BCP-47 / ISO codes to the exact strings XTTS-v2 accepts.
        Falls back to ``'hi'`` (Hindi) if the code is not in the supported set
        so synthesis never crashes with an unsupported-language error.
        """
        lang = (self._config.tts_language or "hi").lower().strip()
        # Direct match
        if lang in self._XTTS_LANGS:
            return lang
        # Common aliases
        aliases = {
            "zh": "zh-cn", "cmn": "zh-cn", "zho": "zh-cn",
            "por": "pt",   "spa": "es",   "fra": "fr",
            "deu": "de",   "ita": "it",   "pol": "pl",
            "tur": "tr",   "rus": "ru",   "nld": "nl",
            "ces": "cs",   "ara": "ar",   "hun": "hu",
            "kor": "ko",   "jpn": "ja",   "hin": "hi",
        }
        if lang in aliases:
            return aliases[lang]
        # If the lang code is a source language (e.g. 'ta' for Tamil) that
        # XTTS doesn't support, fall back to Hindi as the default target.
        print(f"[VoiceCloner] Warning: '{lang}' not supported by XTTS-v2. "
              f"Falling back to 'hi'. Supported: {sorted(self._XTTS_LANGS)}")
        return "hi"

    def _ensure_tts_loaded(self) -> None:
        if self._tts is not None:
            return
        import torch
        if hasattr(torch.serialization, "add_safe_globals"):
            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts, XttsArgs, XttsAudioConfig
                from TTS.config.shared_configs import BaseDatasetConfig
                torch.serialization.add_safe_globals(
                    [XttsConfig, Xtts, XttsArgs, XttsAudioConfig, BaseDatasetConfig]
                )
            except Exception as exc:
                print(f"[VoiceCloner] Warning: could not register safe globals: {exc}")

        from TTS.api import TTS

        self._device = select_torch_device(require_nvidia=False)
        print(f"[VoiceCloner] Loading XTTS on {self._device} …")
        is_cuda = self._device.startswith("cuda")
        self._tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            gpu=is_cuda,
        )
        self._tts.to(self._device)
        print("[VoiceCloner] XTTS ready.")

    def _ensure_translation_models_loaded(self) -> None:
        if self._tok_ta_en is not None:
            return
        import torch
        from transformers import MarianMTModel, MarianTokenizer

        dev = self._device or select_torch_device(require_nvidia=False)
        try:
            self._tok_ta_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
            self._mdl_ta_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en").to(dev)
            print("[VoiceCloner] Tamil→English model loaded.")
        except Exception as exc:
            print(f"[VoiceCloner] Tamil→English model failed ({exc}). Trying NLLB fallback …")
            from transformers import NllbTokenizer, M2M100ForConditionalGeneration
            self._tok_ta_en = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            self._tok_ta_en.src_lang = "tam_Taml"
            self._mdl_ta_en = M2M100ForConditionalGeneration.from_pretrained(
                "facebook/nllb-200-distilled-600M"
            ).to(dev)

        try:
            self._tok_en_hi = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
            self._mdl_en_hi = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi").to(dev)
            print("[VoiceCloner] English→Hindi model loaded.")
        except Exception as exc:
            print(f"[VoiceCloner] English→Hindi model failed ({exc}). Trying NLLB fallback …")
            from transformers import NllbTokenizer, M2M100ForConditionalGeneration
            self._tok_en_hi = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            self._tok_en_hi.src_lang = "eng_Latn"
            self._mdl_en_hi = M2M100ForConditionalGeneration.from_pretrained(
                "facebook/nllb-200-distilled-600M"
            ).to(dev)

    # ------------------------------------------------------------------
    # Private helpers – audio processing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_speaker(speaker_wav: Path, output_path: Path) -> Path:
        """Normalise and trim the speaker reference WAV.

        Uses ``res_type='soxr_vhq'`` for fast resampling without mmap
        disk pressure.
        """
        try:
            import librosa
            import soundfile as sf
        except ImportError as exc:
            raise PipelineError("librosa/soundfile not installed.") from exc

        audio, sr = librosa.load(
            str(speaker_wav), sr=24000, mono=True,
            res_type="soxr_vhq",  # fast resampler, avoids mmap disk pressure
        )
        audio = librosa.util.normalize(audio)
        audio, _ = librosa.effects.trim(audio, top_db=25)
        sf.write(str(output_path), audio, sr)
        return output_path

    @staticmethod
    def _detect_segments(
        audio_path: Path, min_silence_len: int = 300, thresh_db: int = 35
    ) -> list[tuple[int, int, bool]]:
        """Return a timeline of (start_ms, end_ms, is_speech) tuples."""
        from pydub import AudioSegment
        from pydub import silence as pydub_silence

        audio = AudioSegment.from_file(str(audio_path))
        total_ms = len(audio)
        try:
            speech_segs = pydub_silence.detect_nonsilent(
                audio,
                min_silence_len=min_silence_len,
                silence_thresh=audio.dBFS - thresh_db,
            )
        except Exception as exc:
            print(f"[VoiceCloner] Segment detection failed: {exc}. Using full clip.")
            return [(0, total_ms, True)]

        timeline: list[tuple[int, int, bool]] = []
        pos = 0
        for s, e in speech_segs:
            if s > pos:
                timeline.append((pos, s, False))
            timeline.append((s, e, True))
            pos = e
        if pos < total_ms:
            timeline.append((pos, total_ms, False))
        return timeline

    def _translate_line(self, line: str) -> str:
        """Translate one Tamil line → English → Hindi."""
        import torch

        dev = self._device or "cpu"
        # Tamil → English
        inp = self._tok_ta_en(
            line, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inp = {k: v.to(dev) for k, v in inp.items()}
        with torch.no_grad():
            en_tok = self._mdl_ta_en.generate(**inp, max_length=512, num_beams=5, early_stopping=True)
        en_text = self._tok_ta_en.batch_decode(en_tok, skip_special_tokens=True)[0]

        # English → Hindi
        inp2 = self._tok_en_hi(
            en_text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inp2 = {k: v.to(dev) for k, v in inp2.items()}
        with torch.no_grad():
            hi_tok = self._mdl_en_hi.generate(**inp2, max_length=512, num_beams=5, early_stopping=True)
        return self._tok_en_hi.batch_decode(hi_tok, skip_special_tokens=True)[0]

    def _split_text(self, text: str) -> list[str]:
        """Split Hindi text into XTTS-safe chunks (≤ _MAX_TOKENS tokens)."""
        sentences = text.replace("।", "|").split("|")
        chunks: list[str] = []
        current = ""
        max_chars = int(self._MAX_TOKENS * 0.65)

        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            candidate = (current + " " + sent).strip() if current else sent
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = sent
            else:
                current = candidate
        if current:
            chunks.append(current)

        # Further split any chunk that is still too long
        final: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                final.append(chunk)
            else:
                words = chunk.split()
                sub = ""
                for word in words:
                    test = (sub + " " + word).strip() if sub else word
                    if len(test) > max_chars and sub:
                        final.append(sub)
                        sub = word
                    else:
                        sub = test
                if sub:
                    final.append(sub)
        return final

    def _generate_segments(
        self,
        lines: list[str],
        timeline: list[tuple[int, int, bool]],
        speaker_wav: Path,
        orig_audio,
    ) -> tuple[dict[int, object], list[tuple[int, int]]]:
        from pydub import AudioSegment

        speech_segs = [(s, e) for s, e, sp in timeline if sp]
        if not speech_segs:
            print("[VoiceCloner] No speech segments detected!")
            return {}, []

        if len(lines) > len(speech_segs):
            lines = lines[: len(speech_segs)]

        out_segments: dict[int, AudioSegment] = {}
        tmp_dir = tempfile.mkdtemp(prefix="vf_clone_")
        failed_errors: list[str] = []

        for idx, (line, (seg_start, seg_end)) in enumerate(zip(lines, speech_segs)):
            print(f"[VoiceCloner] Segment {idx + 1}/{len(speech_segs)} …")
            try:
                # The text is already translated by the NLLB Translator step.
                # Use it directly – do NOT re-translate through Marian models.
                target_text = line
                chunks = self._split_text(target_text)
                tts_lang = self._resolve_tts_lang()
                seg_audio = AudioSegment.silent(duration=0)
                for j, chunk in enumerate(chunks):
                    tmp_file = os.path.join(tmp_dir, f"seg{idx}_chunk{j}.wav")
                    self._tts.tts_to_file(
                        text=chunk,
                        speaker_wav=str(speaker_wav),
                        language=tts_lang,
                        file_path=tmp_file,
                    )
                    seg_audio += AudioSegment.from_wav(tmp_file)
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass
                out_segments[idx] = seg_audio
            except Exception as exc:
                print(f"[VoiceCloner] Segment {idx} synthesis failed: {exc}")
                failed_errors.append(str(exc))
                out_segments[idx] = AudioSegment.silent(
                    duration=seg_end - seg_start
                )

        if lines and len(failed_errors) == len(lines):
            raise PipelineError(
                f"VoiceCloner failed to synthesise any segments: {failed_errors[0]}"
            )

        silence_gaps = [(s, e) for s, e, sp in timeline if not sp]
        return out_segments, silence_gaps
