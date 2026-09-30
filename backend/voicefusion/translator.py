"""
VoiceFusion AI – Translator
============================
Translates Tamil text → Hindi using Meta AI NLLB-200 via Hugging Face.

Design decisions
----------------
* Text is segmented into sentence-level chunks to stay within the
  512-token model limit without losing cross-sentence context.
* When ``preserve_slang=True`` the casual/colloquial tone is preserved;
  when ``False`` the ``Transcriber.normalise()`` output should be fed in.
* The heavy models are loaded lazily (on first ``translate()`` call) so
  that importing this class does not trigger a multi-GB download.
"""
from __future__ import annotations

import re
from pathlib import Path

from .config import PipelineConfig, PipelineError, select_torch_device


class Translator:
    """Translates text between languages using NLLB-200.

    Parameters
    ----------
    config:
        Shared pipeline configuration object.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._tokenizer = None
        self._model = None
        self._device: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """Translate *text* and persist the result.

        Parameters
        ----------
        text:
            Source-language text (default: Tamil).

        Returns
        -------
        str
            Translated text in the target language (default: Hindi).
        """
        if not text:
            raise PipelineError("Cannot translate empty text.")

        self._ensure_models_loaded()

        sentences = self._segment(text, self._config.source_lang)
        chunks = self._build_chunks(sentences)

        print(f"[Translator] Translating {len(chunks)} chunk(s) …")
        translated_parts: list[str] = [
            self._translate_chunk(chunk) for chunk in chunks
        ]
        translated = " ".join(translated_parts)

        if self._config.translated_file:
            self._config.translated_file.parent.mkdir(parents=True, exist_ok=True)
            self._config.translated_file.write_text(translated, encoding="utf-8")
            print(f"[Translator] Saved translation → {self._config.translated_file}")

        # ── VRAM cleanup ────────────────────────────────────────────────────
        # Unload NLLB so XTTS-v2 (~2 GB) can load on a 4 GB card.
        del self._model, self._tokenizer
        self._model = None
        self._tokenizer = None
        try:
            import torch, gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mb = torch.cuda.mem_get_info()[0] // (1024 * 1024)
                print(f"[Translator] VRAM freed. Available: {free_mb} MB")
        except Exception:
            pass
        # ────────────────────────────────────────────────────────────────────

        return translated

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_models_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return  # already loaded

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise PipelineError(
                f"Translation dependency import failed: {exc}. Please verify torch and transformers are installed."
            ) from exc

        self._device = select_torch_device(require_nvidia=False)
        model_name = f"facebook/nllb-200-{self._config.nllb_model_size}"
        print(f"[Translator] Loading {model_name} on {self._device} …")

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, src_lang=self._config.source_lang
            )
            dtype = torch.float16 if (self._device.startswith("cuda") and torch.cuda.is_available()) else torch.float32
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                dtype=dtype,
            ).to(self._device)
            print("[Translator] Models loaded successfully.")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise PipelineError(
                f"Failed to load translation model '{model_name}': {exc}"
            ) from exc

    def _segment(self, text: str, source_lang: str) -> list[str]:
        """Split text on sentence boundaries appropriate for the source language."""
        if source_lang == "tam_Taml":
            parts = re.split(r"[.।]", text)
        else:
            parts = text.split(". ")
        return [p.strip() for p in parts if p.strip()]

    def _build_chunks(self, sentences: list[str]) -> list[str]:
        """Group sentences into chunks that fit within 512 tokens."""
        max_tokens = 460  # conservative limit
        chunks: list[str] = []
        current = ""
        for sentence in sentences:
            candidate = (current + " " + sentence).strip() if current else sentence
            token_count = len(self._tokenizer.encode(candidate))
            if token_count > max_tokens and current:
                chunks.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    def _translate_chunk(self, text: str) -> str:
        import torch

        preserve = self._config.preserve_slang
        inputs = self._tokenizer(
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=512,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        target_id = self._tokenizer.convert_tokens_to_ids(self._config.target_lang)
        is_cuda = self._device.startswith("cuda") if self._device else False
        gen_kwargs: dict = dict(
            forced_bos_token_id=target_id,
            max_new_tokens=256,
            num_beams=2 if is_cuda else 1,
            num_return_sequences=1,
            do_sample=False,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        with torch.no_grad():
            output_tokens = self._model.generate(**inputs, **gen_kwargs)

        return self._tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
