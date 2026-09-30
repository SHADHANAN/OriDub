"""
VoiceFusion AI – Language Registry
====================================
Single source of truth for all supported languages.

Provides:
  - Whisper language codes  (2-letter ISO 639-1)
  - NLLB FLORES-200 codes   (used by the Translator)
  - XTTS BCP-47 codes       (used by the VoiceCloner for TTS)
  - Human-readable labels

Usage
-----
>>> from voicefusion.languages import LANGUAGES, whisper_to_nllb, nllb_to_tts
>>> LANGUAGES["ta"]
{'label': 'Tamil', 'nllb': 'tam_Taml', 'tts': 'ta', 'flag': '🇮🇳'}
"""
from __future__ import annotations

from typing import TypedDict


class LangEntry(TypedDict):
    label: str
    nllb: str   # FLORES-200 code for NLLB translation
    tts: str    # BCP-47 code for XTTS synthesis
    flag: str


# ---------------------------------------------------------------------------
# Master language table  (keyed by Whisper ISO 639-1 code)
# ---------------------------------------------------------------------------
LANGUAGES: dict[str, LangEntry] = {
    "ta": {"label": "Tamil",      "nllb": "tam_Taml", "tts": "ta", "flag": "🇮🇳"},
    "hi": {"label": "Hindi",      "nllb": "hin_Deva", "tts": "hi", "flag": "🇮🇳"},
    "te": {"label": "Telugu",     "nllb": "tel_Telu", "tts": "te", "flag": "🇮🇳"},
    "kn": {"label": "Kannada",    "nllb": "kan_Knda", "tts": "kn", "flag": "🇮🇳"},
    "ml": {"label": "Malayalam",  "nllb": "mal_Mlym", "tts": "ml", "flag": "🇮🇳"},
    "bn": {"label": "Bengali",    "nllb": "ben_Beng", "tts": "bn", "flag": "🇧🇩"},
    "en": {"label": "English",    "nllb": "eng_Latn", "tts": "en", "flag": "🇬🇧"},
    "fr": {"label": "French",     "nllb": "fra_Latn", "tts": "fr", "flag": "🇫🇷"},
    "es": {"label": "Spanish",    "nllb": "spa_Latn", "tts": "es", "flag": "🇪🇸"},
    "de": {"label": "German",     "nllb": "deu_Latn", "tts": "de", "flag": "🇩🇪"},
    "ja": {"label": "Japanese",   "nllb": "jpn_Jpan", "tts": "ja", "flag": "🇯🇵"},
    "zh": {"label": "Chinese",    "nllb": "zho_Hans", "tts": "zh-cn", "flag": "🇨🇳"},
    "ar": {"label": "Arabic",     "nllb": "arb_Arab", "tts": "ar", "flag": "🇸🇦"},
    "ru": {"label": "Russian",    "nllb": "rus_Cyrl", "tts": "ru", "flag": "🇷🇺"},
    "ko": {"label": "Korean",     "nllb": "kor_Hang", "tts": "ko", "flag": "🇰🇷"},
    "pt": {"label": "Portuguese", "nllb": "por_Latn", "tts": "pt", "flag": "🇧🇷"},
    "tr": {"label": "Turkish",    "nllb": "tur_Latn", "tts": "tr", "flag": "🇹🇷"},
    "it": {"label": "Italian",    "nllb": "ita_Latn", "tts": "it", "flag": "🇮🇹"},
}

# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def whisper_to_nllb(whisper_code: str) -> str:
    """Convert a Whisper ISO code to NLLB FLORES-200 code.

    Falls back to ``eng_Latn`` for unknown codes.
    """
    entry = LANGUAGES.get(whisper_code)
    return entry["nllb"] if entry else "eng_Latn"


def whisper_to_tts(whisper_code: str) -> str:
    """Convert a Whisper ISO code to a BCP-47 TTS language code."""
    entry = LANGUAGES.get(whisper_code)
    return entry["tts"] if entry else "en"


def nllb_to_whisper(nllb_code: str) -> str:
    """Reverse-lookup: NLLB code → Whisper ISO code."""
    for code, entry in LANGUAGES.items():
        if entry["nllb"] == nllb_code:
            return code
    return "en"


def label_for(whisper_code: str) -> str:
    """Return a human-readable language name."""
    entry = LANGUAGES.get(whisper_code)
    return entry["label"] if entry else whisper_code.upper()


def all_language_options() -> list[dict]:
    """Return a JSON-serialisable list of all supported languages.

    Each item: { code, label, nllb, tts, flag }
    """
    return [
        {
            "code": code,
            "label": entry["label"],
            "nllb": entry["nllb"],
            "tts": entry["tts"],
            "flag": entry["flag"],
        }
        for code, entry in LANGUAGES.items()
    ]
