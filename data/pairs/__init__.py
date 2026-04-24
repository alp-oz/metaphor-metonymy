from . import (
    english_pairs,
    french_pairs,
    turkish_pairs,
    russian_pairs,
    swedish_pairs,
    german_pairs,
    arabic_pairs,
    japanese_pairs,
)

_MODULES = {
    "English":  english_pairs,
    "French":   french_pairs,
    "Turkish":  turkish_pairs,
    "Russian":  russian_pairs,
    "Swedish":  swedish_pairs,
    "German":   german_pairs,
    "Arabic":   arabic_pairs,
    "Japanese": japanese_pairs,
}

LANGUAGES = list(_MODULES.keys())

DEAD_METONYMY = {lang: m.DEAD_METONYMY for lang, m in _MODULES.items()}
LIVE_METONYMY = {lang: m.LIVE_METONYMY for lang, m in _MODULES.items()}
DEAD_METAPHOR = {lang: m.DEAD_METAPHOR for lang, m in _MODULES.items()}
LIVE_METAPHOR = {lang: m.LIVE_METAPHOR for lang, m in _MODULES.items()}
