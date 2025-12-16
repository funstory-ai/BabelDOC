from typing import Literal

# Normalized language codes supported by i18n prompts
NormalizedLangCode = Literal["ja", "zh", "en"]

# Mapping of various language identifiers to normalized codes
_LANG_ALIASES: dict[NormalizedLangCode, list[str]] = {
    "en": ["en", "eng", "english"],
    "zh": ["zh", "zh-cn", "zh-hans", "zhs", "chinese", "中文", "简体中文"],
    "ja": ["ja", "jp", "jpn", "japanese", "日本語"],
}


def normalize_lang_code(lang: str) -> NormalizedLangCode | None:
    """Normalize a language identifier to a standard code.

    Args:
        lang: A language identifier (e.g., "ja", "JP", "日本語", "Japanese")

    Returns:
        Normalized language code ("ja", "zh", "en") or None if not recognized.
    """
    if lang in _LANG_ALIASES.keys():
        return lang  # type: ignore[misc]

    lang_lower = lang.lower().strip()
    for normalized_code, aliases in _LANG_ALIASES.items():
        if lang_lower in aliases:
            return normalized_code

    return None
