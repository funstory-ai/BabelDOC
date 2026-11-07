"""Python adapter for translator callbacks used by the Rust translation stages."""

from __future__ import annotations

import re
import threading
from typing import Any


class RustTranslatorService:
    """Expose translator methods through the callback dispatcher."""

    def __init__(
        self,
        translator: Any,
        translation_config: Any | None = None,
        shared_context: Any | None = None,
    ):
        self.translator = translator
        self.translation_config = translation_config
        self.shared_context = shared_context
        self._regex_lock = threading.Lock()
        self._regex_patterns: dict[int, re.Pattern[str]] = {}
        self._regex_keys: dict[tuple[str, int], int] = {}
        self._next_regex_id = 0

    def supports_llm(self) -> bool:
        """Match high_level.translator_supports_llm, including its broad catch."""
        if not hasattr(self.translator, "do_llm_translate"):
            return False
        try:
            self.translator.do_llm_translate(None)
            return True
        except Exception:
            return False

    def probe_il_translator(self) -> bool:
        """The ordinary translator constructor only treats NotImplemented as false."""
        try:
            self.translator.do_llm_translate(None)
            return True
        except NotImplementedError:
            return False

    def probe_llm_only(self) -> bool:
        """The LLM-only constructor converts NotImplemented into ValueError."""
        try:
            self.translator.do_llm_translate(None)
            return True
        except NotImplementedError as error:
            raise ValueError("LLM translator not supported") from error

    def get_formular_placeholder(self, placeholder_id: int | str):
        return self.translator.get_formular_placeholder(placeholder_id)

    def get_rich_text_left_placeholder(self, placeholder_id: int | str):
        return self.translator.get_rich_text_left_placeholder(placeholder_id)

    def get_rich_text_right_placeholder(self, placeholder_id: int | str):
        return self.translator.get_rich_text_right_placeholder(placeholder_id)

    def usage_counters(self) -> tuple[int, int, int, int]:
        def read(name: str) -> int:
            counter = getattr(self.translator, name, None)
            return getattr(counter, "value", 0) if counter is not None else 0

        return (
            read("token_count"),
            read("prompt_token_count"),
            read("completion_token_count"),
            read("cache_hit_prompt_token_count"),
        )

    def record_term_extraction_usage(
        self,
        total_tokens: int,
        translation_tokens: int,
        cache_hit_tokens: int,
        cache_calls: int,
    ) -> None:
        if self.translation_config is None:
            raise RuntimeError(
                "translation_config is required to record term extraction usage"
            )
        self.translation_config.record_term_extraction_usage(
            total_tokens, translation_tokens, cache_hit_tokens, cache_calls
        )

    def translate(self, text: str, rate_limit_params: dict[str, Any] | None = None):
        return self.translator.translate(text, rate_limit_params=rate_limit_params)

    def llm_translate(
        self, prompt: str, rate_limit_params: dict[str, Any] | None = None
    ):
        return self.translator.llm_translate(
            prompt, rate_limit_params=rate_limit_params
        )

    def regex_compile(self, pattern: str, flags: int = 0) -> int:
        key = (pattern, flags)
        with self._regex_lock:
            existing = self._regex_keys.get(key)
            if existing is not None:
                return existing
            pattern_id = self._next_regex_id
            self._next_regex_id += 1
            compiled = re.compile(pattern, flags)
            self._regex_patterns[pattern_id] = compiled
            self._regex_keys[key] = pattern_id
            return pattern_id

    def regex_match(self, pattern_id: int, text: str):
        return self._match_result(self._regex_patterns[pattern_id].match(text))

    def regex_search(self, pattern_id: int, text: str):
        return self._match_result(self._regex_patterns[pattern_id].search(text))

    def regex_finditer(self, pattern_id: int, text: str):
        pattern = self._regex_patterns[pattern_id]
        return [self._match_result(match) for match in pattern.finditer(text)]

    @staticmethod
    def _match_result(match: re.Match[str] | None):
        if match is None:
            return None
        spans = [match.span()]
        spans.extend(match.span(index) for index in range(1, match.re.groups + 1))
        return spans
