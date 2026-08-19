from types import SimpleNamespace

from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator


class _LLMTracker:
    def __init__(self) -> None:
        self.placeholder_full_match = False

    def set_placeholder_full_match(self) -> None:
        self.placeholder_full_match = True


class _Tracker:
    def __init__(self) -> None:
        self.output: str | None = None
        self.llm_tracker = _LLMTracker()

    def set_output(self, output: str) -> None:
        self.output = output

    def last_llm_translate_tracker(self) -> _LLMTracker:
        return self.llm_tracker


def test_unchanged_translation_preserves_paragraph_compositions() -> None:
    translator = ILTranslator.__new__(ILTranslator)
    translator.parse_translate_output = lambda *_args: []
    source = "Chapter 1" + "." * 20
    translate_input = ILTranslator.TranslateInput(source, [], None)
    original_compositions = [object()]
    paragraph = SimpleNamespace(
        unicode=source,
        pdf_paragraph_composition=original_compositions,
        pdf_style=None,
    )
    tracker = _Tracker()

    changed = translator.post_translate_paragraph(
        paragraph,
        tracker,
        translate_input,
        "Chapter 1.",
    )

    assert changed is False
    assert tracker.output == "Chapter 1."
    assert tracker.llm_tracker.placeholder_full_match is True
    assert paragraph.unicode == source
    assert paragraph.pdf_paragraph_composition is original_compositions
