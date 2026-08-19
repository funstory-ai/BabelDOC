from types import SimpleNamespace

from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator


class _Tracker:
    def __init__(self) -> None:
        self.output: str | None = None

    def set_output(self, output: str) -> None:
        self.output = output

    def last_llm_translate_tracker(self) -> None:
        return None


def test_unchanged_translation_preserves_paragraph_compositions() -> None:
    translator = ILTranslator.__new__(ILTranslator)
    translator.parse_translate_output = lambda *_args: []
    translate_input = ILTranslator.TranslateInput("unchanged", [], None)
    original_compositions = [object()]
    paragraph = SimpleNamespace(
        unicode="unchanged",
        pdf_paragraph_composition=original_compositions,
        pdf_style=None,
    )
    tracker = _Tracker()

    changed = translator.post_translate_paragraph(
        paragraph,
        tracker,
        translate_input,
        "unchanged",
    )

    assert changed is False
    assert tracker.output == "unchanged"
    assert paragraph.unicode == "unchanged"
    assert paragraph.pdf_paragraph_composition is original_compositions
