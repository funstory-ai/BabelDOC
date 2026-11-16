"""Native integration regression; requires a built extension and cached font assets."""

import json
import tempfile
import unittest
from pathlib import Path

from babeldoc import _rust_il
from babeldoc.format.pdf.document_il import il_version_1 as il
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.progress_monitor import ProgressMonitor
from babeldoc.rust_run_context import create_document_run_context
from babeldoc.rust_translation_service import RustTranslatorService
from babeldoc.translator.translator import OpenAITranslator


class EchoTranslator(OpenAITranslator):
    def __init__(self, fallback, changed):
        self.fallback = fallback
        self.changed = changed
        self.requests = []

    def do_llm_translate(self, text, rate_limit_params=None):
        assert text is None

    def llm_translate(self, prompt, rate_limit_params=None):
        self.requests.append(prompt)
        if "Now translate the following text:\n\n" in prompt:
            source = prompt.split("Now translate the following text:\n\n")[-1]
            return "Translated: " + source if self.changed else source
        if self.fallback:
            return "invalid JSON"
        inputs = json.loads(prompt[prompt.rindex("\n[") + 1 :])
        return json.dumps(
            [
                {
                    "id": item["id"],
                    "output": (
                        "Translated: " + item["input"]
                        if self.changed
                        else item["input"]
                    ),
                }
                for item in inputs
            ]
        )


def source_document():
    style = il.PdfStyle(font_id="F1", font_size=9)
    lines = ["receive Message", "then wait end."]
    chars = []
    for row, text in enumerate(lines):
        for col, char in enumerate(text):
            x, y = 40 + row * 18 + col * 6, 100 - row * 12
            box = il.Box(x=x, y=y, x2=x + 6, y2=y + 9)
            chars.append(
                il.PdfCharacter(
                    pdf_style=style,
                    box=box,
                    visual_bbox=il.VisualBbox(box=box),
                    char_unicode=char,
                    advance=6,
                    vertical=False,
                    xobj_id=-1,
                )
            )
    paragraph = il.PdfParagraph(
        box=il.Box(x=40, y=88, x2=310, y2=109),
        pdf_style=style,
        unicode=" ".join(lines),
        debug_id="indented",
        layout_label="plain text",
        vertical=False,
        first_line_indent=False,
        xobj_id=-1,
        pdf_paragraph_composition=[
            il.PdfParagraphComposition(
                pdf_same_style_characters=il.PdfSameStyleCharacters(
                    pdf_style=style,
                    pdf_character=chars,
                ),
            )
        ],
    )
    return il.Document(
        page=[
            il.Page(
                page_number=0,
                pdf_font=[il.PdfFont(font_id="F1", name="Courier", monospace=True)],
                pdf_paragraph=[paragraph],
            )
        ]
    )


class TranslationPassthroughTest(unittest.TestCase):
    def test_unchanged_keeps_characters_and_changed_rebuilds(self):
        for path in ("batch", "fallback", "ordinary"):
            for changed in (False, True):
                with (
                    self.subTest(path=path, changed=changed),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    translator = EchoTranslator(path == "fallback", changed)
                    config = TranslationConfig(
                        translator,
                        "fixture.pdf",
                        "en",
                        "zh-CN",
                        object(),
                        working_dir=Path(tmp),
                        disable_same_text_fallback=True,
                        auto_extract_glossary=False,
                    )
                    context = create_document_run_context(config)
                    self.addCleanup(context.close)
                    service = RustTranslatorService(
                        translator, config, context.shared_context
                    )
                    context.dispatcher_a.service = service
                    context.dispatcher_b.service = service
                    native = _rust_il.transfer_document(source_document())
                    before = json.loads(native.artifact())["document"]
                    args = [
                        config,
                        context.shared_context,
                        context.font_mapper,
                        context.dispatcher_a,
                    ]
                    if path != "ordinary":
                        args.append(context.dispatcher_b)
                    args.extend(
                        [
                            service,
                            _rust_il.CancelFlag(),
                            ProgressMonitor([("Translate Paragraphs", 1)]),
                        ]
                    )
                    method = (
                        native.translate_paragraphs
                        if path == "ordinary"
                        else native.translate_paragraphs_llm_only
                    )
                    method(*args)
                    self.assertEqual(
                        len(translator.requests), 2 if path == "fallback" else 1
                    )
                    after = json.loads(native.artifact())["document"]
                    if changed:
                        paragraph = after["page"][0]["pdf_paragraph"][0]
                        self.assertTrue(paragraph["unicode"].startswith("Translated: "))
                        self.assertIsNotNone(
                            paragraph["pdf_paragraph_composition"][0][
                                "pdf_same_style_unicode_characters"
                            ]
                        )
                    else:
                        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
