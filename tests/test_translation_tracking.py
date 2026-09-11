import json
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfFormula
from babeldoc.format.pdf.document_il import PdfLine
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfSameStyleCharacters
from babeldoc.format.pdf.document_il import PdfStyle
from babeldoc.format.pdf.document_il.midend import il_translator
from babeldoc.format.pdf.document_il.midend import il_translator_llm_only
from babeldoc.format.pdf.document_il.midend.il_translator import (
    DocumentTranslateTracker,
)
from babeldoc.format.pdf.document_il.midend.il_translator import FormulaPlaceholder
from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator
from babeldoc.format.pdf.document_il.midend.il_translator import RichTextPlaceholder
from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
)
from babeldoc.format.pdf.result_merger import ResultMerger
from babeldoc.format.pdf.translation_config import SharedContextCrossSplitPart
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.tools.executor import babeldoc_adapter
from babeldoc.translator.translator import BaseTranslator
from babeldoc.translator.translator import OpenAITranslator


def _box(x: float = 1, y: float = 2, x2: float = 3, y2: float = 4):
    return Box(x=x, y=y, x2=x2, y2=y2)


def test_tracking_can_be_disabled_without_retaining_paragraph_data():
    tracker = DocumentTranslateTracker(enabled=False)
    page = tracker.new_page()
    paragraph = page.new_paragraph()
    llm_tracker = paragraph.new_llm_translate_tracker()
    before = set(vars(paragraph))
    llm_before = vars(llm_tracker).copy()

    paragraph.set_pdf_unicode("not retained")
    paragraph.set_geometry(2, _box())
    paragraph.set_layout_label("text")
    paragraph.set_input("not retained")
    paragraph.set_output("not retained")
    paragraph.set_placeholders([object()])
    paragraph.set_original_placeholders({"{v1}": 1})
    paragraph.record_removed_hallucinated_placeholder("{v2}")
    paragraph.record_multi_paragraph_id(1)
    paragraph.record_multi_paragraph_index(2)
    llm_tracker.set_input("not retained")
    llm_tracker.set_output("not retained")
    llm_tracker.set_error_message("not retained")
    llm_tracker.set_placeholder_full_match()
    llm_tracker.set_fallback_to_translate()

    assert tracker.to_dict() == {
        "cross_page": [],
        "cross_column": [],
        "page": [],
    }
    assert tracker.new_page() is tracker.new_cross_page() is tracker.new_cross_column()
    assert page.new_paragraph() is paragraph
    assert paragraph.new_llm_translate_tracker() is llm_tracker
    assert paragraph.last_llm_translate_tracker() is None
    assert vars(llm_tracker) == llm_before
    assert set(vars(paragraph)) == before
    assert paragraph.original_placeholders == {}
    assert paragraph.removed_hallucinated_placeholders == {}
    assert paragraph.llm_translate_trackers == []


@pytest.mark.parametrize(
    ("include_char_boxes", "expected_char_boxes"),
    [
        (False, []),
        (
            True,
            [
                {
                    "char": "x",
                    "box": {"x": 5, "y": 6, "x2": 7, "y2": 8},
                }
            ],
        ),
    ],
)
def test_tracking_serializes_geometry_and_optional_character_boxes(
    include_char_boxes,
    expected_char_boxes,
):
    tracker = DocumentTranslateTracker(char_boxes=include_char_boxes)
    paragraph = tracker.new_page().new_paragraph()
    paragraph.set_pdf_unicode("x")
    paragraph.set_input("x")
    paragraph.set_output("y")
    paragraph.set_geometry(2, _box())
    paragraph.set_layout_label("text")
    paragraph.set_placeholders(
        [
            FormulaPlaceholder(
                1,
                PdfFormula(
                    box=_box(),
                    pdf_character=[
                        PdfCharacter(char_unicode="x", box=_box(5, 6, 7, 8))
                    ],
                ),
                "{v1}",
                r"\{v1\}",
            ),
            RichTextPlaceholder(
                2,
                PdfSameStyleCharacters(
                    box=_box(),
                    pdf_character=[
                        PdfCharacter(char_unicode="x", box=_box(5, 6, 7, 8))
                    ],
                ),
                "<b2>",
                "</b2>",
            ),
        ]
    )

    serialized_document = json.loads(tracker.to_json())
    assert serialized_document == tracker.to_dict()
    serialized = serialized_document["page"][0]["paragraph"][0]

    assert serialized["input"] == serialized["pdf_unicode"] == "x"
    assert serialized["output"] == "y"
    assert serialized["page_number"] == 2
    assert serialized["box"] == {"x": 1, "y": 2, "x2": 3, "y2": 4}
    assert serialized["layout_label"] == "text"
    for placeholder in serialized["placeholders"]:
        assert placeholder["box"] == serialized["box"]
        assert placeholder["char_boxes"] == expected_char_boxes
    assert serialized["placeholders"][0]["formula_chars"] == "x"
    assert serialized["placeholders"][1]["composition_chars"] == "x"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("debug", [False, True])
@pytest.mark.parametrize(
    ("translator_type", "section"),
    [
        (ILTranslator, "page"),
        (ILTranslatorLLMOnly, "page"),
        (ILTranslatorLLMOnly, "cross_page"),
        (ILTranslatorLLMOnly, "cross_column"),
    ],
)
def test_translation_tracks_geometry_only_when_requested(
    monkeypatch, tmp_path, translator_type, section, enabled, debug
):
    # Keep translation, batching and serialization real; replace external services.
    monkeypatch.setattr(il_translator, "FontMapper", Mock())
    monkeypatch.setattr(il_translator_llm_only, "FontMapper", Mock())
    engine = Mock(spec=BaseTranslator)
    for method in (
        "get_formular_placeholder",
        "get_rich_text_left_placeholder",
        "get_rich_text_right_placeholder",
    ):
        getattr(engine, method).side_effect = getattr(OpenAITranslator, method).__get__(
            engine
        )
    if translator_type is ILTranslator:
        engine.do_llm_translate.side_effect = NotImplementedError
    engine.translate.side_effect = lambda text, **_kwargs: text.replace(
        "source", "cible"
    )

    def translate_batch(prompt, **_kwargs):
        inputs = json.loads(prompt.split("## Here is the input:", 1)[1])
        return json.dumps(
            [
                {"id": item["id"], "output": item["input"].replace("source", "cible")}
                for item in inputs
            ]
        )

    engine.llm_translate.side_effect = translate_batch
    pbar = Mock()
    config = SimpleNamespace(
        enable_translation_tracking=enabled,
        debug=debug,
        shared_context_cross_split_part=SharedContextCrossSplitPart(),
        auto_extract_glossary=False,
        lang_out="fr",
        disable_rich_text_translate=False,
        disable_same_text_fallback=False,
        add_formula_placehold_hint=False,
        min_text_length=1,
        pool_max_workers=2,
        progress_monitor=SimpleNamespace(
            stage_start=Mock(return_value=nullcontext(pbar))
        ),
        raise_if_cancelled=Mock(),
        working_dir=tmp_path,
        get_working_file_path=Mock(side_effect=lambda name: tmp_path / name),
    )
    tokenizer = SimpleNamespace(encode=lambda text, **_kwargs: text.split())
    translator = translator_type(engine, config, tokenizer=tokenizer)
    paragraphs = []
    for index, box in enumerate([_box(), _box(10, 20, 30, 40)]):
        text = f"source {index} "
        paragraphs.append(
            PdfParagraph(
                unicode=text + "x",
                debug_id=str(index),
                box=box,
                layout_label="title" if section == "page" and index == 0 else "text",
                pdf_style=PdfStyle(font_id="test", font_size=10),
                pdf_paragraph_composition=[
                    PdfParagraphComposition(
                        pdf_line=PdfLine(
                            pdf_character=[
                                PdfCharacter(char_unicode=c, box=_box()) for c in text
                            ]
                        )
                    ),
                    PdfParagraphComposition(
                        pdf_formula=PdfFormula(
                            box=_box(),
                            pdf_character=[PdfCharacter(char_unicode="x", box=_box())],
                        )
                    ),
                ],
            )
        )
    if section == "cross_page":
        pages = [
            Page(page_number=3 + index, pdf_paragraph=[paragraph])
            for index, paragraph in enumerate(paragraphs)
        ]
    else:
        pages = [Page(page_number=3, pdf_paragraph=paragraphs)]

    tracker = translator.translate(Document(page=pages))

    assert [p.unicode for p in paragraphs] == ["cible 0 {v1}", "cible 1 {v1}"]
    assert (
        sum(call.args[0] if call.args else 1 for call in pbar.advance.call_args_list)
        == 2
    )
    if translator_type is ILTranslatorLLMOnly:
        assert translator.ok_count == 2
        assert translator.fallback_count == 0
        engine.translate.assert_not_called()
    tracking_file = tmp_path / "translate_tracking.json"
    assert tracking_file.exists() is (enabled or debug)
    if not (enabled or debug):
        config.get_working_file_path.assert_not_called()
        assert tracker.to_dict() == {"page": [], "cross_page": [], "cross_column": []}
        return

    serialized_document = json.loads(tracking_file.read_text(encoding="utf-8"))
    assert serialized_document == tracker.to_dict()
    serialized = serialized_document[section][0]["paragraph"]
    assert len(serialized) == 2
    assert [p["page_number"] for p in serialized] == (
        [3, 4] if section == "cross_page" else [3, 3]
    )
    assert [p["layout_label"] for p in serialized] == [
        p.layout_label for p in paragraphs
    ]
    assert serialized[1]["box"] == {"x": 10, "y": 20, "x2": 30, "y2": 40}
    for index, paragraph in enumerate(serialized):
        assert paragraph["pdf_unicode"] == f"source {index} x"
        assert paragraph["input"] == f"source {index} {{v1}}"
        assert paragraph["output"] == f"cible {index} {{v1}}"
        assert bool(paragraph["placeholders"][0]["char_boxes"]) is enabled
        if translator_type is ILTranslatorLLMOnly:
            assert paragraph["multi_paragraph_index"] == index
            attempt = paragraph["llm_translate_trackers"][0]
            assert attempt["input"] and attempt["output"]
            assert not attempt["has_error"]
            assert attempt["placeholder_full_match"]
    if translator_type is ILTranslatorLLMOnly:
        assert (
            serialized[0]["multi_paragraph_id"] == serialized[1]["multi_paragraph_id"]
        )


def test_merging_tracking_does_not_mutate_or_alias_part_results():
    part_tracking = {
        "cross_page": [],
        "cross_column": [],
        "page": [
            {
                "paragraph": [
                    {
                        "input": "source",
                        "multi_paragraph_id": 7,
                        "placeholders": [{"char_boxes": [{"char": "x"}]}],
                    }
                ]
            }
        ],
    }
    other_tracking = deepcopy(part_tracking)
    other_tracking["cross_page"] = other_tracking.pop("page")
    other_tracking["page"] = []
    original_part = deepcopy(part_tracking)
    original_other = deepcopy(other_tracking)
    part_result = SimpleNamespace(translation_tracking=part_tracking)
    other_result = SimpleNamespace(translation_tracking=other_tracking)
    merger = object.__new__(ResultMerger)

    merged = merger._merge_translation_tracking([part_result, other_result])

    assert merged["page"][0]["paragraph"][0]["multi_paragraph_id"] == 0
    assert merged["cross_page"][0]["paragraph"][0]["multi_paragraph_id"] == 1
    assert part_tracking == original_part
    assert other_tracking == original_other
    assert merger._merge_translation_tracking([part_result, other_result]) == merged
    merged["page"][0]["paragraph"][0]["input"] = "changed"
    merged["cross_page"][0]["paragraph"][0]["placeholders"][0]["char_boxes"][0][
        "char"
    ] = "y"
    assert part_tracking == original_part
    assert other_tracking == original_other


@pytest.mark.parametrize("request_value", ["omitted", True, False, None, "true", 0, 1])
def test_executor_forwards_optional_translation_tracking_flag(
    monkeypatch,
    tmp_path,
    request_value,
):
    class FakeTranslationConfig:
        @staticmethod
        def create_max_pages_per_part_split_strategy(max_pages):
            return max_pages

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeLayoutModel:
        def init_font_mapper(self, config):
            self.config = config

    monkeypatch.setattr(babeldoc_adapter, "TranslationConfig", FakeTranslationConfig)
    monkeypatch.setattr(babeldoc_adapter, "get_workroot", lambda: tmp_path)
    monkeypatch.setattr(
        babeldoc_adapter,
        "resolve_file",
        lambda workroot, value: workroot / value,
    )
    monkeypatch.setattr(
        babeldoc_adapter,
        "resolve_dir",
        lambda workroot, value, create: workroot / value,  # noqa: ARG005
    )
    monkeypatch.setattr(
        babeldoc_adapter,
        "set_translate_rate_limiter",
        lambda qps: None,  # noqa: ARG005
    )
    monkeypatch.setattr(
        babeldoc_adapter,
        "_create_translator",
        lambda gateway, translation: object(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        babeldoc_adapter,
        "_create_doc_layout_model",
        lambda layout: FakeLayoutModel(),  # noqa: ARG005
    )
    monkeypatch.setattr(
        babeldoc_adapter,
        "_load_glossaries",
        lambda workroot, assets, lang_out: [],  # noqa: ARG005
    )
    translation_config = {
        "debug": False,
        "lang_in": "en",
        "lang_out": "fr",
        "no_dual": False,
        "no_mono": False,
        "skip_clean": False,
        "dual_translate_first": False,
        "disable_rich_text_translate": False,
        "use_side_by_side_dual": True,
        "use_alternating_pages_dual": False,
        "skip_scanned_detection": False,
        "ocr_workaround": False,
        "auto_extract_glossary": False,
        "auto_enable_ocr_workaround": False,
        "only_include_translated_page": False,
        "merge_alternating_line_numbers": True,
        "remove_non_formula_lines": False,
    }
    if request_value != "omitted":
        translation_config["enable_translation_tracking"] = request_value
    request = {
        "paths": {"input_file": "input.pdf", "output_dir": "output"},
        "translation_config": translation_config,
        "runtime_limits": {
            "qps": 1,
            "report_interval_seconds": 0.1,
            "max_pages_per_part": 10,
            "pool_max_workers": 1,
            "term_pool_max_workers": 1,
        },
        "gateways": {"main_llm": {}, "ate_llm": {}, "layout": {}},
    }

    if request_value == "omitted" or isinstance(request_value, bool):
        config = babeldoc_adapter.build_translation_config(request)
        assert config.kwargs["enable_translation_tracking"] is (request_value is True)
    else:
        with pytest.raises(
            ValueError, match="enable_translation_tracking must be a boolean"
        ):
            babeldoc_adapter.build_translation_config(request)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tracking_available", [False, True])
def test_executor_payload_only_includes_requested_tracking(
    monkeypatch, tmp_path, enabled, tracking_available
):
    monkeypatch.setenv("BABELDOC_EXECUTOR_WORKROOT", str(tmp_path))
    result = TranslateResult(None, None)
    if tracking_available:
        tracker = DocumentTranslateTracker()
        paragraph = tracker.new_page().new_paragraph()
        paragraph.set_pdf_unicode("source")
        paragraph.set_input("source")
        paragraph.set_output("cible")
        paragraph.set_geometry(2, _box())
        paragraph.set_layout_label("text")
        result.translation_tracking = tracker.to_dict()
    config = SimpleNamespace(
        enable_translation_tracking=enabled,
        pages=None,
        page_ranges=None,
    )

    payload = babeldoc_adapter.translate_result_to_payload(result, config)

    assert ("translation_tracking" in payload) is enabled
    assert json.loads(json.dumps(payload)) == payload
    if enabled:
        assert payload["translation_tracking"] == result.translation_tracking
