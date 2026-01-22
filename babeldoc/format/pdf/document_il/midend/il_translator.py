from __future__ import annotations

import copy
import json
import logging
import re
import threading
from pathlib import Path
from string import Template

import tiktoken
from tqdm import tqdm

import babeldoc.format.pdf.document_il.il_version_1 as il_version_1
from babeldoc.babeldoc_exception.BabelDOCException import ContentFilterError
from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import GraphicState
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfFont
from babeldoc.format.pdf.document_il import PdfFormula
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfSameStyleCharacters
from babeldoc.format.pdf.document_il import PdfSameStyleUnicodeCharacters
from babeldoc.format.pdf.document_il import PdfStyle
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string
from babeldoc.format.pdf.document_il.utils.layout_helper import get_paragraph_unicode
from babeldoc.format.pdf.document_il.utils.layout_helper import is_same_style
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    is_same_style_except_font,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    is_same_style_except_size,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_placeholder_only_paragraph,
)
from babeldoc.format.pdf.document_il.utils.paragraph_helper import (
    is_pure_numeric_paragraph,
)
from babeldoc.format.pdf.document_il.utils.style_helper import GRAY80
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.translator.translator import BaseTranslator
from babeldoc.utils.priority_thread_pool_executor import PriorityThreadPoolExecutor

logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = Template(
    """$role_block

## Rules

1. Keep the structure exactly unchanged: do NOT add/remove/reorder any tags, placeholders, or tokens.
2. Keep all tags unchanged (e.g., <style>, <b>, </style>).
   - Translate human-readable text inside tags.
   - Do NOT translate text inside <code>…</code>.
3. Do NOT translate or alter placeholders: {v1}, {name}, %s, %d, [[...]], %%...%%.
4. If the entire input is pure code/identifiers, return it unchanged.
5. Translate ALL human-readable content into $lang_out.

$glossary_block

$context_block

## Output

Output ONLY the translated $lang_out text. No explanations, no backticks, no extra text.

Now translate the following text:

$text_to_translate"""
)


class RichTextPlaceholder:
    def __init__(
        self,
        placeholder_id: int,
        composition: PdfSameStyleCharacters,
        left_placeholder: str,
        right_placeholder: str,
        left_regex_pattern: str = None,
        right_regex_pattern: str = None,
    ):
        self.id = placeholder_id
        self.composition = composition
        self.left_placeholder = left_placeholder
        self.right_placeholder = right_placeholder
        self.left_regex_pattern = left_regex_pattern
        self.right_regex_pattern = right_regex_pattern

    def to_dict(self) -> dict:
        return {
            "type": "rich_text",
            "id": self.id,
            "left_placeholder": self.left_placeholder,
            "right_placeholder": self.right_placeholder,
            "left_regex_pattern": self.left_regex_pattern,
            "right_regex_pattern": self.right_regex_pattern,
            "composition_chars": get_char_unicode_string(self.composition.pdf_character)
            if self.composition and self.composition.pdf_character
            else None,
        }


class FormulaPlaceholder:
    def __init__(
        self,
        placeholder_id: int,
        formula: PdfFormula,
        placeholder: str,
        regex_pattern: str,
    ):
        self.id = placeholder_id
        self.formula = formula
        self.placeholder = placeholder
        self.regex_pattern = regex_pattern

    def to_dict(self) -> dict:
        return {
            "type": "formula",
            "id": self.id,
            "placeholder": self.placeholder,
            "regex_pattern": self.regex_pattern,
            "formula_chars": get_char_unicode_string(self.formula.pdf_character)
            if self.formula and self.formula.pdf_character
            else None,
        }


class PbarContext:
    def __init__(self, pbar):
        self.pbar = pbar

    def __enter__(self):
        return self.pbar

    def __exit__(self, exc_type, exc_value, traceback):
        self.pbar.advance()


class DocumentTranslateTracker:
    def __init__(self):
        self.page = []
        self.cross_page = []
        # Track paragraphs that are combined due to cross-column detection within the same page
        self.cross_column = []

    def new_page(self):
        page = PageTranslateTracker()
        self.page.append(page)
        return page

    def new_cross_page(self):
        page = PageTranslateTracker()
        self.cross_page.append(page)
        return page

    def new_cross_column(self):
        """Create and return a new PageTranslateTracker dedicated to cross-column merging."""
        page = PageTranslateTracker()
        self.cross_column.append(page)
        return page

    def to_json(self):
        pages = []
        for page in self.page:
            paragraphs = self.convert_paragraph(page)
            pages.append({"paragraph": paragraphs})
        cross_page = []
        for page in self.cross_page:
            paragraphs = self.convert_paragraph(page)
            cross_page.append({"paragraph": paragraphs})
        cross_column = []
        for page in self.cross_column:
            paragraphs = self.convert_paragraph(page)
            cross_column.append({"paragraph": paragraphs})
        return json.dumps(
            {
                "cross_page": cross_page,
                "cross_column": cross_column,
                "page": pages,
            },
            ensure_ascii=False,
            indent=2,
        )

    def convert_paragraph(self, page):
        paragraphs = []
        for para in page.paragraph:
            i_str = getattr(para, "input", None)
            o_str = getattr(para, "output", None)
            pdf_unicode = getattr(para, "pdf_unicode", None)
            llm_translate_trackers = getattr(para, "llm_translate_trackers", None)
            placeholders = getattr(para, "placeholders", None)
            original_placeholders = getattr(para, "original_placeholders", None)
            removed_hallucinated_placeholders = getattr(
                para,
                "removed_hallucinated_placeholders",
                None,
            )

            llm_translate_trackers_json = []
            if llm_translate_trackers:
                for tracker in llm_translate_trackers:
                    llm_translate_trackers_json.append(tracker.to_dict())

            placeholders_json = []
            if placeholders:
                for placeholder in placeholders:
                    placeholders_json.append(placeholder.to_dict())

            if pdf_unicode is None or i_str is None:
                continue
            paragraph_json = {
                "input": i_str,
                "output": o_str,
                "pdf_unicode": pdf_unicode,
                "llm_translate_trackers": llm_translate_trackers_json,
                "placeholders": placeholders_json,
                "multi_paragraph_id": getattr(para, "multi_paragraph_id", None),
                "multi_paragraph_index": getattr(para, "multi_paragraph_index", None),
                "original_placeholders": original_placeholders,
                "removed_hallucinated_placeholders": removed_hallucinated_placeholders,
                "layout_label": getattr(para, "layout_label", None),
                "page_no": getattr(para, "page_no", None),
                "para_index_in_page": getattr(para, "para_index_in_page", None),
                "bbox": getattr(para, "bbox", None),
                "token_count": getattr(para, "token_count", None),
            }
            paragraphs.append(
                paragraph_json,
            )
        return paragraphs


class PageTranslateTracker:
    def __init__(self):
        self.paragraph = []

    def new_paragraph(self):
        paragraph = ParagraphTranslateTracker()
        self.paragraph.append(paragraph)
        return paragraph


class ParagraphTranslateTracker:
    def __init__(self):
        self.llm_translate_trackers = []
        self.original_placeholders: dict[str, int] = {}
        self.removed_hallucinated_placeholders: dict[str, int] = {}
        self.layout_label: str | None = None
        self.page_no: int | None = None
        self.para_index_in_page: int | None = None
        self.bbox: dict[str, float | None] | None = None
        self.token_count: int | None = None

    def set_pdf_unicode(self, unicode: str):
        self.pdf_unicode = unicode

    def set_input(self, input_text: str):
        self.input = input_text

    def set_placeholders(
        self, placeholders: list[RichTextPlaceholder | FormulaPlaceholder]
    ):
        self.placeholders = placeholders

    def set_original_placeholders(self, placeholders: dict[str, int] | None):
        """Record original placeholder-like tokens from the source text."""
        self.original_placeholders = placeholders or {}

    def record_multi_paragraph_id(self, mid):
        self.multi_paragraph_id = mid

    def record_multi_paragraph_index(self, index):
        self.multi_paragraph_index = index

    def set_output(self, output: str):
        self.output = output

    def set_token_count(self, token_count: int):
        """Record token count of the input text."""
        self.token_count = token_count

    def record_layout_info(
        self,
        layout_label: str | None,
        page_no: int | None,
        para_index_in_page: int | None,
        bbox: dict[str, float | None] | None = None,
    ):
        """Record layout label, page number (1-based), paragraph index within page (1-based), and bbox."""
        self.layout_label = layout_label
        self.page_no = page_no
        self.para_index_in_page = para_index_in_page
        self.bbox = bbox

    def record_removed_hallucinated_placeholder(self, token: str):
        """Record placeholder-like tokens removed from translated text."""
        if not token:
            return
        self.removed_hallucinated_placeholders[token] = (
            self.removed_hallucinated_placeholders.get(token, 0) + 1
        )

    def new_llm_translate_tracker(self) -> LLMTranslateTracker:
        tracker = LLMTranslateTracker()
        self.llm_translate_trackers.append(tracker)
        return tracker

    def last_llm_translate_tracker(self) -> LLMTranslateTracker | None:
        if self.llm_translate_trackers:
            return self.llm_translate_trackers[-1]
        return None


class LLMTranslateTracker:
    def __init__(self):
        self.input = ""
        self.output = ""
        self.has_error = False
        self.error_message = ""
        self.placeholder_full_match = False
        self.fallback_to_translate = False

    def set_input(self, input_text: str):
        self.input = input_text

    def set_output(self, output_text: str):
        self.output = output_text

    def set_error_message(self, error_message: str):
        self.has_error = True
        self.error_message = error_message

    def set_placeholder_full_match(self):
        self.placeholder_full_match = True

    def set_fallback_to_translate(self):
        self.fallback_to_translate = True

    def to_dict(self):
        return {
            "input": self.input,
            "output": self.output,
            "has_error": self.has_error,
            "error_message": self.error_message,
            "placeholder_full_match": self.placeholder_full_match,
            "fallback_to_translate": self.fallback_to_translate,
        }


class ILTranslator:
    stage_name = "Translate Paragraphs"

    def __init__(
        self,
        translate_engine: BaseTranslator,
        translation_config: TranslationConfig,
        tokenizer=None,
    ):
        self.translate_engine = translate_engine
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)
        self.shared_context_cross_split_part = (
            translation_config.shared_context_cross_split_part
        )
        if tokenizer is None:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        else:
            self.tokenizer = tokenizer

        # Cache glossaries at initialization
        self._cached_glossaries = (
            self.shared_context_cross_split_part.get_glossaries_for_translation(
                self.translation_config.auto_extract_glossary
            )
        )

        self.support_llm_translate = False
        try:
            if translate_engine and hasattr(translate_engine, "do_llm_translate"):
                translate_engine.do_llm_translate(None)
                self.support_llm_translate = True
        except NotImplementedError:
            self.support_llm_translate = False

        self.use_as_fallback = False
        self.add_content_filter_hint_lock = threading.Lock()
        self.docs = None

        # Pre-compile patterns for placeholder-like tokens that may be hallucinated by LLM.
        # We only consider the same shapes as our own formula & rich-text placeholders.
        self._formula_placeholder_pattern = re.compile(
            self.translate_engine.get_formular_placeholder(r"\d+")[1], re.IGNORECASE
        )
        self._style_left_placeholder_pattern = re.compile(
            self.translate_engine.get_rich_text_left_placeholder(r"\d+")[1],
            re.IGNORECASE,
        )
        self._style_right_placeholder_pattern = re.compile(
            self.translate_engine.get_rich_text_right_placeholder(r"\d+")[1],
            re.IGNORECASE,
        )

    def calc_token_count(self, text: str) -> int:
        try:
            return len(self.tokenizer.encode(text, disallowed_special=()))
        except Exception:
            return 0

    def translate(self, docs: Document):
        self.docs = docs
        tracker = DocumentTranslateTracker()

        if not self.translation_config.shared_context_cross_split_part.first_paragraph:
            # Try to find the first title paragraph
            title_paragraph = self.find_title_paragraph(docs)
            self.translation_config.shared_context_cross_split_part.first_paragraph = (
                copy.deepcopy(title_paragraph)
            )
            self.translation_config.shared_context_cross_split_part.recent_title_paragraph = copy.deepcopy(
                title_paragraph
            )
            if title_paragraph:
                logger.info(f"Found first title paragraph: {title_paragraph.unicode}")

        # count total paragraph
        total = sum(len(page.pdf_paragraph) for page in docs.page)
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            with PriorityThreadPoolExecutor(
                max_workers=self.translation_config.pool_max_workers,
            ) as executor:
                for page in docs.page:
                    self.process_page(page, executor, pbar, tracker.new_page())

        path = self.translation_config.get_working_file_path("translate_tracking.json")

        if (
            self.translation_config.debug
            or self.translation_config.working_dir is not None
        ):
            logger.debug(f"save translate tracking to {path}")
            with Path(path).open("w", encoding="utf-8") as f:
                f.write(tracker.to_json())

    def find_title_paragraph(self, docs: Document) -> PdfParagraph | None:
        """Find the first paragraph with layout_label 'title' in the document.

        Args:
            docs: The document to search in

        Returns:
            The first title paragraph found, or None if no title paragraph exists
        """
        for page in docs.page:
            for paragraph in page.pdf_paragraph:
                if paragraph.layout_label == "title":
                    logger.info(f"Found title paragraph: {paragraph.unicode}")
                    return paragraph
        return None

    def process_page(
        self,
        page: Page,
        executor: PriorityThreadPoolExecutor,
        pbar: tqdm | None = None,
        tracker: PageTranslateTracker = None,
    ):
        self.translation_config.raise_if_cancelled()
        page_no = (page.page_number + 1) if page.page_number is not None else None
        for para_index, paragraph in enumerate(page.pdf_paragraph, start=1):
            page_font_map = {}
            for font in page.pdf_font:
                page_font_map[font.font_id] = font
            page_xobj_font_map = {}
            for xobj in page.pdf_xobject:
                page_xobj_font_map[xobj.xobj_id] = page_font_map.copy()
                for font in xobj.pdf_font:
                    page_xobj_font_map[xobj.xobj_id][font.font_id] = font
            # self.translate_paragraph(paragraph, pbar,tracker.new_paragraph(), page_font_map, page_xobj_font_map)
            paragraph_token_count = self.calc_token_count(paragraph.unicode)
            if paragraph.layout_label == "title":
                self.shared_context_cross_split_part.recent_title_paragraph = (
                    copy.deepcopy(paragraph)
                )
            para_tracker = tracker.new_paragraph()
            # Extract bbox from paragraph.box
            bbox = None
            if paragraph.box is not None:
                bbox = {
                    "x": paragraph.box.x,
                    "y": paragraph.box.y,
                    "x2": paragraph.box.x2,
                    "y2": paragraph.box.y2,
                }
            para_tracker.record_layout_info(
                layout_label=paragraph.layout_label,
                page_no=page_no,
                para_index_in_page=para_index,
                bbox=bbox,
            )
            executor.submit(
                self.translate_paragraph,
                paragraph,
                page,
                pbar,
                para_tracker,
                page_font_map,
                page_xobj_font_map,
                priority=1048576 - paragraph_token_count,
                paragraph_token_count=paragraph_token_count,
                title_paragraph=self.translation_config.shared_context_cross_split_part.first_paragraph,
                local_title_paragraph=self.translation_config.shared_context_cross_split_part.recent_title_paragraph,
            )

    class TranslateInput:
        def __init__(
            self,
            unicode: str,
            placeholders: list[RichTextPlaceholder | FormulaPlaceholder],
            base_style: PdfStyle = None,
        ):
            self.unicode = unicode
            self.placeholders = placeholders
            self.base_style = base_style
            # Original placeholder-like tokens extracted from the source text.
            # Key: exact matched token string; Value: occurrence count.
            self.original_placeholder_tokens: dict[str, int] = {}

        def set_original_placeholder_tokens(self, tokens: dict[str, int] | None):
            """Attach original placeholder-like tokens from source text."""
            self.original_placeholder_tokens = tokens or {}

        def get_placeholders_hint(self) -> dict[str, str] | None:
            hint = {}
            for placeholder in self.placeholders:
                if isinstance(placeholder, FormulaPlaceholder):
                    cid_count = 0
                    for char in placeholder.formula.pdf_character:
                        if re.match(r"^\(cid:\d+\)$", char.char_unicode):
                            cid_count += 1
                    if cid_count > len(placeholder.formula.pdf_character) * 0.8:
                        continue

                    hint[placeholder.placeholder] = get_char_unicode_string(
                        placeholder.formula.pdf_character
                    )
            if hint:
                return hint
            return None

    def create_formula_placeholder(
        self,
        formula: PdfFormula,
        formula_id: int,
        paragraph: PdfParagraph,
    ):
        placeholder = self.translate_engine.get_formular_placeholder(formula_id)
        if isinstance(placeholder, tuple):
            placeholder, regex_pattern = placeholder
        else:
            regex_pattern = re.escape(placeholder)
        if re.match(regex_pattern, paragraph.unicode, re.IGNORECASE):
            return self.create_formula_placeholder(formula, formula_id + 1, paragraph)

        return FormulaPlaceholder(formula_id, formula, placeholder, regex_pattern)

    def create_rich_text_placeholder(
        self,
        composition: PdfSameStyleCharacters,
        composition_id: int,
        paragraph: PdfParagraph,
    ):
        left_placeholder = self.translate_engine.get_rich_text_left_placeholder(
            composition_id,
        )
        right_placeholder = self.translate_engine.get_rich_text_right_placeholder(
            composition_id,
        )
        if isinstance(left_placeholder, tuple):
            left_placeholder, left_placeholder_regex_pattern = left_placeholder
        else:
            left_placeholder_regex_pattern = re.escape(left_placeholder)
        if isinstance(right_placeholder, tuple):
            right_placeholder, right_placeholder_regex_pattern = right_placeholder
        else:
            right_placeholder_regex_pattern = re.escape(right_placeholder)
        if re.match(
            f"{left_placeholder_regex_pattern}|{right_placeholder_regex_pattern}",
            paragraph.unicode,
            re.IGNORECASE,
        ):
            return self.create_rich_text_placeholder(
                composition,
                composition_id + 1,
                paragraph,
            )

        return RichTextPlaceholder(
            composition_id,
            composition,
            left_placeholder,
            right_placeholder,
            left_placeholder_regex_pattern,
            right_placeholder_regex_pattern,
        )

    def get_translate_input(
        self,
        paragraph: PdfParagraph,
        page_font_map: dict[str, PdfFont] = None,
        disable_rich_text_translate: bool | None = None,
    ):
        if not paragraph.pdf_paragraph_composition:
            return

        # Skip pure numeric paragraphs
        if is_pure_numeric_paragraph(paragraph):
            return None

        # Skip paragraphs with only placeholders
        if is_placeholder_only_paragraph(paragraph):
            return None

        # Extract original placeholder-like tokens from the raw paragraph text
        original_placeholder_tokens: dict[str, int] = {}

        def scan_placeholder_tokens(text: str, tokens: dict[str, int]):
            for pattern in (
                self._formula_placeholder_pattern,
                self._style_left_placeholder_pattern,
                self._style_right_placeholder_pattern,
            ):
                for match in pattern.finditer(text):
                    token = match.group(0)
                    tokens[token] = tokens.get(token, 0) + 1

        if paragraph.unicode:
            scan_placeholder_tokens(paragraph.unicode, original_placeholder_tokens)
        if len(paragraph.pdf_paragraph_composition) == 1:
            # 如果整个段落只有一个组成部分，那么直接返回，不需要套占位符等
            composition = paragraph.pdf_paragraph_composition[0]
            if (
                composition.pdf_line
                or composition.pdf_same_style_characters
                or composition.pdf_character
            ):
                translate_input = self.TranslateInput(
                    paragraph.unicode,
                    [],
                    paragraph.pdf_style,
                )
                translate_input.set_original_placeholder_tokens(
                    original_placeholder_tokens,
                )
                return translate_input
            elif composition.pdf_formula:
                # 不需要翻译纯公式
                return None
            elif composition.pdf_same_style_unicode_characters:
                # DEBUG INSERT CHAR, NOT TRANSLATE
                return None
            else:
                logger.error(
                    f"Unknown composition type. "
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
                return None

        # 如果没有指定 disable_rich_text_translate，使用配置中的值
        if disable_rich_text_translate is None:
            disable_rich_text_translate = (
                self.translation_config.disable_rich_text_translate
            )

        placeholder_id = 1
        placeholders = []
        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_line:
                chars.extend(composition.pdf_line.pdf_character)
            elif composition.pdf_formula:
                formula_placeholder = self.create_formula_placeholder(
                    composition.pdf_formula,
                    placeholder_id,
                    paragraph,
                )
                placeholders.append(formula_placeholder)
                # 公式只需要一个占位符，所以 id+1
                placeholder_id = formula_placeholder.id + 1
                chars.extend(formula_placeholder.placeholder)
            elif composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_same_style_characters:
                if disable_rich_text_translate:
                    # 如果禁用富文本翻译，直接添加字符
                    chars.extend(composition.pdf_same_style_characters.pdf_character)
                    continue

                fonta = self.font_mapper.map(
                    page_font_map[
                        composition.pdf_same_style_characters.pdf_style.font_id
                    ],
                    "1",
                )
                fontb = self.font_mapper.map(
                    page_font_map[paragraph.pdf_style.font_id],
                    "1",
                )
                if (
                    # 样式和段落基准样式一致，无需占位符
                    is_same_style(
                        composition.pdf_same_style_characters.pdf_style,
                        paragraph.pdf_style,
                    )
                    # 字号差异在 0.7-1.3 之间，可能是首字母变大效果，无需占位符
                    or is_same_style_except_size(
                        composition.pdf_same_style_characters.pdf_style,
                        paragraph.pdf_style,
                    )
                    or (
                        # 除了字体以外样式都和基准一样，并且字体都映射到同一个字体。无需占位符
                        is_same_style_except_font(
                            composition.pdf_same_style_characters.pdf_style,
                            paragraph.pdf_style,
                        )
                        and fonta
                        and fontb
                        and fonta.font_id == fontb.font_id
                    )
                    # or len(composition.pdf_same_style_characters.pdf_character) == 1
                ):
                    chars.extend(composition.pdf_same_style_characters.pdf_character)
                    continue
                placeholder = self.create_rich_text_placeholder(
                    composition.pdf_same_style_characters,
                    placeholder_id,
                    paragraph,
                )
                placeholders.append(placeholder)
                # 样式需要一左一右两个占位符，所以 id+2
                placeholder_id = placeholder.id + 2
                chars.append(placeholder.left_placeholder)
                chars.extend(composition.pdf_same_style_characters.pdf_character)
                chars.append(placeholder.right_placeholder)
            else:
                logger.error(
                    "Unexpected PdfParagraphComposition type "
                    "in PdfParagraph during translation. "
                    f"Composition: {composition}. "
                    f"Paragraph: {paragraph}. ",
                )
                return None

            # 如果占位符数量超过阈值，且未禁用富文本翻译，则递归调用并禁用富文本翻译
            if len(placeholders) > 40 and not disable_rich_text_translate:
                logger.warning(
                    f"Too many placeholders ({len(placeholders)}) in paragraph[{paragraph.debug_id}], "
                    "disabling rich text translation for this paragraph",
                )
                return self.get_translate_input(paragraph, page_font_map, True)

        text = get_char_unicode_string(chars)
        translate_input = self.TranslateInput(text, placeholders, paragraph.pdf_style)
        translate_input.set_original_placeholder_tokens(original_placeholder_tokens)
        return translate_input

    def process_formula(
        self,
        formula: PdfFormula,
        formula_id: int,
        paragraph: PdfParagraph,
    ):
        placeholder = self.create_formula_placeholder(formula, formula_id, paragraph)
        if placeholder.placeholder in paragraph.unicode:
            return self.process_formula(formula, formula_id + 1, paragraph)

        return placeholder

    def process_composition(
        self,
        composition: PdfSameStyleCharacters,
        composition_id: int,
        paragraph: PdfParagraph,
    ):
        placeholder = self.create_rich_text_placeholder(
            composition,
            composition_id,
            paragraph,
        )
        if (
            placeholder.left_placeholder in paragraph.unicode
            or placeholder.right_placeholder in paragraph.unicode
        ):
            return self.process_composition(
                composition,
                composition_id + 1,
                paragraph,
            )

        return placeholder

    def parse_translate_output(
        self,
        input_text: TranslateInput,
        output: str,
        tracker: ParagraphTranslateTracker | None = None,
        llm_translate_tracker: LLMTranslateTracker | None = None,
    ) -> [PdfParagraphComposition]:
        result = []

        # 如果没有占位符，直接返回整个文本
        if not input_text.placeholders:
            comp = PdfParagraphComposition()
            comp.pdf_same_style_unicode_characters = PdfSameStyleUnicodeCharacters()
            comp.pdf_same_style_unicode_characters.unicode = output
            comp.pdf_same_style_unicode_characters.pdf_style = input_text.base_style
            if llm_translate_tracker:
                llm_translate_tracker.set_placeholder_full_match()
            return [comp]

        # 构建正则表达式模式
        patterns = []
        placeholder_patterns = []
        placeholder_map = {}

        for placeholder in input_text.placeholders:
            if isinstance(placeholder, FormulaPlaceholder):
                # 转义特殊字符
                # pattern = re.escape(placeholder.placeholder)
                pattern = placeholder.regex_pattern
                patterns.append(f"({pattern})")
                placeholder_patterns.append(f"({pattern})")
                placeholder_map[placeholder.placeholder] = placeholder
            else:
                left = placeholder.left_regex_pattern
                right = placeholder.right_regex_pattern
                patterns.append(f"({left}.*?{right})")
                placeholder_patterns.append(f"({left})")
                placeholder_patterns.append(f"({right})")
                placeholder_map[placeholder.left_placeholder] = placeholder
        all_match = True
        for pattern in patterns:
            if not re.search(pattern, output, flags=re.IGNORECASE):
                all_match = False
                break
        if all_match:
            if llm_translate_tracker:
                llm_translate_tracker.set_placeholder_full_match()
        else:
            logger.debug(f"Failed to match all placeholder for {input_text.unicode}")
        # 合并所有模式
        combined_pattern = "|".join(patterns)
        combined_placeholder_pattern = "|".join(placeholder_patterns)
        # Build allowed placeholder tokens: originals from source + placeholders we injected.
        allowed_placeholder_tokens: set[str] = set()
        if getattr(input_text, "original_placeholder_tokens", None):
            allowed_placeholder_tokens.update(input_text.original_placeholder_tokens)
        for placeholder in input_text.placeholders:
            if isinstance(placeholder, FormulaPlaceholder):
                allowed_placeholder_tokens.add(placeholder.placeholder)
            else:
                allowed_placeholder_tokens.add(placeholder.left_placeholder)
                allowed_placeholder_tokens.add(placeholder.right_placeholder)

        def remove_placeholder(text: str):
            """Remove placeholder artifacts and hallucinated placeholder-like tokens."""
            # First, remove any leftover placeholders built from our own regex patterns.
            if combined_placeholder_pattern:
                text = re.sub(
                    combined_placeholder_pattern,
                    "",
                    text,
                    flags=re.IGNORECASE,
                )

            # Then, detect placeholder-like tokens of the same shapes as our own
            # formula and rich-text placeholders. Only keep those in the allowed set.
            def _replace_token(match: re.Match) -> str:
                token = match.group(0)
                if token in allowed_placeholder_tokens:
                    return token
                if tracker is not None:
                    tracker.record_removed_hallucinated_placeholder(token)
                return ""

            text = self._formula_placeholder_pattern.sub(_replace_token, text)
            text = self._style_left_placeholder_pattern.sub(_replace_token, text)
            text = self._style_right_placeholder_pattern.sub(_replace_token, text)
            return text

        # 找到所有匹配
        last_end = 0
        for match in re.finditer(combined_pattern, output, flags=re.IGNORECASE):
            # 处理匹配之前的普通文本
            if match.start() > last_end:
                text = output[last_end : match.start()]
                if text:
                    comp = PdfParagraphComposition()
                    comp.pdf_same_style_unicode_characters = (
                        PdfSameStyleUnicodeCharacters()
                    )
                    comp.pdf_same_style_unicode_characters.unicode = remove_placeholder(
                        text,
                    )
                    comp.pdf_same_style_unicode_characters.pdf_style = (
                        input_text.base_style
                    )
                    result.append(comp)

            matched_text = match.group(0)

            # 处理占位符
            if any(
                isinstance(p, FormulaPlaceholder)
                and re.match(f"^{p.regex_pattern}$", matched_text, re.IGNORECASE)
                for p in input_text.placeholders
            ):
                # 处理公式占位符
                placeholder = next(
                    p
                    for p in input_text.placeholders
                    if isinstance(p, FormulaPlaceholder)
                    and re.match(f"^{p.regex_pattern}$", matched_text, re.IGNORECASE)
                )
                comp = PdfParagraphComposition()
                comp.pdf_formula = placeholder.formula
                result.append(comp)
            else:
                # 处理富文本占位符
                placeholder = next(
                    p
                    for p in input_text.placeholders
                    if not isinstance(p, FormulaPlaceholder)
                    and re.match(
                        f"^{p.left_regex_pattern}", matched_text, re.IGNORECASE
                    )
                )
                text = re.match(
                    f"^{placeholder.left_regex_pattern}(.*){placeholder.right_regex_pattern}$",
                    matched_text,
                    re.IGNORECASE,
                ).group(1)

                if isinstance(
                    placeholder.composition,
                    PdfSameStyleCharacters,
                ) and text.replace(" ", "") == "".join(
                    x.char_unicode for x in placeholder.composition.pdf_character
                ).replace(
                    " ",
                    "",
                ):
                    comp = PdfParagraphComposition(
                        pdf_same_style_characters=placeholder.composition,
                    )
                else:
                    comp = PdfParagraphComposition()
                    comp.pdf_same_style_unicode_characters = (
                        PdfSameStyleUnicodeCharacters()
                    )
                    comp.pdf_same_style_unicode_characters.pdf_style = (
                        placeholder.composition.pdf_style
                    )
                    comp.pdf_same_style_unicode_characters.unicode = remove_placeholder(
                        text,
                    )
                result.append(comp)

            last_end = match.end()

        # 处理最后的普通文本
        if last_end < len(output):
            text = output[last_end:]
            if text:
                comp = PdfParagraphComposition()
                comp.pdf_same_style_unicode_characters = PdfSameStyleUnicodeCharacters()
                comp.pdf_same_style_unicode_characters.unicode = remove_placeholder(
                    text,
                )
                comp.pdf_same_style_unicode_characters.pdf_style = input_text.base_style
                result.append(comp)

        return result

    def pre_translate_paragraph(
        self,
        paragraph: PdfParagraph,
        tracker: ParagraphTranslateTracker,
        page_font_map: dict[str, PdfFont],
        xobj_font_map: dict[int, dict[str, PdfFont]],
    ):
        """Pre-translation processing: prepare text for translation."""
        if paragraph.vertical:
            return None, None
        tracker.set_pdf_unicode(paragraph.unicode)
        # Calculate and record token count
        token_count = self.calc_token_count(paragraph.unicode)
        tracker.set_token_count(token_count)
        if paragraph.xobj_id in xobj_font_map:
            page_font_map = xobj_font_map[paragraph.xobj_id]
        disable_rich_text_translate = (
            self.translation_config.disable_rich_text_translate
        )
        if not self.support_llm_translate:
            disable_rich_text_translate = True

        translate_input = self.get_translate_input(
            paragraph, page_font_map, disable_rich_text_translate
        )
        if not translate_input:
            return None, None
        tracker.set_input(translate_input.unicode)
        tracker.set_placeholders(translate_input.placeholders)
        tracker.set_original_placeholders(
            getattr(translate_input, "original_placeholder_tokens", None),
        )
        text = translate_input.unicode
        if len(text) < self.translation_config.min_text_length:
            logger.debug(
                f"Text too short to translate, skip. Text: {text}. Paragraph id: {paragraph.debug_id}."
            )
            return None, None
        return text, translate_input

    def post_translate_paragraph(
        self,
        paragraph: PdfParagraph,
        tracker: ParagraphTranslateTracker,
        translate_input,
        translated_text: str,
    ):
        """Post-translation processing: update paragraph with translated text."""
        tracker.set_output(translated_text)
        if translated_text == translate_input:
            if llm_translate_tracker := tracker.last_llm_translate_tracker():
                llm_translate_tracker.set_placeholder_full_match()
            return False
        paragraph.unicode = translated_text
        paragraph.pdf_paragraph_composition = self.parse_translate_output(
            translate_input,
            translated_text,
            tracker,
            tracker.last_llm_translate_tracker(),
        )
        for composition in paragraph.pdf_paragraph_composition:
            if (
                composition.pdf_same_style_unicode_characters
                and composition.pdf_same_style_unicode_characters.pdf_style is None
            ):
                composition.pdf_same_style_unicode_characters.pdf_style = (
                    paragraph.pdf_style
                )
        return True

    def _build_role_block(self) -> str:
        """Build the role block for LLM prompt.

        Returns:
            Role block string with custom_system_prompt or default role description.
        """
        custom_prompt = getattr(self.translation_config, "custom_system_prompt", None)
        if custom_prompt:
            role_block = custom_prompt.strip()
            if "Follow all rules strictly." not in role_block:
                if not role_block.endswith("\n"):
                    role_block += "\n"
                role_block += "Follow all rules strictly."
        else:
            role_block = (
                f"You are a professional {self.translation_config.lang_out} native translator who needs to fluently translate text "
                f"into {self.translation_config.lang_out}.\n\n"
                "Follow all rules strictly."
            )
        return role_block

    def _build_context_block(
        self,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
        translate_input: TranslateInput | None = None,
    ) -> str:
        """Build the context/hints block for LLM prompt.

        Args:
            title_paragraph: First title paragraph in the document
            local_title_paragraph: Most recent title paragraph
            translate_input: TranslateInput containing placeholder hints

        Returns:
            Context block string, empty if no context hints available
        """
        context_lines: list[str] = []
        hint_idx = 1

        if title_paragraph:
            context_lines.append(
                f"{hint_idx}. First title in the full text: {title_paragraph.unicode}"
            )
            hint_idx += 1

        if local_title_paragraph:
            is_different_from_global = True
            if title_paragraph:
                if local_title_paragraph.debug_id == title_paragraph.debug_id:
                    is_different_from_global = False

            if is_different_from_global:
                context_lines.append(
                    f"{hint_idx}. The most recent title is: {local_title_paragraph.unicode}"
                )
                hint_idx += 1

        if translate_input and self.translation_config.add_formula_placehold_hint:
            placeholders_hint = translate_input.get_placeholders_hint()
            if placeholders_hint:
                context_lines.append(
                    f"{hint_idx}. Formula placeholder hint:\n{placeholders_hint}"
                )

        if context_lines:
            return "## Context / Hints\n" + "\n".join(context_lines) + "\n"
        return ""

    def _build_glossary_block(self, text: str) -> str:
        """Build the glossary block for LLM prompt.

        Args:
            text: Text to match against glossary entries

        Returns:
            Glossary block string with tables, empty if no active glossary entries
        """
        if not self._cached_glossaries:
            return ""

        glossary_entries_per_glossary: dict[str, list[tuple[str, str]]] = {}

        for glossary in self._cached_glossaries:
            active_entries = glossary.get_active_entries_for_text(text)
            if active_entries:
                glossary_entries_per_glossary[glossary.name] = sorted(active_entries)

        if not glossary_entries_per_glossary:
            return ""

        glossary_block_lines: list[str] = [
            "## Glossary",
            "",
            "Always use the glossary's **Target Term** for any occurrence of its **Source Term** "
            "(including variants, inside tags, or broken across lines).",
            "",
            "Unlisted terms are translated naturally.",
            "",
        ]

        for glossary_name, entries in glossary_entries_per_glossary.items():
            glossary_block_lines.append(f"### Glossary: {glossary_name}")
            glossary_block_lines.append("")
            glossary_block_lines.append(
                "| Source Term | Target Term |\n|-------------|-------------|"
            )
            for original_source, target_text in entries:
                glossary_block_lines.append(f"| {original_source} | {target_text} |")
            glossary_block_lines.append("")

        return "\n".join(glossary_block_lines)

    def generate_prompt_for_llm(
        self,
        text: str,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
        translate_input: TranslateInput | None = None,
    ):
        """Generate LLM prompt using template-based approach.

        Args:
            text: Text to be translated
            title_paragraph: First title paragraph in the document
            local_title_paragraph: Most recent title paragraph
            translate_input: TranslateInput containing placeholder information

        Returns:
            Final LLM prompt string
        """
        role_block = self._build_role_block()
        context_block = self._build_context_block(
            title_paragraph, local_title_paragraph, translate_input
        )
        glossary_block = self._build_glossary_block(text)

        return PROMPT_TEMPLATE.substitute(
            role_block=role_block,
            glossary_block=glossary_block,
            context_block=context_block,
            lang_out=self.translation_config.lang_out,
            text_to_translate=text,
        )

    def add_content_filter_hint(self, page: Page, paragraph: PdfParagraph):
        with self.add_content_filter_hint_lock:
            new_box = il_version_1.Box(
                x=paragraph.box.x,
                y=paragraph.box.y2,
                x2=paragraph.box.x2,
                y2=paragraph.box.y2 + 1.1,
            )
            page.pdf_paragraph.append(
                self._create_text(
                    "翻译服务检测到内容可能包含不安全或敏感内容，请您避免翻译敏感内容，感谢您的配合。",
                    GRAY80,
                    new_box,
                    1,
                )
            )
            logger.info("success add content filter hint")

    def _create_text(
        self,
        text: str,
        color: GraphicState,
        box: il_version_1.Box,
        font_size: float = 4,
    ):
        style = il_version_1.PdfStyle(
            font_id="base",
            font_size=font_size,
            graphic_state=color,
        )
        return il_version_1.PdfParagraph(
            first_line_indent=False,
            box=box,
            vertical=False,
            pdf_style=style,
            unicode=text,
            pdf_paragraph_composition=[
                il_version_1.PdfParagraphComposition(
                    pdf_same_style_unicode_characters=il_version_1.PdfSameStyleUnicodeCharacters(
                        unicode=text,
                        pdf_style=style,
                        debug_info=True,
                    ),
                ),
            ],
            xobj_id=-1,
        )

    def translate_paragraph(
        self,
        paragraph: PdfParagraph,
        page: Page,
        pbar: tqdm | None = None,
        tracker: ParagraphTranslateTracker = None,
        page_font_map: dict[str, PdfFont] = None,
        xobj_font_map: dict[int, dict[str, PdfFont]] = None,
        paragraph_token_count: int = 0,
        title_paragraph: PdfParagraph | None = None,
        local_title_paragraph: PdfParagraph | None = None,
    ):
        """Translate a paragraph using pre and post processing functions."""
        self.translation_config.raise_if_cancelled()
        with PbarContext(pbar):
            try:
                if self.use_as_fallback:
                    # il translator llm only modifies unicode in some situations
                    paragraph.unicode = get_paragraph_unicode(paragraph)
                # Pre-translation processing
                text, translate_input = self.pre_translate_paragraph(
                    paragraph, tracker, page_font_map, xobj_font_map
                )
                if text is None:
                    return
                llm_translate_tracker = tracker.new_llm_translate_tracker()
                # Perform translation
                if self.support_llm_translate:
                    llm_prompt = self.generate_prompt_for_llm(
                        text,
                        title_paragraph,
                        local_title_paragraph,
                        translate_input,
                    )
                    llm_translate_tracker.set_input(llm_prompt)
                    translated_text = self.translate_engine.llm_translate(
                        llm_prompt,
                        rate_limit_params={
                            "paragraph_token_count": paragraph_token_count
                        },
                    )
                    llm_translate_tracker.set_output(translated_text)
                else:
                    translated_text = self.translate_engine.translate(
                        text,
                        rate_limit_params={
                            "paragraph_token_count": paragraph_token_count
                        },
                    )
                translated_text = re.sub(r"[. 。…，]{20,}", ".", translated_text)

                # Post-translation processing
                self.post_translate_paragraph(
                    paragraph, tracker, translate_input, translated_text
                )
            except ContentFilterError as e:
                logger.warning(f"ContentFilterError: {e.message}")
                self.add_content_filter_hint(page, paragraph)
                return
            except Exception as e:
                logger.exception(
                    f"Error translating paragraph. Paragraph: {paragraph.debug_id} ({paragraph.unicode}). Error: {e}. ",
                )
                # ignore error and continue
                return
