import logging
import random
import re
import statistics
import unicodedata

import numpy as np

from babeldoc.babeldoc_exception.BabelDOCException import ExtractTextError
from babeldoc.format.pdf.document_il import Box
from babeldoc.format.pdf.document_il import Document
from babeldoc.format.pdf.document_il import Page
from babeldoc.format.pdf.document_il import PdfCharacter
from babeldoc.format.pdf.document_il import PdfLine
from babeldoc.format.pdf.document_il import PdfParagraph
from babeldoc.format.pdf.document_il import PdfParagraphComposition
from babeldoc.format.pdf.document_il import PdfRectangle
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.formular_helper import (
    collect_page_formula_font_ids,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    HEIGHT_NOT_USFUL_CHAR_IN_CHAR,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import SPACE_REGEX
from babeldoc.format.pdf.document_il.utils.layout_helper import Layout
from babeldoc.format.pdf.document_il.utils.layout_helper import add_space_dummy_chars
from babeldoc.format.pdf.document_il.utils.layout_helper import build_layout_index
from babeldoc.format.pdf.document_il.utils.layout_helper import calculate_iou_for_boxes
from babeldoc.format.pdf.document_il.utils.layout_helper import get_char_unicode_string
from babeldoc.format.pdf.document_il.utils.layout_helper import get_character_layout
from babeldoc.format.pdf.document_il.utils.layout_helper import is_bullet_point
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    is_character_in_formula_layout,
)
from babeldoc.format.pdf.document_il.utils.layout_helper import is_text_layout
from babeldoc.format.pdf.document_il.utils.paragraph_helper import is_cid_paragraph
from babeldoc.format.pdf.document_il.utils.style_helper import INDIGO
from babeldoc.format.pdf.document_il.utils.style_helper import WHITE
from babeldoc.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)

# Base58 alphabet (Bitcoin style, without numbers 0, O, I, l)
BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# ---- Table of Contents (TOC) line detection ----
# 连续点引导符（兼容旧的目录识别规则）
TOC_DOT_LEADER_REGEX = re.compile(r"\.{20,}")
# 带间隔的点引导符中点的最少数量（如 ". . . . ."）
TOC_SPACED_DOT_MIN_COUNT = 4
# 一个多行段落被判定为目录所需的最少条目行数
TOC_MIN_ENTRY_LINES = 3
# 条目行数占全部行的最低比例
TOC_MIN_ENTRY_RATIO = 0.6
# 各条目尾部页码右边缘对齐容差（pt）
TOC_NUMBER_RIGHT_EDGE_TOLERANCE = 8.0
# 标题与页码之间水平空隙的最小绝对距离（pt）
TOC_MIN_GAP = 10.0
# 标题与页码之间空隙相对行内空格宽度的倍数
TOC_GAP_SPACE_WIDTH_FACTOR = 4.5
# 阿拉伯页码最大位数
TOC_MAX_PAGE_DIGITS = 4
# 罗马页码（如 iii、xiv）
TOC_ROMAN_NUMERAL_REGEX = re.compile(
    r"^m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)
TOC_MAX_ROMAN_LEN = 6
# 引导点/省略号字符（翻译输出中可能出现的点线字符）
TOC_LEADER_CHARS = (".", "．", "…", "·", "‧", "⋅")
# 同一个 PdfLine 内多个视觉行的最小纵向间距（目录行距通常 >12pt）
TOC_VISUAL_BAND_MIN_GAP = 6.0


def is_toc_separator_char(ch: str | None) -> bool:
    """空格、零宽字符等不参与标题/页码内容判定的分隔字符。"""
    if not ch:
        return True
    if ch.isspace():
        return True
    return unicodedata.category(ch) == "Cf"


def is_toc_leader_char(ch: str | None) -> bool:
    """点引导符字符（各种形态的句点/中点/省略号）。"""
    return bool(ch) and ch in TOC_LEADER_CHARS


def is_toc_page_number_token(text: str) -> bool:
    """判断文本是否为目录页码（阿拉伯数字或罗马数字）。"""
    if not text:
        return False
    if text.isdigit():
        return len(text) <= TOC_MAX_PAGE_DIGITS
    if len(text) <= TOC_MAX_ROMAN_LEN:
        return bool(TOC_ROMAN_NUMERAL_REGEX.match(text))
    return False


def generate_base58_id(length: int = 5) -> str:
    """Generate a random base58 ID of specified length."""
    return "".join(random.choice(BASE58_ALPHABET) for _ in range(length))


class ParagraphFinder:
    stage_name = "Parse Paragraphs"

    # 定义项目符号的正则表达式模式

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        self.font_mapper = FontMapper(translation_config)

    def _preprocess_formula_layouts(self, page: Page):
        """
        Identifies 'formula' layouts that do not significantly overlap with any text layouts
        and re-labels them as 'isolate_formula'.
        """
        # Use a simplified Layout object for is_text_layout check
        text_layouts = [
            layout
            for layout in page.page_layout
            if is_text_layout(Layout(layout.id, layout.class_name))
        ]
        formula_layouts = [
            layout for layout in page.page_layout if layout.class_name == "formula"
        ]

        if not text_layouts or not formula_layouts:
            return

        for formula_layout in formula_layouts:
            is_isolated = True
            for text_layout in text_layouts:
                iou = calculate_iou_for_boxes(formula_layout.box, text_layout.box)
                if iou >= 0.5:
                    is_isolated = False
                    break

            if is_isolated:
                formula_layout.class_name = "isolate_formula"

    def add_text_fill_background(self, page: Page):
        layout_map = {layout.id: layout for layout in page.page_layout}
        for paragraph in page.pdf_paragraph:
            layout_id = paragraph.layout_id
            if layout_id is None:
                continue
            layout = layout_map[layout_id]
            if paragraph.box is None:
                continue
            x1, y1, x2, y2 = (
                paragraph.box.x,
                paragraph.box.y,
                paragraph.box.x2,
                paragraph.box.y2,
            )
            layout_box = layout.box
            if layout_box.x < x1:
                x1 = layout_box.x
            if layout_box.y < y1:
                y1 = layout_box.y
            if layout_box.x2 > x2:
                x2 = layout_box.x2
            if layout_box.y2 > y2:
                y2 = layout_box.y2
            assert x2 > x1 and y2 > y1
            page.pdf_rectangle.append(
                PdfRectangle(
                    box=Box(x1, y1, x2, y2),
                    fill_background=True,
                    graphic_state=WHITE,
                    debug_info=False,
                    xobj_id=paragraph.xobj_id,
                )
            )

    def update_paragraph_data(self, paragraph: PdfParagraph, update_unicode=False):
        if not paragraph.pdf_paragraph_composition:
            return

        chars = []
        for composition in paragraph.pdf_paragraph_composition:
            if composition.pdf_line:
                chars.extend(composition.pdf_line.pdf_character)
            elif composition.pdf_formula:
                chars.extend(composition.pdf_formula.pdf_character)
            elif composition.pdf_character:
                chars.append(composition.pdf_character)
            elif composition.pdf_same_style_unicode_characters:
                continue
            else:
                logger.error(
                    "Unexpected composition type"
                    " in PdfParagraphComposition. "
                    "This type only appears in the IL "
                    "after the translation is completed.",
                )
                continue

        if update_unicode and chars:
            paragraph.unicode = get_char_unicode_string(chars)
        if not chars:
            return
        # 更新边界框
        min_x = min(char.visual_bbox.box.x for char in chars)
        min_y = min(char.visual_bbox.box.y for char in chars)
        max_x = max(char.visual_bbox.box.x2 for char in chars)
        max_y = max(char.visual_bbox.box.y2 for char in chars)
        paragraph.box = Box(min_x, min_y, max_x, max_y)
        paragraph.vertical = chars[0].vertical
        paragraph.xobj_id = chars[0].xobj_id

        paragraph.first_line_indent = False
        if (
            paragraph.pdf_paragraph_composition
            and paragraph.pdf_paragraph_composition[0].pdf_line
            and paragraph.pdf_paragraph_composition[0]
            .pdf_line.pdf_character[0]
            .visual_bbox.box.x
            - paragraph.box.x
            > 1
        ):
            paragraph.first_line_indent = True

    def update_line_data(self, line: PdfLine):
        min_x = min(char.visual_bbox.box.x for char in line.pdf_character)
        min_y = min(char.visual_bbox.box.y for char in line.pdf_character)
        max_x = max(char.visual_bbox.box.x2 for char in line.pdf_character)
        max_y = max(char.visual_bbox.box.y2 for char in line.pdf_character)
        line.box = Box(min_x, min_y, max_x, max_y)

    def add_debug_info(self, page: Page):
        if not self.translation_config.debug:
            return
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    page.pdf_rectangle.append(
                        PdfRectangle(
                            box=line.box,
                            fill_background=False,
                            graphic_state=INDIGO,
                            debug_info=True,
                            line_width=0.2,
                        )
                    )

    def process(self, document):
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            len(document.page),
        ) as pbar:
            if not document.page:
                return
            for page in document.page:
                self.translation_config.raise_if_cancelled()
                self.process_page(page)
                pbar.advance()

            total_paragraph_count = 0
            for page in document.page:
                total_paragraph_count += len(page.pdf_paragraph)
            if total_paragraph_count == 0:
                raise ExtractTextError("The document contains no paragraphs.")

            if self.check_cid_paragraph(document):
                raise ExtractTextError("The document contains too many CID paragraphs.")

    def check_cid_paragraph(self, doc: Document):
        cid_para_count = 0
        para_total = 0
        for page in doc.page:
            para_total += len(page.pdf_paragraph)
            for para in page.pdf_paragraph:
                if is_cid_paragraph(para):
                    cid_para_count += 1
        return cid_para_count / para_total > 0.8

    def bbox_overlap(self, bbox1: Box, bbox2: Box) -> bool:
        return (
            bbox1.x < bbox2.x2
            and bbox1.x2 > bbox2.x
            and bbox1.y < bbox2.y2
            and bbox1.y2 > bbox2.y
        )

    def process_page(self, page: Page):
        layout_index, layout_map = build_layout_index(page)
        # 预处理公式布局的标签
        self._preprocess_formula_layouts(page)

        # 第一步：根据 layout 创建 paragraphs
        # 在这一步中，page.pdf_character 中的字符会被移除
        paragraphs = self._group_characters_into_paragraphs(
            page, layout_index, layout_map
        )
        page.pdf_paragraph = paragraphs

        page_level_formula_font_ids, xobj_specific_formula_font_ids = (
            collect_page_formula_font_ids(
                page, self.translation_config.formular_font_pattern
            )
        )

        # for para in paragraphs:
        #     if not para.debug_id:
        #         continue
        #     new_line = PdfLine(
        #         pdf_character=[x.pdf_character for x in para.pdf_paragraph_composition]
        #     )
        #     self.update_line_data(new_line)
        #     para.pdf_paragraph_composition = [
        #         PdfParagraphComposition(pdf_line=new_line)
        #     ]

        # 第二步：将段落内的字符拆分为行
        for paragraph in paragraphs:
            if (
                paragraph.xobj_id
                and paragraph.xobj_id in xobj_specific_formula_font_ids
            ):
                current_formula_font_ids = xobj_specific_formula_font_ids[
                    paragraph.xobj_id
                ]
            else:
                current_formula_font_ids = page_level_formula_font_ids
            self._split_paragraph_into_lines(paragraph, current_formula_font_ids)

        # 第三步：处理段落中的空格
        for paragraph in paragraphs:
            add_space_dummy_chars(paragraph)
            self.process_paragraph_spacing(paragraph)
            self.update_paragraph_data(paragraph)

        # 第四步：计算所有行宽度的中位数
        median_width = self.calculate_median_line_width(paragraphs)

        # 第五步：处理独立段落
        self.process_independent_paragraphs(paragraphs, median_width)

        # 新增后处理：合并带行号交替的正文段落（a 正文、b 行号、c 正文 -> 合并 a 与 c，保留 b）
        if getattr(self.translation_config, "merge_alternating_line_numbers", True):
            self.merge_alternating_line_number_paragraphs(paragraphs)

        # 新增后处理：将目录（TOC）段落按条目拆分为独立段落，
        # 避免所有目录条目被合并成一个连续大段而丢失换行与缩进
        if getattr(self.translation_config, "split_toc_lines", True):
            self.split_toc_paragraphs(paragraphs)

        for paragraph in paragraphs:
            self.update_paragraph_data(paragraph, update_unicode=True)

        if self.translation_config.ocr_workaround:
            self.add_text_fill_background(page)
            # since this is ocr file,
            # image characters are not needed
            page.pdf_character = []

        self.fix_overlapping_paragraphs(page)

        # 第六步：对每一行的字符进行排序
        # self._sort_characters_in_lines(page)

        self.add_debug_info(page)

        # 新阶段：设置段落的 renderorder 为所有组成部分中 renderorder 最小的
        self._set_paragraph_render_order(page)

    def _set_paragraph_render_order(self, page: Page):
        """
        设置段落的 renderorder 为段落所有组成部分中 renderorder 最小的值
        """
        for paragraph in page.pdf_paragraph:
            min_render_order = 9999999999999999

            # 遍历段落的所有组成部分
            for composition in paragraph.pdf_paragraph_composition:
                # 检查 PdfLine 中的字符
                if composition.pdf_line:
                    for char in composition.pdf_line.pdf_character:
                        if (
                            hasattr(char, "render_order")
                            and char.render_order is not None
                        ):
                            min_render_order = min(min_render_order, char.render_order)

                # 检查单个字符
                elif composition.pdf_character:
                    char = composition.pdf_character
                    if hasattr(char, "render_order") and char.render_order is not None:
                        min_render_order = min(min_render_order, char.render_order)

                # 检查公式中的字符
                elif composition.pdf_formula:
                    for char in composition.pdf_formula.pdf_character:
                        if (
                            hasattr(char, "render_order")
                            and char.render_order is not None
                        ):
                            min_render_order = min(min_render_order, char.render_order)

            # 如果找到了有效的 renderorder，设置段落的 renderorder
            if min_render_order != 9999999999999999:
                paragraph.render_order = min_render_order

    def is_isolated_formula(self, char: PdfCharacter):
        return char.char_unicode in (
            "(cid:122)",
            "(cid:123)",
            "(cid:124)",
            "(cid:125)",
        )

    def _paragraph_text_ascii(self, p: PdfParagraph) -> str:
        parts: list[str] = []
        for comp in p.pdf_paragraph_composition or []:
            if comp.pdf_line:
                for ch in comp.pdf_line.pdf_character or []:
                    if ch.char_unicode is not None:
                        parts.append(ch.char_unicode)
            elif comp.pdf_character and comp.pdf_character.char_unicode is not None:
                parts.append(comp.pdf_character.char_unicode)
        return "".join(parts)

    def _is_ascii_digit_or_space_paragraph(self, p: PdfParagraph) -> bool:
        text = self._paragraph_text_ascii(p)
        if not text:
            return True
        has_digit = False
        for c in text:
            if c.isdigit() and ord(c) < 128:
                has_digit = True
                continue
            if c.isspace():
                continue
            return False
        return True if has_digit or text.strip() == "" else False

    @staticmethod
    def _same_layout_and_xobj(a: PdfParagraph, c: PdfParagraph) -> bool:
        return (
            a.layout_id is not None
            and c.layout_id is not None
            and a.layout_id == c.layout_id
            and a.xobj_id is not None
            and c.xobj_id is not None
            and a.xobj_id == c.xobj_id
        )

    def merge_alternating_line_number_paragraphs(self, paragraphs: list[PdfParagraph]):
        # a 代表正文
        # l 代表行号
        if not paragraphs or len(paragraphs) < 3:
            return
        i = 0
        while i < len(paragraphs) - 2:
            a = paragraphs[i]
            # 吞掉一个或多个连续的行号段 l
            j = i + 1
            saw_l = False
            while j < len(paragraphs) and self._is_ascii_digit_or_space_paragraph(
                paragraphs[j]
            ):
                saw_l = True
                j += 1
            # 现在 j 指向候选的 c
            if saw_l and j < len(paragraphs):
                c = paragraphs[j]
                if self._same_layout_and_xobj(a, c):
                    a.pdf_paragraph_composition.extend(c.pdf_paragraph_composition)
                    self.update_paragraph_data(a)
                    del paragraphs[j]
                    # 不移动 i，继续尝试把更多正文接到 a，实现 a l+ a l+ a ... 链式合并
                    continue
            i += 1

    def _group_characters_into_paragraphs(
        self, page: Page, layout_index, layout_map
    ) -> list[PdfParagraph]:
        paragraphs: list[PdfParagraph] = []
        if page.pdf_paragraph:
            paragraphs.extend(page.pdf_paragraph)
            page.pdf_paragraph = []

        char_areas = [
            (char.visual_bbox.box.x2 - char.visual_bbox.box.x)
            * (char.visual_bbox.box.y2 - char.visual_bbox.box.y)
            for char in page.pdf_character
        ]
        median_char_area = 0.0
        if char_areas:
            char_areas.sort()
            mid = len(char_areas) // 2
            median_char_area = (
                char_areas[mid]
                if len(char_areas) % 2 == 1
                else (char_areas[mid - 1] + char_areas[mid]) / 2
            )

        current_paragraph: PdfParagraph | None = None
        current_layout: Layout | None = None
        skip_chars = []

        for char in page.pdf_character:
            char_layout = get_character_layout(char, layout_index, layout_map)
            # Check if character is in any formula layout and set formula_layout_id
            char.formula_layout_id = is_character_in_formula_layout(
                char, page, layout_index, layout_map
            )

            if not is_text_layout(char_layout) or self.is_isolated_formula(char):
                skip_chars.append(char)
                continue

            char_box = char.visual_bbox.box
            # char_pdf_box = char.box
            # if calculate_iou_for_boxes(char_box, char_pdf_box) < 0.2:
            #     char_box = char_pdf_box
            char_area = (char_box.x2 - char_box.x) * (char_box.y2 - char_box.y)
            is_small_char = char_area < median_char_area * 0.05

            is_new_paragraph = False
            if current_paragraph is None:
                is_new_paragraph = True
            elif (
                not (
                    is_small_char
                    and current_paragraph.pdf_paragraph_composition
                    and char_layout.id == current_layout.id
                )
                and char.char_unicode not in HEIGHT_NOT_USFUL_CHAR_IN_CHAR
            ):
                if (
                    (
                        char_layout.id != current_layout.id
                        and not SPACE_REGEX.match(char.char_unicode)
                    )
                    or (  # not same xobject
                        current_paragraph.pdf_paragraph_composition
                        and current_paragraph.pdf_paragraph_composition[
                            -1
                        ].pdf_character.xobj_id
                        != char.xobj_id
                    )
                    or (
                        is_bullet_point(char)
                        and not current_paragraph.pdf_paragraph_composition
                    )
                ):
                    is_new_paragraph = True

            if is_new_paragraph:
                current_layout = char_layout
                current_paragraph = PdfParagraph(
                    pdf_paragraph_composition=[],
                    layout_id=current_layout.id,
                    debug_id=generate_base58_id(),
                    layout_label=current_layout.name,
                )
                paragraphs.append(current_paragraph)

            current_paragraph.pdf_paragraph_composition.append(
                PdfParagraphComposition(pdf_character=char)
            )

        page.pdf_character = skip_chars
        for para in paragraphs:
            self.update_paragraph_data(para)
        return paragraphs

    def _merge_overlapping_clusters(
        self, lines: dict[int, list[PdfCharacter]], char_height_average: float
    ) -> dict[int, list[PdfCharacter]]:
        """
        Merge clusters that have significant y-axis overlap.
        If y_intersection / min_height > 0.5 or the distance between y-midlines is less than char_height_average, merge the two clusters.
        """
        if len(lines) <= 1:
            return lines

        # Calculate y-axis ranges for each cluster
        cluster_ranges = {}
        cluster_midlines = {}
        for label, chars in lines.items():
            y_values = [char.visual_bbox.box.y for char in chars] + [
                char.visual_bbox.box.y2 for char in chars
            ]
            y_min, y_max = min(y_values), max(y_values)
            cluster_ranges[label] = (y_min, y_max)
            cluster_midlines[label] = (y_min + y_max) / 2

        # Keep merging until no more merges are possible
        changed = True
        while changed:
            changed = False
            labels_to_check = list(lines.keys())

            for i in range(len(labels_to_check)):
                if not changed:  # Only continue if no merge happened in this iteration
                    for j in range(i + 1, len(labels_to_check)):
                        label1, label2 = labels_to_check[i], labels_to_check[j]

                        # Skip if either label has been merged away
                        if label1 not in lines or label2 not in lines:
                            continue

                        y1_min, y1_max = cluster_ranges[label1]
                        y2_min, y2_max = cluster_ranges[label2]

                        # Calculate intersection
                        intersection_start = max(y1_min, y2_min)
                        intersection_end = min(y1_max, y2_max)

                        # Calculate midline distance
                        midline_distance = abs(
                            cluster_midlines[label1] - cluster_midlines[label2]
                        )

                        should_merge = False
                        if (
                            intersection_end > intersection_start
                        ):  # There is intersection
                            intersection_height = intersection_end - intersection_start
                            height1 = y1_max - y1_min
                            height2 = y2_max - y2_min
                            min_height = min(height1, height2)

                            # Check if intersection ratio exceeds threshold
                            if (
                                min_height > 0
                                and intersection_height / min_height > 0.3
                            ):
                                should_merge = True

                        # Check if midline distance is less than char_height_average
                        if midline_distance < char_height_average:
                            should_merge = True

                        if should_merge:
                            # Merge label2 into label1
                            lines[label1].extend(lines[label2])
                            del lines[label2]

                            # Update cluster range and midline for the merged cluster
                            new_y_min = min(y1_min, y2_min)
                            new_y_max = max(y1_max, y2_max)
                            cluster_ranges[label1] = (new_y_min, new_y_max)
                            cluster_midlines[label1] = (new_y_min + new_y_max) / 2
                            del cluster_ranges[label2]
                            del cluster_midlines[label2]

                            changed = True
                            break

        return lines

    def _get_effective_y_bounds(self, char: PdfCharacter) -> tuple[float, float]:
        """
        Determines the effective vertical boundaries (y1, y2) for a character.

        It prioritizes the visual bounding box if its Intersection over Union (IoU)
        with the PDF bounding box is high (>= 0.5), otherwise, it falls back to the
        PDF bounding box. This helps use more accurate layout information when available.
        """
        visual_box = char.visual_bbox.box
        return visual_box.y, visual_box.y2
        pdf_box = char.box
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.5:
            return visual_box.y, visual_box.y2
        return pdf_box.y, pdf_box.y2

    @staticmethod
    def _compute_collision_counts_histogram(
        y1_arr: np.ndarray,
        y2_arr: np.ndarray,
        para_y_min: float,
        para_y_max: float,
        step: float,
    ) -> np.ndarray:
        """Compute overlap counts at each scan line using a difference-array histogram.

        Args:
            y1_arr: 1-D array with lower y bounds of characters (inclusive).
            y2_arr: 1-D array with upper y bounds of characters (exclusive).
            para_y_min: Minimum y of the paragraph.
            para_y_max: Maximum y of the paragraph.
            step: Scan step size.

        Returns:
            1-D NumPy int32 array where index i corresponds to y = para_y_max - i × step.
        """
        # Number of scan positions
        m = int(np.ceil((para_y_max - para_y_min) / step))
        if m <= 0:
            return np.array([], dtype=np.int32)

        # Map character bounds to discrete indices (top inclusive, bottom exclusive)
        starts = np.floor((para_y_max - y2_arr) / step).astype(np.int32)
        ends = np.floor((para_y_max - y1_arr) / step).astype(np.int32) + 1
        # Clip ends to the valid range [0, m]
        np.clip(ends, 0, m, out=ends)

        hist = np.zeros(m + 1, dtype=np.int32)
        np.add.at(hist, starts, 1)
        np.add.at(hist, ends, -1)

        return np.cumsum(hist[:-1])

    def _split_paragraph_into_lines(
        self, paragraph: PdfParagraph, formula_font_ids: set[str]
    ):
        """
        Splits a paragraph into lines using a "line-threading" method.

        This method works by scanning vertically across the paragraph's bounding
        box and counting how many characters intersect with a horizontal line
        at each y-coordinate. The regions with a low number of intersections
        (less than 2) are identified as gaps between lines. The characters
        are then partitioned into lines based on these identified gaps.
        """
        if not paragraph.pdf_paragraph_composition:
            return

        # 1. Extract all characters and other compositions from the paragraph.
        all_chars: list[PdfCharacter] = []
        other_compositions: list[PdfParagraphComposition] = []
        for comp in paragraph.pdf_paragraph_composition:
            if comp.pdf_character:
                all_chars.append(comp.pdf_character)
            else:
                other_compositions.append(comp)

        if not all_chars:
            return

        # 2. Determine effective y-bounds for each character and the paragraph's total vertical range.
        char_y_bounds = [
            {"char": char, "y1": y1, "y2": y2}
            for char in all_chars
            for y1, y2 in [self._get_effective_y_bounds(char)]
        ]

        if not char_y_bounds:
            paragraph.pdf_paragraph_composition = other_compositions
            self.update_paragraph_data(paragraph)
            return

        para_y_min = min(b["y1"] for b in char_y_bounds)
        para_y_max = max(b["y2"] for b in char_y_bounds)

        # If the paragraph is vertically flat, treat it as a single line.
        if (para_y_max - para_y_min) < 5:  # Using a small threshold
            # all_chars.sort(key=lambda c: c.visual_bbox.box.x)
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return

        # 3. Perform "threading" scan to create a collision histogram.
        # Scan from top (max y) to bottom (min y) with a step of 0.5.
        scan_y_min = para_y_min
        scan_y_max = para_y_max
        step = 0.25

        y_coordinates = np.arange(scan_y_max, scan_y_min, -step)

        # Compute collision counts using NumPy histogram (O(m + n))
        y1_arr = np.array([b["y1"] for b in char_y_bounds], dtype=np.float32)
        y2_arr = np.array([b["y2"] for b in char_y_bounds], dtype=np.float32)
        collision_counts = self._compute_collision_counts_histogram(
            y1_arr,
            y2_arr,
            scan_y_min,
            scan_y_max,
            step,
        )

        # 4. Find gaps (regions with low collision count) from the histogram.
        gaps = []
        in_gap = False
        for i, count in enumerate(collision_counts):
            if count < 1 and not in_gap:
                in_gap = True
                gap_start_index = i
            elif count >= 1 and in_gap:
                in_gap = False
                gaps.append((gap_start_index, i - 1))
        if in_gap:
            gaps.append((gap_start_index, len(collision_counts) - 1))

        # If no significant gaps are found, treat it as a single line.
        if not gaps:
            # all_chars.sort(key=lambda c: c.visual_bbox.box.x)
            single_line_composition = self.create_line(all_chars)
            paragraph.pdf_paragraph_composition = [
                single_line_composition
            ] + other_compositions
            self.update_paragraph_data(paragraph)
            return

        # 5. Assign characters to lines based on the identified gaps.
        # Calculate separator y-coordinates from the midpoints of the gaps.
        separator_y_coords = sorted(
            [y_coordinates[start_idx] for start_idx, end_idx in gaps],
            reverse=True,
        )

        lines: list[list[PdfCharacter]] = [
            [] for _ in range(len(separator_y_coords) + 1)
        ]

        for b in char_y_bounds:
            char_y_center = (b["y1"] + b["y2"]) / 2
            line_idx = 0
            # Find which line bucket the character belongs to.
            for sep_y in separator_y_coords:
                if char_y_center > sep_y:
                    break
                line_idx += 1
            lines[line_idx].append(b["char"])

        # 6. Rebuild the paragraph's composition list from the new lines.
        new_line_compositions = []
        for line_chars in lines:
            if line_chars:
                # Sort characters within each line by x-coordinate (left-to-right).
                # line_chars.sort(key=lambda c: c.visual_bbox.box.x)
                new_line_compositions.append(self.create_line(line_chars))

        # The lines are already sorted vertically due to the scanning process.
        paragraph.pdf_paragraph_composition = new_line_compositions + other_compositions
        self.update_paragraph_data(paragraph)

    def process_paragraph_spacing(self, paragraph: PdfParagraph):
        if not paragraph.pdf_paragraph_composition:
            return

        # 处理行级别的空格
        processed_lines = []
        for composition in paragraph.pdf_paragraph_composition:
            if not composition.pdf_line:
                processed_lines.append(composition)
                continue

            line = composition.pdf_line
            if not "".join(
                x.char_unicode for x in line.pdf_character
            ).strip():  # 跳过完全空白的行
                continue

            # 处理行内字符的尾随空格
            processed_chars = []
            for char in line.pdf_character:
                if not char.char_unicode.isspace():
                    processed_chars = processed_chars + [char]
                elif processed_chars:  # 只有在有非空格字符后才考虑保留空格
                    processed_chars.append(char)

            # 移除尾随空格
            while processed_chars and processed_chars[-1].char_unicode.isspace():
                processed_chars.pop()

            if processed_chars:  # 如果行内还有字符
                line = self.create_line(processed_chars)
                processed_lines.append(line)

        paragraph.pdf_paragraph_composition = processed_lines
        self.update_paragraph_data(paragraph)

    def create_line(self, chars: list[PdfCharacter]) -> PdfParagraphComposition:
        assert chars

        line = PdfLine(pdf_character=chars)
        self.update_line_data(line)
        return PdfParagraphComposition(pdf_line=line)

    def calculate_median_line_width(self, paragraphs: list[PdfParagraph]) -> float:
        # 收集所有行的宽度
        line_widths = []
        for paragraph in paragraphs:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line_widths.append(line.box.x2 - line.box.x)

        if not line_widths:
            return 0.0

        # 计算中位数
        line_widths.sort()
        mid = len(line_widths) // 2
        if len(line_widths) % 2 == 0:
            return (line_widths[mid - 1] + line_widths[mid]) / 2
        return line_widths[mid]

    def process_independent_paragraphs(
        self,
        paragraphs: list[PdfParagraph],
        median_width: float,
    ):
        i = 0
        while i < len(paragraphs):
            paragraph = paragraphs[i]
            if len(paragraph.pdf_paragraph_composition) <= 1:  # 跳过只有一行的段落
                i += 1
                continue

            j = 1
            while j < len(paragraph.pdf_paragraph_composition):
                prev_composition = paragraph.pdf_paragraph_composition[j - 1]
                if not prev_composition.pdf_line:
                    j += 1
                    continue

                prev_line = prev_composition.pdf_line
                prev_width = prev_line.box.x2 - prev_line.box.x

                # 如果前一行宽度小于中位数的一半，将当前行及后续行分割成新段落
                # 注意：目录条目的拆分统一由 split_toc_paragraphs 处理
                if (
                    self.translation_config.split_short_lines
                    and prev_width
                    < median_width * self.translation_config.short_line_split_factor
                ) or (
                    paragraph.pdf_paragraph_composition
                    and (current_line := paragraph.pdf_paragraph_composition[j])
                    and (line := current_line.pdf_line)
                    and (chars := line.pdf_character)
                    and (char := chars[0])
                    and is_bullet_point(char)
                ):
                    # 创建新的段落
                    new_paragraph = PdfParagraph(
                        box=Box(0, 0, 0, 0),  # 临时边界框
                        pdf_paragraph_composition=(
                            paragraph.pdf_paragraph_composition[j:]
                        ),
                        unicode="",
                        debug_id=generate_base58_id(),
                        layout_label=paragraph.layout_label,
                        layout_id=paragraph.layout_id,
                    )
                    # 更新原段落
                    paragraph.pdf_paragraph_composition = (
                        paragraph.pdf_paragraph_composition[:j]
                    )

                    # 更新两个段落的数据
                    self.update_paragraph_data(paragraph)
                    self.update_paragraph_data(new_paragraph)

                    # 在原段落后插入新段落
                    paragraphs.insert(i + 1, new_paragraph)
                    break
                j += 1
            i += 1

    def _analyze_toc_entry_line(self, line: PdfLine) -> dict | None:
        """判断一行是否为目录条目（标题 + 大段空隙/点引导符 + 页码）。

        典型目录行形如 "1.2.3 Some Title ........ 12"、
        "Foreword ........ iii" 或
        "1.2.3 Some Title              12"（页码右对齐、中间为空白）。
        仅靠句点无法识别空白引导（引导点也可能是图形而非文本），
        因此同时依据“行尾页码（阿拉伯/罗马数字）+ 与标题之间存在
        大水平空隙 + 多行页码右边缘对齐”的几何特征判断。

        Returns:
            包含条目几何信息的 dict；不是候选条目时返回 None。
        """
        return self._analyze_toc_entry_chars(line.pdf_character)

    def _analyze_toc_entry_chars(self, chars) -> dict | None:
        """字符序列版本的目录条目判定（供视觉行拆分后复用）。"""
        if len(chars) < 2 or any(c.visual_bbox is None for c in chars):
            return None

        ordered = sorted(chars, key=lambda c: c.visual_bbox.box.x)
        text = "".join(c.char_unicode or "" for c in ordered)

        # 1. 行尾页码（忽略尾部空格/零宽字符），支持阿拉伯与罗马数字
        i = len(ordered)
        while i > 0 and is_toc_separator_char(ordered[i - 1].char_unicode):
            i -= 1
        j = i
        while j > 0:
            ch = ordered[j - 1].char_unicode or ""
            if ch.isdigit() or ("a" <= ch.lower() <= "z"):
                j -= 1
            else:
                break
        number_chars = ordered[j:i]
        number_text = "".join(c.char_unicode or "" for c in number_chars)
        if not is_toc_page_number_token(number_text):
            return None

        # 2. 页码之前必须有标题文字（跳过页码与标题之间的分隔字符
        #    与引导点，使两种引导符共用同一套空隙计算）
        k = j
        while k > 0 and (
            is_toc_separator_char(ordered[k - 1].char_unicode)
            or is_toc_leader_char(ordered[k - 1].char_unicode)
        ):
            k -= 1
        if k == 0:
            return None  # 纯页码行
        title_chars = ordered[:k]
        if not any((c.char_unicode or "").isalpha() for c in title_chars):
            return None

        # 3. 点引导符（连续点或间隔点），不依赖空隙大小
        has_dot_leader = bool(TOC_DOT_LEADER_REGEX.search(text)) or (
            text.count(".") >= TOC_SPACED_DOT_MIN_COUNT
        )

        # 4. 标题末字符与页码首字符之间的水平空隙
        number_x0 = min(c.visual_bbox.box.x for c in number_chars)
        number_x2 = max(c.visual_bbox.box.x2 for c in number_chars)
        last_title_char = title_chars[-1]
        gap = number_x0 - last_title_char.visual_bbox.box.x2

        # 以行内空格宽度估计“正常词间距”，避免不同字号下阈值失真
        space_widths = [
            c.visual_bbox.box.x2 - c.visual_bbox.box.x
            for c in title_chars
            if c.char_unicode == " "
            and c.visual_bbox.box.x2 > c.visual_bbox.box.x
        ]
        if space_widths:
            space_width = statistics.median(space_widths)
        else:
            font_sizes = [
                getattr(c.pdf_style, "font_size", None) for c in title_chars
            ]
            font_sizes = [s for s in font_sizes if s]
            space_width = statistics.median(font_sizes) * 0.27 if font_sizes else 2.5
        gap_threshold = max(
            TOC_MIN_GAP,
            TOC_GAP_SPACE_WIDTH_FACTOR * space_width,
        )

        return {
            "has_dot_leader": has_dot_leader,
            "big_gap": gap >= gap_threshold,
            "number_x2": number_x2,
            "number_text": number_text,
        }

    def _split_line_visual_bands(self, line: PdfLine) -> tuple[list[PdfLine], bool]:
        """把一个 PdfLine 中被错误合并的多个视觉行按 y 坐标带拆开。

        部分 PDF（如澳大利亚预算文件目录）相邻视觉行的字符框纵向重叠，
        ``_split_paragraph_into_lines`` 会把多条目录行并入同一个 PdfLine，
        表现为行内字符存在多个相距约 13pt 的 y 值。按
        ``TOC_VISUAL_BAND_MIN_GAP`` 聚类为自上而下的多个视觉行，各自行内
        再按 x 排序；无法拆分时原样返回。
        """
        chars = line.pdf_character
        if len(chars) < 2 or any(c.box is None for c in chars):
            return [line], False
        # 注意必须用字符布局框 box（同一视觉行内各字符的 box.y 一致），
        # 不能用墨迹框 visual_bbox——其 y 随字形（大写/小写/句点）变化，
        # 会把同一行的字符撕碎。
        ordered = sorted(
            chars,
            key=lambda c: (-c.box.y, c.box.x),
        )
        bands: list[list[PdfCharacter]] = []
        anchor_y = None
        for ch in ordered:
            y = ch.box.y
            if anchor_y is None or abs(y - anchor_y) >= TOC_VISUAL_BAND_MIN_GAP:
                bands.append([ch])
                anchor_y = y
            else:
                bands[-1].append(ch)
        if len(bands) == 1:
            return [line], False
        band_lines = []
        for band_chars in bands:
            band_line = PdfLine(pdf_character=band_chars)
            self.update_line_data(band_line)
            band_lines.append(band_line)
        return band_lines, True

    def split_toc_paragraphs(self, paragraphs: list[PdfParagraph]):
        """将目录（TOC）多行段落拆分为“每条目一个段落”。

        判定需要整段满足：
        - 至少 ``TOC_MIN_ENTRY_LINES`` 行被识别为目录条目；
        - 条目行占比不低于 ``TOC_MIN_ENTRY_RATIO``；
        - 基于空隙的条目，其尾部页码右边缘需基本对齐。

        拆分在每个条目行之前进行；条目之间的非条目行（如过长标题的
        折行）附加到上一个条目段落中；首个条目之前的标题行独立成段。
        """
        idx = 0
        while idx < len(paragraphs):
            paragraph = paragraphs[idx]
            compositions = paragraph.pdf_paragraph_composition
            if not compositions or any(
                comp.pdf_line is None for comp in compositions
            ):
                idx += 1
                continue

            # 先把被错误合并到同一 PdfLine 的多个视觉行按 y 坐标带拆开，
            # 得到按阅读顺序排列的“视觉行”序列（每个视觉行即一个目录条目）
            visual_lines: list[PdfLine] = []
            any_band_split = False
            for comp in compositions:
                band_lines, expanded = self._split_line_visual_bands(comp.pdf_line)
                visual_lines.extend(band_lines)
                any_band_split = any_band_split or expanded

            infos = [
                self._analyze_toc_entry_line(visual_lines[vi])
                for vi in range(len(visual_lines))
            ]

            # 单个视觉行：本身就是一个完整条目，只需打标记供排版右对齐页码。
            # 点引导符足够特异；空白引导则要求页码贴近本行右边缘。
            if len(visual_lines) == 1:
                info = infos[0]
                if info is not None and (
                    info["has_dot_leader"]
                    or (
                        info["big_gap"]
                        and paragraph.box is not None
                        and abs(info["number_x2"] - paragraph.box.x2)
                        <= TOC_NUMBER_RIGHT_EDGE_TOLERANCE
                    )
                ):
                    paragraph.toc_page_number = info["number_text"]
                    paragraph.toc_page_number_x2 = info["number_x2"]
                    paragraph.toc_dot_leader = info["has_dot_leader"]
                idx += 1
                continue

            # 页码右边缘对齐校验（仅针对大空隙候选行）
            gap_candidates = [
                vi
                for vi, info in enumerate(infos)
                if info is not None and info["big_gap"]
            ]
            edge_confirmed: set[int] = set()
            if gap_candidates:
                edge_x2 = statistics.median(
                    infos[vi]["number_x2"] for vi in gap_candidates
                )
                for vi in gap_candidates:
                    if (
                        abs(infos[vi]["number_x2"] - edge_x2)
                        <= TOC_NUMBER_RIGHT_EDGE_TOLERANCE
                    ):
                        edge_confirmed.add(vi)

            entry_flags = [False] * len(visual_lines)
            for vi, info in enumerate(infos):
                if info is None:
                    continue
                if info["has_dot_leader"] or vi in edge_confirmed:
                    entry_flags[vi] = True

            entry_count = sum(entry_flags)
            ratio_ok = entry_count / len(visual_lines) >= TOC_MIN_ENTRY_RATIO
            # 普通多行段落沿用“至少 3 个条目”的保守阈值；
            # 若检测到 PdfLine 内部确有多视觉行被合并，该几何证据本身很强，
            # 放宽为至少 2 个条目（如只有 2 行的图表列表）。
            enough_entries = (
                entry_count >= TOC_MIN_ENTRY_LINES
                if not any_band_split
                else entry_count >= 2
            )
            if not ratio_ok or not enough_entries:
                idx += 1
                continue

            # 按条目切分视觉行；非条目行（如过长标题的折行）附加到上一条目
            groups: list[tuple[list[PdfLine], dict | None]] = []
            current: list[PdfLine] = []
            current_info: dict | None = None
            for vi, line in enumerate(visual_lines):
                if entry_flags[vi]:
                    if current:
                        groups.append((current, current_info))
                    current = [line]
                    current_info = infos[vi]
                else:
                    current.append(line)
            if current:
                groups.append((current, current_info))

            if len(groups) < 2:
                idx += 1
                continue

            new_paragraphs: list[PdfParagraph] = []
            for group_lines, entry_info in groups:
                group = [
                    PdfParagraphComposition(pdf_line=group_line)
                    for group_line in group_lines
                ]
                new_paragraph = PdfParagraph(
                    box=Box(0, 0, 0, 0),  # 临时边界框，稍后重算
                    pdf_paragraph_composition=group,
                    unicode="",
                    debug_id=generate_base58_id(),
                    layout_label=paragraph.layout_label,
                    layout_id=paragraph.layout_id,
                    xobj_id=paragraph.xobj_id,
                )
                if entry_info is not None:
                    # 标记为目录条目，供排版阶段强制页码右对齐
                    new_paragraph.toc_page_number = entry_info["number_text"]
                    new_paragraph.toc_page_number_x2 = entry_info["number_x2"]
                    new_paragraph.toc_dot_leader = entry_info["has_dot_leader"]
                self.update_paragraph_data(new_paragraph)
                new_paragraphs.append(new_paragraph)

            logger.debug(
                "Split TOC paragraph %s into %d entry paragraphs on page %s",
                paragraph.debug_id,
                len(new_paragraphs),
                paragraph.box,
            )
            paragraphs[idx : idx + 1] = new_paragraphs
            idx += len(new_paragraphs)

    @staticmethod
    def is_bbox_contain_in_vertical(bbox1: Box, bbox2: Box) -> bool:
        """Check if one bounding box is completely contained within the other."""
        # Check if bbox1 is contained in bbox2
        bbox1_in_bbox2 = bbox1.y >= bbox2.y and bbox1.y2 <= bbox2.y2
        # Check if bbox2 is contained in bbox1
        bbox2_in_bbox1 = bbox2.y >= bbox1.y and bbox2.y2 <= bbox1.y2
        return bbox1_in_bbox2 or bbox2_in_bbox1

    def fix_overlapping_paragraphs(self, page: Page):
        """
        Adjusts the bounding boxes of paragraphs on a page to resolve vertical overlaps.

        Iteratively checks pairs of paragraphs and adjusts their vertical boundaries
        (y and y2) if they overlap, aiming to place the boundary at the midpoint
        of the vertical overlap.
        """
        paragraphs = page.pdf_paragraph
        if not paragraphs or len(paragraphs) < 2:
            return

        max_iterations = len(paragraphs) * len(paragraphs)  # Safety break
        iterations = 0

        while iterations < max_iterations:
            iterations += 1
            overlap_found_in_pass = False

            for i in range(len(paragraphs)):
                for j in range(i + 1, len(paragraphs)):
                    para1 = paragraphs[i]
                    para2 = paragraphs[j]

                    if para1.box is None or para2.box is None:
                        continue

                    if para1.xobj_id != para2.xobj_id:
                        continue

                    # Check for overlap using the existing method
                    if self.bbox_overlap(para1.box, para2.box):
                        if self.is_bbox_contain_in_vertical(para1.box, para2.box):
                            continue
                        # Calculate vertical overlap details
                        overlap_y_start = max(para1.box.y, para2.box.y)
                        overlap_y_end = min(para1.box.y2, para2.box.y2)
                        overlap_height = overlap_y_end - overlap_y_start

                        # Calculate horizontal overlap details
                        overlap_x_start = max(para1.box.x, para2.box.x)
                        overlap_x_end = min(para1.box.x2, para2.box.x2)
                        overlap_width = overlap_x_end - overlap_x_start

                        # Ensure there's a real 2D overlap, focusing on vertical adjustment
                        if overlap_height > 1e-6 and overlap_width > 1e-6:
                            overlap_found_in_pass = True

                            # Determine which paragraph is visually higher
                            if para1.box.y2 > para2.box.y and para1.box.y < para2.box.y:
                                lower_para = para1
                                higher_para = para2
                            # Handle cases where y values are identical (or very close)
                            # Prefer the one with smaller y2 as the higher one, or break tie arbitrarily
                            elif para1.box.y2 < para2.box.y2:
                                lower_para = para1
                                higher_para = para2
                            else:
                                lower_para = para2
                                higher_para = para1

                            # Calculate the midpoint of the vertical overlap
                            mid_y = overlap_y_start + overlap_height / 2

                            # Adjust boxes, ensuring they remain valid (y2 > y)
                            if mid_y > higher_para.box.y and mid_y < lower_para.box.y2:
                                higher_para.box.y = mid_y + 1
                                lower_para.box.y2 = mid_y - 1
                            else:
                                # This might happen if one box is fully contained vertically
                                # within another, or due to floating point issues.
                                # Log a warning and skip adjustment for this pair in this iteration.
                                # A more complex strategy might be needed for full containment.
                                logger.warning(
                                    "Could not resolve overlap between paragraphs"
                                    f" {higher_para.debug_id} and {lower_para.debug_id}"
                                    " using simple midpoint strategy."
                                    f" Midpoint: {mid_y},"
                                    f" Higher Box: {higher_para.box},"
                                    f" Lower Box: {lower_para.box}"
                                )

            # If no overlaps were found and adjusted in this pass, we're done.
            if not overlap_found_in_pass:
                break

        if iterations == max_iterations:
            logger.warning(
                f"Maximum iterations ({max_iterations}) reached in"
                f" fix_overlapping_paragraphs for page {page.page_number}."
                " Some overlaps might remain."
            )

    def _sort_characters_in_lines(self, page: Page):
        """Sort characters in each line from left to right, top to bottom."""
        for paragraph in page.pdf_paragraph:
            for composition in paragraph.pdf_paragraph_composition:
                if composition.pdf_line:
                    line = composition.pdf_line
                    line.pdf_character.sort(key=self._get_char_sort_key)

    def _get_char_sort_key(self, char: PdfCharacter):
        """Get sort key for character positioning (top to bottom, left to right)."""
        visual_box = char.visual_bbox.box
        pdf_box = char.box

        # Use visual box if IoU with bbox is >= 0.1, otherwise use bbox
        if calculate_iou_for_boxes(visual_box, pdf_box) >= 0.1:
            box = visual_box
        else:
            box = pdf_box

        # Sort by y coordinate first (top to bottom), then x coordinate (left to right)
        # Note: In PDF coordinate system, y increases upward, so we negate y for top-to-bottom sorting
        return (box.x, -box.y)
