import logging
import math
import re
import unicodedata
from typing import Literal

import regex
from pymupdf import Font

from babeldoc.format.pdf.document_il import GraphicState
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.il_version_1 import Box
from babeldoc.format.pdf.document_il.il_version_1 import PdfCharacter
from babeldoc.format.pdf.document_il.il_version_1 import PdfParagraph
from babeldoc.format.pdf.document_il.il_version_1 import PdfParagraphComposition

logger = logging.getLogger(__name__)
HEIGHT_NOT_USFUL_CHAR_IN_CHAR = (
    "∑︁",
    # 暂时假设 cid:17 和 cid 16 是特殊情况
    # 来源于 arXiv:2310.18608v2 第九页公式大括号
    "(cid:17)",
    "(cid:16)",
    # arXiv:2411.19509v2 第四页 []
    "(cid:104)",
    "(cid:105)",
    # arXiv:2411.19509v2 第四页 公式的 | 竖线
    "(cid:13)",
    "∑︁",
    # arXiv:2412.05265 27 页 累加号
    "(cid:88)",
    # arXiv:2412.05265 16 页 累乘号
    "(cid:89)",
    # arXiv:2412.05265 27 页 积分
    "(cid:90)",
    # arXiv:2412.05265 32 页 公式左右的中括号
    "(cid:2)",
    "(cid:3)",
    "·",
    "√",
)


LEFT_BRACKET = ("(cid:8)", "(", "(cid:16)", "{", "[", "(cid:104)", "(cid:2)")
RIGHT_BRACKET = ("(cid:9)", ")", "(cid:17)", "}", "]", "(cid:105)", "(cid:3)")

BULLET_POINT_PATTERN = re.compile(
    r"[■•⚫⬤◆◇○●◦‣⁃▪▫∗†‡¹²³⁴⁵⁶⁷⁸⁹⁰₁₂₃₄₅₆₇₈₉₀ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖᵍʳˢᵗᵘᵛʷˣʸᶻ†‡§¶※⁑⁂⁕⁎⁜⁑❧☙⁋‖‽·]"
)


def is_bullet_point(char: PdfCharacter) -> bool:
    """Check if the character is a bullet point.

    Args:
        char: The character to check

    Returns:
        bool: True if the character is a bullet point
    """
    is_bullet = bool(BULLET_POINT_PATTERN.match(char.char_unicode))
    return is_bullet


def calculate_box_iou(box1: Box, box2: Box) -> float:
    """Calculate the Intersection over Union (IOU) between two boxes.

    Args:
        box1: First box
        box2: Second box

    Returns:
        float: IOU value between 0 and 1
    """
    if box1 is None or box2 is None:
        return 0.0

    # Calculate intersection
    x_left = max(box1.x, box2.x)
    y_top = max(box1.y, box2.y)
    x_right = min(box1.x2, box2.x2)
    y_bottom = min(box1.y2, box2.y2)

    # Check if there's no intersection
    if x_left >= x_right or y_top >= y_bottom:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of both boxes
    box1_area = (box1.x2 - box1.x) * (box1.y2 - box1.y)
    box2_area = (box2.x2 - box2.x) * (box2.y2 - box2.y)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Avoid division by zero
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def formular_height_ignore_char(char: PdfCharacter):
    return (
        char.pdf_character_id is None
        or char.char_unicode in HEIGHT_NOT_USFUL_CHAR_IN_CHAR
    )


def box_to_tuple(box: Box) -> tuple[float, float, float, float]:
    """Converts a Box object to a tuple of its coordinates."""
    if box is None:
        return (0, 0, 0, 0)
    return (box.x, box.y, box.x2, box.y2)


class Layout:
    def __init__(self, layout_id, name):
        self.id = layout_id
        self.name = name

    @staticmethod
    def is_newline(prev_char: PdfCharacter, curr_char: PdfCharacter) -> bool:
        # 如果没有前一个字符，不是换行
        if prev_char is None:
            return False

        # 获取两个字符的中心 y 坐标
        # prev_y = (prev_char.box.y + prev_char.box.y2) / 2
        # curr_y = (curr_char.box.y + curr_char.box.y2) / 2

        # 如果当前字符的 y 坐标明显低于前一个字符，说明换行了
        # 这里使用字符高度的一半作为阈值
        char_height = max(
            curr_char.box.y2 - curr_char.box.y,
            prev_char.box.y2 - prev_char.box.y,
        )
        char_width = max(
            curr_char.box.x2 - curr_char.box.x,
            prev_char.box.x2 - prev_char.box.x,
        )
        should_new_line = (
            curr_char.box.y2 < prev_char.box.y
            or curr_char.box.x2 < prev_char.box.x - char_width * 10
        )
        if should_new_line and (
            formular_height_ignore_char(curr_char)
            or formular_height_ignore_char(prev_char)
        ):
            return False
        return should_new_line


def get_paragraph_length_except(
    paragraph: PdfParagraph,
    except_chars: str,
    font: Font,
) -> int:
    length = 0
    for composition in paragraph.pdf_paragraph_composition:
        if composition.pdf_character:
            length += (
                composition.pdf_character[0].box.x2 - composition.pdf_character[0].box.x
            )
        elif composition.pdf_same_style_characters:
            for pdf_char in composition.pdf_same_style_characters.pdf_character:
                if pdf_char.char_unicode in except_chars:
                    continue
                length += pdf_char.box.x2 - pdf_char.box.x
        elif composition.pdf_same_style_unicode_characters:
            for char_unicode in composition.pdf_same_style_unicode_characters.unicode:
                if char_unicode in except_chars:
                    continue
                length += font.char_lengths(
                    char_unicode,
                    composition.pdf_same_style_unicode_characters.pdf_style.font_size,
                )[0]
        elif composition.pdf_line:
            for pdf_char in composition.pdf_line.pdf_character:
                if pdf_char.char_unicode in except_chars:
                    continue
                length += pdf_char.box.x2 - pdf_char.box.x
        elif composition.pdf_formula:
            length += composition.pdf_formula.box.x2 - composition.pdf_formula.box.x
        else:
            logger.error(
                f"Unknown composition type. "
                f"Composition: {composition}. "
                f"Paragraph: {paragraph}. ",
            )
            continue
    return length


def get_paragraph_unicode(paragraph: PdfParagraph) -> str:
    chars = []
    for composition in paragraph.pdf_paragraph_composition:
        if composition.pdf_line:
            chars.extend(composition.pdf_line.pdf_character)
        elif composition.pdf_same_style_characters:
            chars.extend(composition.pdf_same_style_characters.pdf_character)
        elif composition.pdf_same_style_unicode_characters:
            chars.extend(composition.pdf_same_style_unicode_characters.unicode)
        elif composition.pdf_formula:
            chars.extend(composition.pdf_formula.pdf_character)
        elif composition.pdf_character:
            chars.append(composition.pdf_character)
        else:
            logger.error(
                f"Unknown composition type. "
                f"Composition: {composition}. "
                f"Paragraph: {paragraph}. ",
            )
            continue
    return get_char_unicode_string(chars)


SPACE_REGEX = regex.compile(r"\s+", regex.UNICODE)


def get_char_unicode_string(chars: list[PdfCharacter | str]) -> str:
    """
    将字符列表转换为 Unicode 字符串，根据字符间距自动插入空格。
    有些 PDF 不会显式编码空格，这时需要根据间距自动插入空格。

    Args:
        chars: 字符列表，可以是 PdfCharacter 对象或字符串

    Returns:
        str: 处理后的 Unicode 字符串
    """
    # 计算字符间距的中位数
    distances = []
    for i in range(len(chars) - 1):
        if not (
            isinstance(chars[i], PdfCharacter)
            and isinstance(chars[i + 1], PdfCharacter)
        ):
            continue
        distance = chars[i + 1].box.x - chars[i].box.x2
        if distance > 1:  # 只考虑正向距离
            distances.append(distance)

    # 去重后的距离
    distinct_distances = sorted(set(distances))

    if not distinct_distances:
        median_distance = 1
    elif len(distinct_distances) == 1:
        median_distance = distinct_distances[0]
    else:
        median_distance = distinct_distances[1]

    # 构建 unicode 字符串，根据间距插入空格
    unicode_chars = []
    for i in range(len(chars)):
        # 如果不是字符对象，直接添加，一般来说这个时候 chars[i] 是字符串
        if not isinstance(chars[i], PdfCharacter):
            unicode_chars.append(chars[i])
            continue

        # use unicode regex to replace all space with " "
        unicode_chars.append(
            regex.sub(
                r"\s+",
                " ",
                unicodedata.normalize("NFKC", chars[i].char_unicode),
            )
        )

        # 如果是空格，跳过
        if chars[i].char_unicode == " ":
            continue

        # 如果两个字符都是 PdfCharacter，检查间距
        if i < len(chars) - 1 and isinstance(chars[i + 1], PdfCharacter):
            distance = chars[i + 1].box.x - chars[i].box.x2
            if distance >= median_distance or Layout.is_newline(  # 间距大于中位数
                chars[i],
                chars[i + 1],
            ):  # 换行
                unicode_chars.append(" ")  # 添加空格

    result = "".join(unicode_chars)
    # use unicode regex to replace all space with " "
    normalize = unicodedata.normalize("NFKC", result)
    result = SPACE_REGEX.sub(" ", normalize).strip()
    return result


def get_paragraph_max_height(paragraph: PdfParagraph) -> float:
    """
    获取段落中最高的排版单元高度。

    Args:
        paragraph: PDF 段落对象

    Returns:
        float: 最大高度值
    """
    max_height = 0.0
    for composition in paragraph.pdf_paragraph_composition:
        if composition is None:
            continue
        if composition.pdf_character:
            char_height = (
                composition.pdf_character[0].box.y2 - composition.pdf_character[0].box.y
            )
            max_height = max(max_height, char_height)
        elif composition.pdf_same_style_characters:
            for pdf_char in composition.pdf_same_style_characters.pdf_character:
                char_height = pdf_char.box.y2 - pdf_char.box.y
                max_height = max(max_height, char_height)
        elif composition.pdf_same_style_unicode_characters:
            # 对于纯 Unicode 字符，我们使用其样式中的字体大小作为高度估计
            font_size = (
                composition.pdf_same_style_unicode_characters.pdf_style.font_size
            )
            max_height = max(max_height, font_size)
        elif composition.pdf_line:
            for pdf_char in composition.pdf_line.pdf_character:
                char_height = pdf_char.box.y2 - pdf_char.box.y
                max_height = max(max_height, char_height)
        elif composition.pdf_formula:
            formula_height = (
                composition.pdf_formula.box.y2 - composition.pdf_formula.box.y
            )
            max_height = max(max_height, formula_height)
        else:
            logger.error(
                f"Unknown composition type. "
                f"Composition: {composition}. "
                f"Paragraph: {paragraph}. ",
            )
            continue
    return max_height


def is_same_style(style1, style2) -> bool:
    """判断两个样式是否相同"""
    if style1 is None or style2 is None:
        return style1 is style2

    return (
        style1.font_id == style2.font_id
        and math.fabs(style1.font_size - style2.font_size) < 0.02
        and is_same_graphic_state(style1.graphic_state, style2.graphic_state)
    )


def is_same_style_except_size(style1, style2) -> bool:
    """判断两个样式是否相同"""
    if style1 is None or style2 is None:
        return style1 is style2

    return (
        style1.font_id == style2.font_id
        and 0.7 < math.fabs(style1.font_size / style2.font_size) < 1.3
        and is_same_graphic_state(style1.graphic_state, style2.graphic_state)
    )


def is_same_style_except_font(style1, style2) -> bool:
    """判断两个样式是否相同"""
    if style1 is None or style2 is None:
        return style1 is style2

    return math.fabs(
        style1.font_size - style2.font_size,
    ) < 0.02 and is_same_graphic_state(style1.graphic_state, style2.graphic_state)


def is_same_graphic_state(state1: GraphicState, state2: GraphicState) -> bool:
    """判断两个 GraphicState 是否相同"""
    if state1 is None or state2 is None:
        return state1 is state2

    return (
        state1.linewidth == state2.linewidth
        and state1.dash == state2.dash
        and state1.flatness == state2.flatness
        and state1.intent == state2.intent
        and state1.linecap == state2.linecap
        and state1.linejoin == state2.linejoin
        and state1.miterlimit == state2.miterlimit
        and state1.ncolor == state2.ncolor
        and state1.scolor == state2.scolor
        and state1.stroking_color_space_name == state2.stroking_color_space_name
        and state1.non_stroking_color_space_name == state2.non_stroking_color_space_name
        and state1.passthrough_per_char_instruction
        == state2.passthrough_per_char_instruction
    )


def add_space_dummy_chars(paragraph: PdfParagraph) -> None:
    """
    在 PDF 段落中添加表示空格的 dummy 字符。
    这个函数会直接修改传入的 paragraph 对象，在需要空格的地方添加 dummy 字符。
    同时也会处理不同组成部分之间的空格。

    Args:
        paragraph: 需要处理的 PDF 段落对象
    """
    # 首先处理每个组成部分内部的空格
    for composition in paragraph.pdf_paragraph_composition:
        if composition.pdf_line:
            chars = composition.pdf_line.pdf_character
            _add_space_dummy_chars_to_list(chars)
        elif composition.pdf_same_style_characters:
            chars = composition.pdf_same_style_characters.pdf_character
            _add_space_dummy_chars_to_list(chars)
        elif composition.pdf_same_style_unicode_characters:
            # 对于 unicode 字符，不需要处理。
            # 这种类型只会出现在翻译好的结果中
            continue
        elif composition.pdf_formula:
            chars = composition.pdf_formula.pdf_character
            _add_space_dummy_chars_to_list(chars)

    # 然后处理组成部分之间的空格
    for i in range(len(paragraph.pdf_paragraph_composition) - 1):
        curr_comp = paragraph.pdf_paragraph_composition[i]
        next_comp = paragraph.pdf_paragraph_composition[i + 1]

        # 获取当前组成部分的最后一个字符
        curr_last_char = _get_last_char_from_composition(curr_comp)
        if not curr_last_char:
            continue

        # 获取下一个组成部分的第一个字符
        next_first_char = _get_first_char_from_composition(next_comp)
        if not next_first_char:
            continue

        # 检查两个组成部分之间是否需要添加空格
        distance = next_first_char.box.x - curr_last_char.box.x2
        if distance > 1:  # 只考虑正向距离
            # 创建一个 dummy 字符作为空格
            space_box = Box(
                x=curr_last_char.box.x2,
                y=curr_last_char.box.y,
                x2=curr_last_char.box.x2 + distance,
                y2=curr_last_char.box.y2,
            )

            space_char = PdfCharacter(
                pdf_style=curr_last_char.pdf_style,
                box=space_box,
                char_unicode=" ",
                scale=curr_last_char.scale,
                advance=space_box.x2 - space_box.x,
                visual_bbox=il_version_1.VisualBbox(box=space_box),
            )

            # 将空格添加到当前组成部分的末尾
            if curr_comp.pdf_line:
                curr_comp.pdf_line.pdf_character.append(space_char)
            elif curr_comp.pdf_same_style_characters:
                curr_comp.pdf_same_style_characters.pdf_character.append(space_char)
            elif curr_comp.pdf_formula:
                curr_comp.pdf_formula.pdf_character.append(space_char)


def _get_first_char_from_composition(
    comp: PdfParagraphComposition,
) -> PdfCharacter | None:
    """获取组成部分的第一个字符"""
    if comp.pdf_line and comp.pdf_line.pdf_character:
        return comp.pdf_line.pdf_character[0]
    elif (
        comp.pdf_same_style_characters and comp.pdf_same_style_characters.pdf_character
    ):
        return comp.pdf_same_style_characters.pdf_character[0]
    elif comp.pdf_formula and comp.pdf_formula.pdf_character:
        return comp.pdf_formula.pdf_character[0]
    elif comp.pdf_character:
        return comp.pdf_character
    return None


def _get_last_char_from_composition(
    comp: PdfParagraphComposition,
) -> PdfCharacter | None:
    """获取组成部分的最后一个字符"""
    if comp.pdf_line and comp.pdf_line.pdf_character:
        return comp.pdf_line.pdf_character[-1]
    elif (
        comp.pdf_same_style_characters and comp.pdf_same_style_characters.pdf_character
    ):
        return comp.pdf_same_style_characters.pdf_character[-1]
    elif comp.pdf_formula and comp.pdf_formula.pdf_character:
        return comp.pdf_formula.pdf_character[-1]
    elif comp.pdf_character:
        return comp.pdf_character
    return None


def _add_space_dummy_chars_to_list(chars: list[PdfCharacter]) -> None:
    """
    在字符列表中的适当位置添加表示空格的 dummy 字符。

    Args:
        chars: PdfCharacter 对象列表
    """
    if not chars:
        return

    # 计算字符间距的中位数
    distances = []
    for i in range(len(chars) - 1):
        distance = chars[i + 1].box.x - chars[i].box.x2
        if distance > 1:  # 只考虑正向距离
            distances.append(distance)

    # 去重后的距离
    distinct_distances = sorted(set(distances))

    if not distinct_distances:
        median_distance = 1
    elif len(distinct_distances) == 1:
        median_distance = distinct_distances[0]
    else:
        median_distance = distinct_distances[1]

    # 在需要的地方插入空格字符
    i = 0
    while i < len(chars) - 1:
        curr_char = chars[i]
        next_char = chars[i + 1]

        distance = next_char.box.x - curr_char.box.x2
        if distance >= median_distance or Layout.is_newline(curr_char, next_char):
            if distance < 0:
                distance = -distance
            # 创建一个 dummy 字符作为空格
            space_box = Box(
                x=curr_char.box.x2,
                y=curr_char.box.y,
                x2=curr_char.box.x2 + min(distance, median_distance),
                y2=curr_char.box.y2,
            )

            space_char = PdfCharacter(
                pdf_style=curr_char.pdf_style,
                box=space_box,
                char_unicode=" ",
                scale=curr_char.scale,
                advance=space_box.x2 - space_box.x,
                visual_bbox=il_version_1.VisualBbox(box=space_box),
            )

            # 在当前位置后插入空格字符
            chars.insert(i + 1, space_char)
            i += 2  # 跳过刚插入的空格
        else:
            i += 1


def build_layout_index(page):
    """Builds an R-tree index for all layouts on the page."""
    from rtree import index

    layout_index = index.Index()
    layout_map = {}
    for i, layout in enumerate(page.page_layout):
        layout_map[i] = layout
        if layout.box:
            layout_index.insert(i, box_to_tuple(layout.box))
    page.layout_index = layout_index
    page.layout_map = layout_map


def calculate_iou_for_boxes(box1: Box, box2: Box) -> float:
    """Calculate the intersection area divided by the first box area."""
    x_left = max(box1.x, box2.x)
    y_bottom = max(box1.y, box2.y)
    x_right = min(box1.x2, box2.x2)
    y_top = min(box1.y2, box2.y2)

    if x_right <= x_left or y_top <= y_bottom:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_top - y_bottom)

    # Calculate area of first box
    first_box_area = (box1.x2 - box1.x) * (box1.y2 - box1.y)

    # Return intersection divided by first box area, handle division by zero
    if first_box_area <= 0:
        return 0.0

    return intersection_area / first_box_area


def calculate_y_iou_for_boxes(box1: Box, box2: Box) -> float:
    """Calculate the intersection ratio in y-axis direction divided by the first box height.

    Args:
        box1: First box
        box2: Second box

    Returns:
        float: Intersection ratio in y-axis direction between 0 and 1
    """
    y_bottom = max(box1.y, box2.y)
    y_top = min(box1.y2, box2.y2)

    if y_top <= y_bottom:
        return 0.0

    # Calculate intersection height
    intersection_height = y_top - y_bottom

    # Calculate height of first box
    first_box_height = box1.y2 - box1.y

    # Return intersection divided by first box height, handle division by zero
    if first_box_height <= 0:
        return 0.0

    return intersection_height / first_box_height


def get_character_layout(
    char,
    page,
    layout_priority=None,
    bbox_mode: Literal["auto", "visual", "box"] = "auto",
):
    """Get the layout for a character based on priority and IoU."""
    if layout_priority is None:
        layout_priority = [
            "image",
            "number",
            "reference",
            "algorithm",
            "formula_caption",
            "isolate_formula",
            "table_footnote",
            "table_caption",
            "figure_caption",
            "table_text",
            "table",
            "figure",
            "abandon",
            "title",
            "paragraph_title",
            "abstract",
            "content",
            "figure_title",
            "chart_title",
            "table_title",
            "doc_title",
            "footnote",
            "header",
            "footer",
            "sealplain text",
            "tiny text",
            "text",
            "formula",
        ]

    char_box = char.visual_bbox.box
    char_box2 = char.box
    if bbox_mode == "auto":
        # Calculate IOU to decide which box to use
        intersection_area = max(
            0, min(char_box.x2, char_box2.x2) - max(char_box.x, char_box2.x)
        ) * max(0, min(char_box.y2, char_box2.y2) - max(char_box.y, char_box2.y))
        char_box_area = (char_box.x2 - char_box.x) * (char_box.y2 - char_box.y)

        if char_box_area > 0:
            iou = intersection_area / char_box_area
            if iou < 0.2:
                char_box = char_box2
    elif bbox_mode == "box":
        char_box = char_box2

    # Check if page has layout_index and layout_map
    if not hasattr(page, "layout_index") or not hasattr(page, "layout_map"):
        return None

    # Collect all intersecting layouts and their IoU values
    matching_layouts = []
    candidate_ids = list(page.layout_index.intersection(box_to_tuple(char_box)))
    candidate_layouts = [page.layout_map[i] for i in candidate_ids]

    for layout in candidate_layouts:
        # Calculate IoU
        intersection_area = max(
            0, min(char_box.x2, layout.box.x2) - max(char_box.x, layout.box.x)
        ) * max(0, min(char_box.y2, layout.box.y2) - max(char_box.y, layout.box.y))
        char_area = (char_box.x2 - char_box.x) * (char_box.y2 - char_box.y)

        if char_area > 0:
            iou = intersection_area / char_area
            if iou > 0:
                matching_layouts.append(
                    {
                        "layout": Layout(layout.id, layout.class_name),
                        "priority": layout_priority.index(layout.class_name)
                        if layout.class_name in layout_priority
                        else len(layout_priority),
                        "iou": iou,
                    }
                )

    if not matching_layouts:
        return None

    # Sort by priority (ascending) and IoU value (descending)
    matching_layouts.sort(key=lambda x: (x["priority"], -x["iou"]))

    return matching_layouts[0]["layout"]


def is_text_layout(layout: Layout):
    """Check if a layout is a text layout."""
    return layout is not None and layout.name in [
        "plain text",
        "tiny text",
        "title",
        "abandon",
        "figure_caption",
        "table_caption",
        "table_text",
        # "reference",
        "title",
        "paragraph_title",
        "abstract",
        "content",
        "figure_title",
        "table_title",
        "doc_title",
        "footnote",
        "header",
        "footer",
        "seal",
        "text",
        "chart_title",
    ]


def is_character_in_formula_layout(
    char: il_version_1.PdfCharacter, page: il_version_1.Page
) -> int | None:
    """Check if character is contained within any formula-related layout."""
    formula_layout_types = {"formula"}

    char_box = char.visual_bbox.box
    char_box2 = char.box

    if calculate_iou_for_boxes(char_box, char_box2) < 0.2:
        char_box = char_box2

    # Check if page has layout_index and layout_map
    if not hasattr(page, "layout_index") or not hasattr(page, "layout_map"):
        return False

    # Get all candidate layouts that intersect with the character
    candidate_ids = list(page.layout_index.intersection(box_to_tuple(char_box)))
    candidate_layouts: list[il_version_1.PageLayout] = [
        page.layout_map[i] for i in candidate_ids
    ]

    # Check if any intersecting layout is a formula type
    for layout in candidate_layouts:
        if layout.class_name in formula_layout_types:
            iou = calculate_iou_for_boxes(char_box, layout.box)
            if iou > 0.4:  # Character has overlap with formula layout
                return layout.id

    return None
