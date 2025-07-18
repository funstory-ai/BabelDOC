import enum
import functools
import logging
import re
from pathlib import Path

import pymupdf

from babeldoc.assets import assets
from babeldoc.format.pdf.document_il import PdfFont
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)


class PrimaryFontFamily(enum.IntEnum):
    SERIF = 1
    SANS_SERIF = 2
    SCRIPT = 3
    NONE = 4

    @classmethod
    def from_str(cls, value: str):
        if value == "serif":
            return cls.SERIF
        elif value == "sans-serif":
            return cls.SANS_SERIF
        elif value == "script":
            return cls.SCRIPT
        else:
            return cls.NONE


class FontMapper:
    stage_name = "Add Fonts"

    def __init__(self, translation_config: TranslationConfig):
        self.translation_config = translation_config
        assert translation_config.primary_font_family in [
            None,
            "serif",
            "sans-serif",
            "script",
        ]
        self.primary_font_family = PrimaryFontFamily.from_str(
            translation_config.primary_font_family,
        )

        font_family = assets.get_font_family(translation_config.lang_out)
        self.font_file_names = []
        for k in (
            "normal",
            "script",
            "fallback",
            "base",
        ):
            self.font_file_names.extend(font_family[k])

        self.fonts: dict[str, pymupdf.Font] = {}
        self.fontid2fontpath: dict[str, Path] = {}
        for font_file_name in self.font_file_names:
            if font_file_name in self.fontid2fontpath:
                continue
            font_path, font_metadata = assets.get_font_and_metadata(font_file_name)
            pymupdf_font = pymupdf.Font(fontfile=str(font_path))
            pymupdf_font.has_glyph = functools.lru_cache(maxsize=10240, typed=True)(
                pymupdf_font.has_glyph,
            )
            pymupdf_font.char_lengths = functools.lru_cache(maxsize=10240, typed=True)(
                pymupdf_font.char_lengths,
            )
            self.fonts[font_file_name] = pymupdf_font
            self.fontid2fontpath[font_file_name] = font_path
            self.fonts[font_file_name].font_id = font_file_name
            self.fonts[font_file_name].font_path = font_path
            self.fonts[font_file_name].ascent_fontmap = font_metadata["ascent"]
            self.fonts[font_file_name].descent_fontmap = font_metadata["descent"]
            self.fonts[font_file_name].encoding_length = font_metadata[
                "encoding_length"
            ]

        self.normal_font_ids: list[str] = font_family["normal"]
        self.script_font_ids: list[str] = font_family["script"]
        self.fallback_font_ids: list[str] = font_family["fallback"]
        self.base_font_ids: list[str] = font_family["base"]
        self.fontid2fontpath["base"] = self.fontid2fontpath[font_family["base"][0]]

        self.fontid2font: dict[str, pymupdf.Font] = {
            f.font_id: f for f in self.fonts.values()
        }

        self.fontid2font["base"] = self.fontid2font[self.base_font_ids[0]]

        self.normal_fonts: list[pymupdf.Font] = [
            self.fontid2font[font_id] for font_id in self.normal_font_ids
        ]
        self.script_fonts: list[pymupdf.Font] = [
            self.fontid2font[font_id] for font_id in self.script_font_ids
        ]
        self.fallback_fonts: list[pymupdf.Font] = [
            self.fontid2font[font_id] for font_id in self.fallback_font_ids
        ]

        self.base_font = self.fontid2font["base"]

        self.type2font: dict[str, list[pymupdf.Font]] = {
            "normal": self.normal_fonts,
            "script": self.script_fonts,
            "fallback": self.fallback_fonts,
            "base": [self.base_font],
        }

        self.has_char = functools.lru_cache(maxsize=10240, typed=True)(self.has_char)
        self.map_in_type = functools.lru_cache(maxsize=10240, typed=True)(
            self.map_in_type
        )

    def has_char(self, char_unicode: str):
        if len(char_unicode) != 1:
            return False
        current_char = ord(char_unicode)
        for font in self.fonts.values():
            if font.has_glyph(current_char):
                return True
        return False

    def map_in_type(
        self,
        bold: bool,
        italic: bool,
        monospaced: bool,
        serif: bool,
        char_unicode: str,
        font_type: str,
    ):
        if font_type == "script" and not italic:
            return None
        current_char = ord(char_unicode)
        for font in self.type2font[font_type]:
            if not font.has_glyph(current_char):
                continue
            if bool(bold) != bool(font.is_bold):
                continue
            # 不知道什么原因，思源黑体的 serif 属性为 1，先 workaround
            if bool(serif) and "serif" not in font.font_id.lower():
                continue
            if not bool(serif) and "serif" in font.font_id.lower():
                continue
            return font

        return None

    def map(self, original_font: PdfFont, char_unicode: str):
        current_char = ord(char_unicode)
        if isinstance(original_font, pymupdf.Font):
            bold = original_font.is_bold
            italic = original_font.is_italic
            monospaced = original_font.is_monospaced
            serif = original_font.is_serif
        elif isinstance(original_font, PdfFont):
            bold = original_font.bold
            italic = original_font.italic
            monospaced = original_font.monospace
            serif = original_font.serif
        else:
            logger.error(
                f"Unknown font type: {type(original_font)}. "
                f"Original font: {original_font}. "
                f"Char unicode: {char_unicode}. ",
            )
            return None

        if self.primary_font_family == PrimaryFontFamily.SERIF:
            serif = True
        elif self.primary_font_family == PrimaryFontFamily.SANS_SERIF:
            serif = False
        elif self.primary_font_family == PrimaryFontFamily.SCRIPT:
            serif = False
            italic = True

        script_font_map_result = self.map_in_type(
            bold, italic, monospaced, serif, char_unicode, "script"
        )
        if script_font_map_result:
            return script_font_map_result

        for script_font in self.script_fonts:
            if italic and script_font.has_glyph(current_char):
                return script_font

        normal_font_map_result = self.map_in_type(
            bold, italic, monospaced, serif, char_unicode, "normal"
        )
        if normal_font_map_result is not None:
            return normal_font_map_result

        fallback_font_map_result = self.map_in_type(
            bold, italic, monospaced, serif, char_unicode, "fallback"
        )
        if fallback_font_map_result is not None:
            return fallback_font_map_result

        for font in self.fallback_fonts:
            if font.has_glyph(current_char):
                return font

        logger.warning(
            f"Can't find font for {char_unicode}({current_char}). "
            f"Original font: {original_font.name}[{original_font.font_id}]. "
            f"Char unicode: {char_unicode}. ",
        )
        return None

    def get_used_font_ids(self, il: il_version_1.Document) -> set[str]:
        result = set()
        for page in il.page:
            for char in page.pdf_character:
                if char.pdf_style and char.pdf_style.font_id:
                    result.add(char.pdf_style.font_id)
            for para in page.pdf_paragraph:
                for comp in para.pdf_paragraph_composition:
                    if char := comp.pdf_character:
                        if char.pdf_style and char.pdf_style.font_id:
                            result.add(char.pdf_style.font_id)
        return result

    def add_font(self, doc_zh: pymupdf.Document, il: il_version_1.Document):
        used_font_ids = self.get_used_font_ids(il)
        font_list = [
            (k, v) for k, v in self.fontid2fontpath.items() if k in used_font_ids
        ]

        font_id = {}
        xreflen = doc_zh.xref_length()
        total = xreflen - 1 + len(font_list) + len(il.page) + len(font_list)
        with self.translation_config.progress_monitor.stage_start(
            self.stage_name,
            total,
        ) as pbar:
            if not il.page:
                pbar.advance(total)
                return
            for font in font_list:
                if font[0] in font_id:
                    continue
                font_id[font[0]] = doc_zh[0].insert_font(font[0], font[1])
                pbar.advance(1)
            for xref in range(1, xreflen):
                pbar.advance(1)
                # xref_type = doc_zh.xref_get_key(xref, "Type")
                # if xref_type[1] == "/Page":
                #     resources_xref = doc_zh.xref_get_key(xref, "Resources")
                #     if resources_xref[0] == 'null':
                #         doc_zh.xref_set_key(xref, "Resources", f"<</Font<<>>>>")
                for label in ["Resources/", ""]:  # 可能是基于 xobj 的 res
                    try:  # xref 读写可能出错
                        font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                        if font_res is None:
                            continue
                        target_key_prefix = f"{label}Font/"
                        if font_res[0] == "xref":
                            resource_xref_id = re.search(
                                "(\\d+) 0 R",
                                font_res[1],
                            ).group(1)
                            xref = int(resource_xref_id)
                            font_res = ("dict", doc_zh.xref_object(xref))
                            target_key_prefix = ""
                        if font_res[0] == "dict":
                            for font in font_list:
                                target_key = f"{target_key_prefix}{font[0]}"
                                font_exist = doc_zh.xref_get_key(xref, target_key)
                                if font_exist[0] == "null":
                                    doc_zh.xref_set_key(
                                        xref,
                                        target_key,
                                        f"{font_id[font[0]]} 0 R",
                                    )
                    except Exception:
                        pass

            # Create PdfFont for each font
            # 预先创建所有字体对象
            pdf_fonts = []
            for font_name, _ in font_list:
                # Get descent_fontmap from fontid2font
                assert font_name in self.fontid2font, f"Font {font_name} not found"
                mupdf_font = self.fontid2font[font_name]
                descent_fontmap = mupdf_font.descent_fontmap
                ascent_fontmap = mupdf_font.ascent_fontmap
                encoding_length = mupdf_font.encoding_length

                pdf_fonts.append(
                    il_version_1.PdfFont(
                        name=font_name,
                        xref_id=font_id[font_name],
                        font_id=font_name,
                        encoding_length=encoding_length,
                        bold=mupdf_font.is_bold,
                        italic=mupdf_font.is_italic,
                        monospace=mupdf_font.is_monospaced,
                        serif=mupdf_font.is_serif,
                        descent=descent_fontmap,
                        ascent=ascent_fontmap,
                    ),
                )
                pbar.advance(1)

            # 批量添加字体到页面和 XObject
            for page in il.page:
                page.pdf_font.extend(pdf_fonts)
                for xobj in page.pdf_xobject:
                    xobj.pdf_font.extend(pdf_fonts)
                pbar.advance(1)
