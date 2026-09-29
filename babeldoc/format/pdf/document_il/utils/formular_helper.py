import base64
import functools
import re
import unicodedata

from babeldoc.format.pdf.document_il.il_version_1 import Box
from babeldoc.format.pdf.document_il.il_version_1 import Page
from babeldoc.format.pdf.document_il.il_version_1 import PdfFormula
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.layout_helper import (
    formular_height_ignore_char,
)
from babeldoc.format.pdf.translation_config import TranslationConfig


def is_formulas_start_char(
    char: str,
    font_mapper: FontMapper,
    translation_config: TranslationConfig,
) -> bool:
    if not char:
        return False
    if "(cid:" in char:
        return True
    if not font_mapper.has_char(char):
        if len(char) > 1 and all(font_mapper.has_char(x) for x in char):
            return False
        return True
    if translation_config.formular_char_pattern:
        pattern = translation_config.formular_char_pattern
        if re.match(pattern, char):
            return True
    if char != " " and (
        unicodedata.category(char[0])
        in [
            # "Lm",
            "Mn",
            "Sk",
            "Sm",
            "Zl",
            "Zp",
            "Zs",
            "Co",  # private use character
            # "So",  # symbol
        ]  # 文字修饰符、数学符号、分隔符号
        or ord(char[0]) in range(0x370, 0x400)  # 希腊字母
    ):
        return True
    if re.match("[0-9\\[\\]•]", char):
        return True
    return False


def is_formulas_middle_char(
    char: str,
    font_mapper: FontMapper,
    translation_config: TranslationConfig,
) -> bool:
    if is_formulas_start_char(char, font_mapper, translation_config):
        return True

    if re.match(",", char):
        return True

    return False


def collect_page_formula_font_ids(
    page: Page, formular_font_pattern: str | None
) -> tuple[set[int], dict[str, set[int]]]:
    """
    Collects formula font IDs from page fonts and XObject fonts.

    Args:
        page: The Page object to process.
        formular_font_pattern: The regex pattern to identify formula fonts by name.

    Returns:
        A tuple containing:
            - A set of font_ids considered formula fonts at the page level.
            - A dictionary mapping xobj_id to a set of font_ids considered
              formula fonts for that specific XObject.
    """
    # Page-level formula font IDs
    page_formula_font_ids = set()
    if page.pdf_font:
        for font in page.pdf_font:
            if is_formulas_font(font.name, formular_font_pattern):
                page_formula_font_ids.add(font.font_id)

    # XObject-level formula font IDs
    xobj_formula_font_ids_map = {}
    if page.pdf_xobject:
        for xobj in page.pdf_xobject:
            # Start with a copy of page-level formula fonts for this XObject
            current_xobj_fonts = page_formula_font_ids.copy()
            if xobj.pdf_font:
                for font in xobj.pdf_font:
                    if is_formulas_font(font.name, formular_font_pattern):
                        current_xobj_fonts.add(font.font_id)
                    else:
                        # If a font within an XObject is explicitly not a formula font,
                        # remove it from this XObject's set.
                        current_xobj_fonts.discard(font.font_id)
            xobj_formula_font_ids_map[xobj.xobj_id] = current_xobj_fonts

    return page_formula_font_ids, xobj_formula_font_ids_map


@functools.cache
def is_formulas_font(font_name: str, formular_font_pattern: str | None) -> bool:
    pattern_text = (
        r"^("
        r"|BLKFort.*"
        r"|Cambria.*"
        r"|EUAlbertina.*"
        r"|NimbusRomNo9L.*"
        r"|GlosaMath.*"
        r"|URWPalladioL.*"
        r"|CMSS.+"
        r"|Arial.*"
        r"|TimesNewRoman.*"
        r"|SegoeUI.*"
        r"|CMTT9.*"
        r"|CMSL10.*"
        r"|CMTI10.*"
        r"|CMTT10.*"
        r"|CMTI12.*"
        r"|CMR12.*"
        r"|MeridienLTStd.*"
        r"|Calibri.*"
        r"|STIXMathJax_Main.*"
        r"|.*NewBaskerville.*"
        r"|.*FranklinGothic.*"
        r"|.*AGaramondPro.*"
        r"|.*PalatinoItalCOR.*"
        r"|.*ITCSymbolStd.*"
        r"|.*PlantinStd.*"
        r"|.*DJ5EscrowCond.*"
        r"|.*ExchangeBook.*"
        r"|.*DJ5Exchange.*"
        r"|.*Times.*"
        r"|.*PalatinoLTStd.*"
        r"|.*Times New Roman,Italic.*"
        r"|.*EhrhardtMT.*"
        r"|.*GillSansMTStd.*"
        r"|.*MedicineSymbols3.*"
        r"|.*HardingText.*"
        r"|.*GraphikNaturel.*"
        r"|.*HelveticaNeue.*"
        r"|.*GoudyOldStyleT.*"
        r"|.*Symbol.*"
        r"|.*ScalaSansLF.*"
        r"|.*ScalaLF.*"
        r"|.*ScalaSansPro.*"
        r"|.*PetersburgC.*"
        r"|.*ColiseumC.*"
        r"|.*Gantari.*"
        r"|.*OptimaLTStd.*"
        r"|.*CronosPro.*"
        r"|.*ACaslon.*"
        r"|.*Frutiger.*"
        r"|.*BrandonGrotesque.*"
        r"|.*FairfieldLH.*"
        r"|.*CaeciliaLTStd.*"
        r"|.*Whitney.*"
        r"|.*Mercury.*"
        r"|.*SabonLTStd.*"
        r"|.*AnonymousPro.*"
        r"|.*SabonLTPro.*"
        r"|.*ArnoPro.*"
        r"|.*CharisSIL.*"
        r"|.*MSReference.*"
        r"|.*CMUSerif-Roman.*"
        r"|.*CourierNewPS.*"
        r"|.*XCharter.*"
        r"|.*GillSans.*"
        r"|.*Perpetua.*"
        r"|.*GEInspira.*"
        r"|.*AGaramond.*"
        r"|.*BMath.*"
        r"|.*MSTT.*"
        r"|.*Bookinsanity.*"
        r"|.*ScalySans.*"
        r"|.*Code2000.*"
        r"|.*Minion.*"
        r"|.*JansonTextLT.*"
        r"|.*MathPack.*"
        r"|.*Macmillan.*"
        r"|.*NimbusSan.*"
        r"|.*Mincho.*"
        r"|.*Amerigo.*"
        r"|.*MSGloriolaIIStd.*"
        r"|.*CMU.+"
        r"|.*LinLibertine.*"
        r"|.*txsys.*"
        r")$"
    )
    precise_formula_font_pattern = (
        r"^("
        # r"|.*CambriaMath.*"
        # r"|.*Cambria Math.*"
        r"|.*Asana.*"
        r"|.*MiriamMonoCLM-BookOblique.*"
        r"|.*Miriam Mono CLM.*"
        r"|.*Logix.*"
        r"|.*AeBonum.*"
        r"|.*AeMRoman.*"
        r"|.*AePagella.*"
        r"|.*AeSchola.*"
        r"|.*Concrete.*"
        r"|.*LatinModernMathCompanion.*"
        r"|.*Latin Modern Math Companion.*"
        r"|.*RalphSmithsFormalScriptCompanion.*"
        r"|.*Ralph Smiths Formal Script Companion.*"
        r"|.*TeXGyreBonumMathCompanion.*"
        r"|.*TeX Gyre Bonum Companion.*"
        r"|.*TeXGyrePagellaMathCompanion.*"
        r"|.*TeX Gyre Pagella Math Companion.*"
        r"|.*TeXGyreTermesMathCompanion.*"
        r"|.*TeX Gyre Termes Math Companion.*"
        r"|.*XITSMathCompanion.*"
        r"|.*XITS Math Companion.*"
        r"|.*Erewhon.*"
        r"|.*Euler-Math.*"
        r"|.*Euler Math.*"
        r"|.*FiraMath-Regular.*"
        r"|.*Fira Math.*"
        r"|.*Garamond-Math.*"
        r"|.*GFSNeohellenicMath.*"
        r"|.*KpMath.*"
        r"|.*Lete Sans Math.*"
        r"|.*LeteSansMath.*"
        # r"|.*LinLibertineO.*"
        r"|.*Linux Libertine O.*"
        r"|.*LibertinusMath-Regular.*"
        r"|.*Libertinus Math.*"
        r"|.*LatinModernMath-Regular.*"
        r"|.*Latin Modern Math.*"
        r"|.*Luciole.*"
        r"|.*NewCMMath.*"
        r"|.*NewComputerModernMath.*"
        r"|.*OldStandard-Math.*"
        r"|.*STIXMath-Regular.*"
        r"|.*STIX Math.*"
        r"|.*STIXTwoMath-Regular.*"
        r"|.*STIX Two Math.*"
        r"|.*TeXGyreBonumMath.*"
        r"|.*TeX Gyre Bonum Math.*"
        r"|.*TeXGyreDejaVuMath.*"
        r"|.*TeX Gyre DejaVu Math.*"
        r"|.*TeXGyrePagellaMath.*"
        r"|.*TeX Gyre Pagella Math.*"
        r"|.*TeXGyreScholaMath.*"
        r"|.*TeX Gyre Schola Math.*"
        r"|.*TeXGyreTermesMath.*"
        r"|.*TeX Gyre Termes Math.*"
        r"|.*XCharter-Math.*"
        r"|.*XCharter Math.*"
        r"|.*XITSMath-Bold.*"
        r"|.*XITS Math.*"
        r"|.*XITSMath.*"
        r"|.*IBMPlexMath.*"
        r"|.*IBM Plex Math.*"
        r")$"
    )
    if formular_font_pattern:
        broad_formula_font_pattern = formular_font_pattern
    else:
        broad_formula_font_pattern = (
            r"(CM[^RB]"
            r"|(MS|XY|MT|BL|RM|EU|LA|RS)[A-Z]"
            r"|LINE"
            r"|LCIRCLE"
            r"|TeX-"
            r"|rsfs"
            r"|txsy"
            r"|wasy"
            r"|stmary"
            r"|.*Mono"
            r"|.*Code"
            # r"|.*Ital"
            r"|.*Sym"
            r"|.*Math"
            r"|AdvP4C4E74"
            r"|AdvPSSym"
            r"|AdvP4C4E59"
            r")"
        )

    if font_name.startswith("BASE64:"):
        font_name_bytes = base64.b64decode(font_name[7:])
        font = font_name_bytes.split(b"+")[-1]
        pattern_text = pattern_text.encode()
        broad_formula_font_pattern = broad_formula_font_pattern.encode()
    else:
        font = font_name.split("+")[-1]

    if not font:
        return False

    if re.match(precise_formula_font_pattern, font):
        return True
    elif re.match(pattern_text, font):
        return False
    elif re.match(broad_formula_font_pattern, font):
        return True

    return False


def update_formula_data(formula: PdfFormula):
    min_x = min(char.visual_bbox.box.x for char in formula.pdf_character)
    max_x = max(char.visual_bbox.box.x2 for char in formula.pdf_character)
    if not all(map(formular_height_ignore_char, formula.pdf_character)):
        min_y = min(
            char.visual_bbox.box.y
            for char in formula.pdf_character
            if not formular_height_ignore_char(char)
        )
        max_y = max(
            char.visual_bbox.box.y2
            for char in formula.pdf_character
            if not formular_height_ignore_char(char)
        )
    else:
        min_y = min(char.visual_bbox.box.y for char in formula.pdf_character)
        max_y = max(char.visual_bbox.box.y2 for char in formula.pdf_character)
    formula.box = Box(min_x, min_y, max_x, max_y)
    if not formula.y_offset:
        formula.y_offset = 0
    if not formula.x_offset:
        formula.x_offset = 0
    if not formula.x_advance:
        formula.x_advance = 0
