import io
import itertools
import logging
import os
import re
import time
import unicodedata
from multiprocessing import Process
from pathlib import Path

import freetype
import pymupdf

from babeldoc.assets.embedding_assets_metadata import FONT_NAMES
from babeldoc.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)

SUBSET_FONT_STAGE_NAME = "Subset font"
SAVE_PDF_STAGE_NAME = "Save PDF"
_EXTGSTATE_USAGE_RE = re.compile(rb"/([!#$%&'*+,\-.0-9:;=?@A-Z\\^_`a-z{|}~]+)\s+gs\b")
_SHADING_USAGE_RE = re.compile(rb"/([!#$%&'*+,\-.0-9:;=?@A-Z\\^_`a-z{|}~]+)\s+sh\b")


def to_int(src):
    return int(re.search(r"\d+", src).group(0))


def parse_mapping(text):
    mapping = []
    for x in re.finditer(rb"<(?P<num>[a-fA-F0-9]+)>", text):
        mapping.append(int(x.group("num"), 16))
    return mapping


def apply_normalization(cmap, gid, code):
    need = False
    if 0x2F00 <= code <= 0x2FD5:  # Kangxi Radicals
        need = True
    if 0xF900 <= code <= 0xFAFF:  # CJK Compatibility Ideographs
        need = True
    if need:
        norm = unicodedata.normalize("NFD", chr(code))
        cmap[gid] = ord(norm)
    else:
        cmap[gid] = code


def batched(iterable, n, *, strict=False):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        if strict and len(batch) != n:
            raise ValueError("batched(): incomplete batch")
        yield batch


def update_tounicode_cmap_pair(cmap, data):
    for start, stop, value in batched(data, 3):
        for gid in range(start, stop + 1):
            code = value + gid - start
            apply_normalization(cmap, gid, code)


def update_tounicode_cmap_code(cmap, data):
    for gid, code in batched(data, 2):
        apply_normalization(cmap, gid, code)


def parse_tounicode_cmap(data):
    cmap = {}
    for x in re.finditer(
        rb"\s+beginbfrange\s*(?P<r>(<[0-9a-fA-F]+>\s*)+)endbfrange\s+", data
    ):
        update_tounicode_cmap_pair(cmap, parse_mapping(x.group("r")))
    for x in re.finditer(
        rb"\s+beginbfchar\s*(?P<c>(<[0-9a-fA-F]+>\s*)+)endbfchar", data
    ):
        update_tounicode_cmap_code(cmap, parse_mapping(x.group("c")))
    return cmap


def parse_truetype_data(data):
    glyph_in_use = []
    face = freetype.Face(io.BytesIO(data))
    for i in range(face.num_glyphs):
        face.load_glyph(i)
        if face.glyph.outline.contours:
            glyph_in_use.append(i)
    return glyph_in_use


TOUNICODE_HEAD = """\
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo <</Registry(Adobe)/Ordering(UCS)/Supplement 0>> def
/CMapName /Adobe-Identity-UCS def
/CMapType 2 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange"""

TOUNICODE_TAIL = """\
endcmap
CMapName currentdict /CMap defineresource pop
end
end"""


def make_tounicode(cmap, used):
    short = []
    for x in used:
        if x in cmap:
            short.append((x, cmap[x]))
    line = [TOUNICODE_HEAD]
    for block in batched(short, 100):
        line.append(f"{len(block)} beginbfchar")
        for glyph, code in block:
            if code < 0x10000:
                line.append(f"<{glyph:04x}><{code:04x}>")
            else:
                code -= 0x10000
                high = 0xD800 + (code >> 10)
                low = 0xDC00 + (code & 0b1111111111)
                line.append(f"<{glyph:04x}><{high:04x}{low:04x}>")
        line.append("endbfchar")
    line.append(TOUNICODE_TAIL)
    return "\n".join(line)


def reproduce_one_font(doc, index):
    m = doc.xref_get_key(index, "ToUnicode")
    f = doc.xref_get_key(index, "DescendantFonts")
    if m[0] == "xref" and f[0] == "array":
        mi = to_int(m[1])
        fi = to_int(f[1])
        ff = doc.xref_get_key(fi, "FontDescriptor/FontFile2")
        ms = doc.xref_stream(mi)
        fs = doc.xref_stream(to_int(ff[1]))
        cmap = parse_tounicode_cmap(ms)
        used = parse_truetype_data(fs)
        text = make_tounicode(cmap, used)
        doc.update_stream(mi, bytes(text, "U8"))


def reproduce_cmap(doc):
    assert doc
    font_set = set()
    for page in doc:
        try:
            font_list = page.get_fonts()
            for font in font_list:
                if font[1] == "ttf" and font[3] in FONT_NAMES and ".ttf" in font[4]:
                    font_set.add(font)
        except Exception as e:
            logger.error(f"Error in getting page fonts: {e}")
    for font in font_set:
        reproduce_one_font(doc, font[0])
    return doc


def _subset_fonts_process(pdf_path, output_path):
    """Function to run in subprocess for font subsetting.

    Args:
        pdf_path: Path to the PDF file to subset
        output_path: Path where to save the result
    """
    try:
        pdf = pymupdf.open(pdf_path)
        pdf.subset_fonts(fallback=False)
        pdf.save(output_path)
        # 返回 0 表示成功
        os._exit(0)
    except Exception as e:
        logger.error(f"Error in font subsetting subprocess: {e}")
        # 返回 1 表示失败
        os._exit(1)


def _save_pdf_clean_process(
    pdf_path,
    output_path,
    garbage=1,
    deflate=True,
    clean=True,
    deflate_fonts=True,
    linear=False,
):
    """Function to run in subprocess for saving PDF with clean=True which can be time-consuming.

    Args:
        pdf_path: Path to the PDF file to save
        output_path: Path where to save the result
        garbage: Garbage collection level (0, 1, 2, 3, 4)
        deflate: Whether to deflate the PDF
        clean: Whether to clean the PDF
        deflate_fonts: Whether to deflate fonts
        linear: Whether to linearize the PDF
    """
    try:
        pdf = pymupdf.open(pdf_path)
        pdf.save(
            output_path,
            garbage=garbage,
            deflate=deflate,
            clean=clean,
            deflate_fonts=deflate_fonts,
            linear=linear,
        )
        # 返回 0 表示成功
        os._exit(0)
    except Exception as e:
        logger.error(f"Error in save PDF with clean=True subprocess: {e}")
        # 返回 1 表示失败
        os._exit(1)


class PDFCreater:
    @staticmethod
    def _parse_xref_ref(value: str) -> int | None:
        match = re.match(r"^\s*(\d+)\s+0\s+R\s*$", value)
        if match is None:
            return None
        return int(match.group(1))

    @classmethod
    def _find_resource_ref(
        cls,
        pdf: pymupdf.Document,
        candidate_resource_xrefs: list[int],
        resource_kind: str,
        name: str,
    ) -> str | None:
        for candidate_xref in candidate_resource_xrefs:
            try:
                value_type, value = pdf.xref_get_key(
                    candidate_xref,
                    f"Resources/{resource_kind}/{name}",
                )
            except Exception:
                continue
            if value_type != "null" and value != "null":
                return value
        return None

    @classmethod
    def _set_resource_ref(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        resource_kind: str,
        name: str,
        resource_ref: str,
    ) -> None:
        resource_key = f"Resources/{resource_kind}/{name}"
        try:
            pdf.xref_set_key(target_xref, resource_key, resource_ref)
            return
        except Exception as exc:
            if "has indirects" not in str(exc):
                raise
            direct_set_error = exc

        resources_type, resources_value = pdf.xref_get_key(
            target_xref,
            "Resources",
        )
        resources_xref = (
            cls._parse_xref_ref(resources_value) if resources_type == "xref" else None
        )
        if resources_xref is None:
            raise direct_set_error

        kind_type, kind_value = pdf.xref_get_key(resources_xref, resource_kind)
        kind_xref = cls._parse_xref_ref(kind_value) if kind_type == "xref" else None
        if kind_xref is not None:
            pdf.xref_set_key(kind_xref, name, resource_ref)
            return

        pdf.xref_set_key(
            resources_xref,
            f"{resource_kind}/{name}",
            resource_ref,
        )

    @classmethod
    def _ensure_stream_named_resources(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        stream: bytes,
        candidate_resource_xrefs: list[int],
        *,
        resource_kind: str,
        usage_re: re.Pattern[bytes],
    ) -> None:
        used_names = {
            match.group(1).decode("latin-1") for match in usage_re.finditer(stream)
        }
        if not used_names:
            return

        for name in sorted(used_names):
            try:
                current_type, current_value = pdf.xref_get_key(
                    target_xref,
                    f"Resources/{resource_kind}/{name}",
                )
            except Exception:
                current_type, current_value = "null", "null"
            if current_type != "null" and current_value != "null":
                continue

            resource_ref = cls._find_resource_ref(
                pdf,
                candidate_resource_xrefs,
                resource_kind,
                name,
            )
            if resource_ref is None:
                continue
            cls._set_resource_ref(
                pdf,
                target_xref,
                resource_kind,
                name,
                resource_ref,
            )

    @classmethod
    def _ensure_stream_extgstate_resources(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        stream: bytes,
        candidate_resource_xrefs: list[int],
    ) -> None:
        cls._ensure_stream_named_resources(
            pdf,
            target_xref,
            stream,
            candidate_resource_xrefs,
            resource_kind="ExtGState",
            usage_re=_EXTGSTATE_USAGE_RE,
        )

    @classmethod
    def _ensure_stream_shading_resources(
        cls,
        pdf: pymupdf.Document,
        target_xref: int,
        stream: bytes,
        candidate_resource_xrefs: list[int],
    ) -> None:
        cls._ensure_stream_named_resources(
            pdf,
            target_xref,
            stream,
            candidate_resource_xrefs,
            resource_kind="Shading",
            usage_re=_SHADING_USAGE_RE,
        )

    def create_side_by_side_dual_pdf(
        self,
        original_pdf: pymupdf.Document,
        translated_pdf: pymupdf.Document,
        dual_out_path: str,
        translation_config: TranslationConfig,
    ) -> pymupdf.Document:
        """Create a dual PDF with side-by-side pages (original and translation).

        Args:
            original_pdf: Original PDF document
            translated_pdf: Translated PDF document
            dual_out_path: Output path for the dual PDF
            translation_config: Translation configuration

        Returns:
            The created dual PDF document
        """
        # Create a new PDF for side-by-side pages
        dual = pymupdf.open()
        page_count = min(original_pdf.page_count, translated_pdf.page_count)

        for page_id in range(page_count):
            # Get pages from both PDFs
            orig_page = original_pdf[page_id]
            trans_page = translated_pdf[page_id]
            rotate_angle = orig_page.rotation
            total_width = orig_page.rect.width + trans_page.rect.width
            max_height = max(orig_page.rect.height, trans_page.rect.height)
            left_width = (
                orig_page.rect.width
                if not translation_config.dual_translate_first
                else trans_page.rect.width
            )

            orig_page.set_rotation(0)
            trans_page.set_rotation(0)

            # Create new page with combined width
            dual_page = dual.new_page(width=total_width, height=max_height)

            # Define rectangles for left and right sides
            rect_left = pymupdf.Rect(0, 0, left_width, max_height)
            rect_right = pymupdf.Rect(left_width, 0, total_width, max_height)

            # Show pages according to dual_translate_first setting
            if translation_config.dual_translate_first:
                # Show translated page on left and original on right
                rect_left, rect_right = rect_right, rect_left
            try:
                # Show original page on left and translated on right (default)
                dual_page.show_pdf_page(
                    rect_left,
                    original_pdf,
                    page_id,
                    keep_proportion=True,
                    rotate=-rotate_angle,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to show original page on left and translated on right (default). "
                    f"Page ID: {page_id}. "
                    f"Original PDF: {self.original_pdf_path}. "
                    f"Translated PDF: {translation_config.input_file}. ",
                    exc_info=e,
                )
            try:
                dual_page.show_pdf_page(
                    rect_right,
                    translated_pdf,
                    page_id,
                    keep_proportion=True,
                    rotate=-rotate_angle,
                )
            except Exception as e:
                logger.warning(
                    f"Failed to show translated page on left and original on right. "
                    f"Page ID: {page_id}. "
                    f"Original PDF: {self.original_pdf_path}. "
                    f"Translated PDF: {translation_config.input_file}. ",
                    exc_info=e,
                )
        return dual

    def create_alternating_pages_dual_pdf(
        self,
        original_pdf: pymupdf.Document,
        translated_pdf: pymupdf.Document,
        translation_config: TranslationConfig,
    ) -> pymupdf.Document:
        """Create a dual PDF with alternating pages (original and translation).

        Args:
            original_pdf_path: Path to the original PDF
            translated_pdf: Translated PDF document
            translation_config: Translation configuration

        Returns:
            The created dual PDF document
        """
        # Open the original PDF and insert translated PDF
        dual = original_pdf
        dual.insert_file(translated_pdf)

        # Rearrange pages to alternate between original and translated
        page_count = translated_pdf.page_count
        for page_id in range(page_count):
            if translation_config.dual_translate_first:
                dual.move_page(page_count + page_id, page_id * 2)
            else:
                dual.move_page(page_count + page_id, page_id * 2 + 1)

        return dual

    @staticmethod
    def subset_fonts_in_subprocess(
        pdf: pymupdf.Document, translation_config: TranslationConfig, tag: str
    ) -> pymupdf.Document:
        """Run font subsetting in a subprocess with timeout.

        Args:
            pdf: The PDF document object
            translation_config: Translation configuration

        Returns:
            Path to the PDF with subsetted fonts, or original path if subsetting failed or timed out
        """
        original_pdf = pdf
        # Create temporary file paths
        temp_input = str(
            translation_config.get_working_file_path(f"temp_subset_input_{tag}.pdf")
        )
        temp_output = str(
            translation_config.get_working_file_path(f"temp_subset_output_{tag}.pdf")
        )

        # Save PDF to temporary file without subsetting
        pdf.save(temp_input)

        # Create and start subprocess
        process = Process(target=_subset_fonts_process, args=(temp_input, temp_output))
        process.start()

        # Wait for subprocess with timeout (1 minute)
        timeout = 60  # 1 minutes in seconds
        start_time = time.time()

        while process.is_alive():
            if time.time() - start_time > timeout:
                logger.warning(
                    f"Font subsetting timeout after {timeout} seconds, terminating subprocess"
                )
                process.terminate()
                try:
                    process.join(5)  # Give it 5 seconds to clean up
                    if process.is_alive():
                        logger.warning("Subprocess did not terminate, killing it")
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                        process.kill()
                        process.terminate()
                except Exception as e:
                    logger.error(f"Error terminating font subsetting process: {e}")

                return original_pdf

            time.sleep(0.5)  # Check every half second

        # Process completed, check exit code
        exit_code = process.exitcode
        success = exit_code == 0

        # Check if subsetting was successful
        if (
            success
            and Path(temp_output).exists()
            and Path(temp_output).stat().st_size > 0
        ):
            logger.info("Font subsetting completed successfully")
            return pymupdf.open(temp_output)
        else:
            logger.warning(
                f"Font subsetting failed with exit code {exit_code} or produced empty file"
            )
            return original_pdf

    @staticmethod
    def save_pdf_with_timeout(
        pdf: pymupdf.Document,
        output_path: str,
        translation_config: TranslationConfig,
        garbage: int = 1,
        deflate: bool = True,
        clean: bool = True,
        deflate_fonts: bool = True,
        linear: bool = False,
        timeout: int = 120,
        tag: str = "",
    ) -> bool:
        """Save a PDF document with a timeout for the clean=True operation.

        Args:
            pdf: The PDF document object
            output_path: Path where to save the PDF
            translation_config: Translation configuration
            garbage: Garbage collection level (0, 1, 2, 3, 4)
            deflate: Whether to deflate the PDF
            clean: Whether to clean the PDF
            deflate_fonts: Whether to deflate fonts
            linear: Whether to linearize the PDF
            timeout: Timeout in seconds (default: 2 minutes)

        Returns:
            True if saved with clean=True successfully, False if fallback to clean=False was used
        """
        # Create temporary file paths
        temp_input = str(
            translation_config.get_working_file_path(f"temp_save_input_{tag}.pdf")
        )
        temp_output = str(
            translation_config.get_working_file_path(f"temp_save_output_{tag}.pdf")
        )

        # Save PDF to temporary file first
        pdf.save(temp_input)

        # Try to save with clean=True in a subprocess
        process = Process(
            target=_save_pdf_clean_process,
            args=(
                temp_input,
                temp_output,
                garbage,
                deflate,
                clean,
                deflate_fonts,
                linear,
            ),
        )
        process.start()

        # Wait for subprocess with timeout
        process.join(timeout)
        if process.is_alive():
            logger.warning(
                f"PDF save with clean={clean} timeout after {timeout} seconds, terminating subprocess"
            )
            process.terminate()
            try:
                process.join(5)  # Give it 5 seconds to clean up
                if process.is_alive():
                    logger.warning("Subprocess did not terminate, killing it")
                    process.kill()
                    process.terminate()
                    process.kill()
                    process.terminate()
                    process.kill()
                    process.terminate()
            except Exception as e:
                logger.error(f"Error terminating PDF save process: {e}")

            # Fallback to save without clean parameter
            logger.info("Falling back to save with clean=False")
            try:
                pdf.save(
                    output_path,
                    garbage=garbage,
                    deflate=deflate,
                    clean=False,
                    deflate_fonts=deflate_fonts,
                    linear=linear,
                )
                return False
            except Exception as e:
                logger.error(f"Error in fallback save: {e}")
                # Last resort: basic save
                pdf.save(output_path)
                return False

        # Process completed, check exit code
        exit_code = process.exitcode
        success = exit_code == 0

        # Check if save was successful
        if (
            success
            and Path(temp_output).exists()
            and Path(temp_output).stat().st_size > 0
        ):
            logger.info(f"PDF save with clean={clean} completed successfully")
            # Copy the successfully created file to the target path
            try:
                import shutil

                shutil.copy2(temp_output, output_path)
                return True
            except Exception as e:
                logger.error(f"Error copying saved PDF: {e}")
                pdf.save(output_path)  # Fallback to direct save
                return False
            finally:
                Path(temp_input).unlink()
                Path(temp_output).unlink()
        else:
            logger.warning(
                f"PDF save with clean={clean} failed with exit code {exit_code} or produced empty file"
            )
            # Fallback to save without clean parameter
            try:
                pdf.save(
                    output_path,
                    garbage=garbage,
                    deflate=deflate,
                    clean=False,
                    deflate_fonts=deflate_fonts,
                    linear=linear,
                )
            except Exception as e:
                logger.error(f"Error in fallback save: {e}")
                # Last resort: basic save
                pdf.save(output_path)

            return False

    def restore_media_box(self, doc: pymupdf.Document, mediabox_data: dict) -> None:
        for xref, page_box_data in mediabox_data.items():
            for name, box in page_box_data.items():
                try:
                    doc.xref_set_key(xref, name, box)
                except Exception:
                    logger.debug(f"Error restoring media box {name} from PDF")
