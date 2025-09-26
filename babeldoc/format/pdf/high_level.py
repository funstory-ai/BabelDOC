import asyncio
import copy
import hashlib
import io
import logging
import pathlib
import re
import shutil
import threading
import time
from asyncio import CancelledError
from pathlib import Path
from typing import Any
from typing import BinaryIO

import pymupdf
from pymupdf import Document
from pymupdf import Font

from babeldoc import asynchronize
from babeldoc.assets.assets import warmup
from babeldoc.babeldoc_exception.BabelDOCException import ExtractTextError
from babeldoc.const import CACHE_FOLDER
from babeldoc.format.pdf.converter import TranslateConverter
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.backend.pdf_creater import SAVE_PDF_STAGE_NAME
from babeldoc.format.pdf.document_il.backend.pdf_creater import SUBSET_FONT_STAGE_NAME
from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater
from babeldoc.format.pdf.document_il.backend.pdf_creater import reproduce_cmap
from babeldoc.format.pdf.document_il.frontend.il_creater import ILCreater
from babeldoc.format.pdf.document_il.midend.add_debug_information import (
    AddDebugInformation,
)
from babeldoc.format.pdf.document_il.midend.automatic_term_extractor import (
    AutomaticTermExtractor,
)
from babeldoc.format.pdf.document_il.midend.detect_scanned_file import DetectScannedFile
from babeldoc.format.pdf.document_il.midend.il_translator import ILTranslator
from babeldoc.format.pdf.document_il.midend.il_translator_llm_only import (
    ILTranslatorLLMOnly,
)
from babeldoc.format.pdf.document_il.midend.layout_parser import LayoutParser
from babeldoc.format.pdf.document_il.midend.paragraph_finder import ParagraphFinder
from babeldoc.format.pdf.document_il.midend.styles_and_formulas import StylesAndFormulas
from babeldoc.format.pdf.document_il.midend.table_parser import TableParser
from babeldoc.format.pdf.document_il.midend.typesetting import Typesetting
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.xml_converter import XMLConverter
from babeldoc.format.pdf.pdfinterp import PDFPageInterpreterEx
from babeldoc.format.pdf.result_merger import ResultMerger
from babeldoc.format.pdf.split_manager import SplitManager
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode
from babeldoc.pdfminer.pdfdocument import PDFDocument
from babeldoc.pdfminer.pdfinterp import PDFResourceManager
from babeldoc.pdfminer.pdfpage import PDFPage
from babeldoc.pdfminer.pdfparser import PDFParser
from babeldoc.progress_monitor import ProgressMonitor

logger = logging.getLogger(__name__)

TRANSLATE_STAGES = [
    (ILCreater.stage_name, 14.12),  # Parse PDF and Create IR
    (DetectScannedFile.stage_name, 2.45),  # DetectScannedFile
    (LayoutParser.stage_name, 14.03),  # Parse Page Layout
    (TableParser.stage_name, 1.0),  # Parse Table
    (ParagraphFinder.stage_name, 6.26),  # Parse Paragraphs
    (StylesAndFormulas.stage_name, 1.66),  # Parse Formulas and Styles
    # (RemoveDescent.stage_name, 0.15),  # Remove Char Descent
    (AutomaticTermExtractor.stage_name, 30.0),  # Extract Terms
    (ILTranslator.stage_name, 46.96),  # Translate Paragraphs
    (Typesetting.stage_name, 4.71),  # Typesetting
    (FontMapper.stage_name, 0.61),  # Add Fonts
    (PDFCreater.stage_name, 1.96),  # Generate drawing instructions
    (SUBSET_FONT_STAGE_NAME, 0.92),  # Subset font
    (SAVE_PDF_STAGE_NAME, 6.34),  # Save PDF
]

resfont_map = {
    "zh-cn": "china-ss",
    "zh-tw": "china-ts",
    "zh-hans": "china-ss",
    "zh-hant": "china-ts",
    "zh": "china-ss",
    "ja": "japan-s",
    "ko": "korea-s",
}


def safe_save(doc, *args, **kwargs):
    try:
        # first try, saving without options
        doc.save(*args, **kwargs)
    except Exception:
        # second try, saving with 'garbage=3' for object missing
        doc.ez_save(*args, **kwargs)


def fix_cmap(translate_result: TranslateResult, translate_config: TranslationConfig):
    processed = []
    for attr in (
        "mono_pdf_path",
        "dual_pdf_path",
        "no_watermark_mono_pdf_path",
        "no_watermark_dual_pdf_path",
    ):
        path = getattr(translate_result, attr)
        if not path or path in processed:
            continue
        processed.append(path)

        temp_path = translate_config.get_working_file_path(f"{path.stem}.cmap.pdf")
        pdf = pymupdf.open(path)
        reproduce_cmap(pdf)
        safe_save(pdf, temp_path)
        shutil.move(temp_path, path)


def verify_file_hash(file_path: str, expected_hash: str) -> bool:
    """Verify the SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with Path(file_path).open("rb") as f:
        # Read the file in chunks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest() == expected_hash


def translator_supports_llm(translator) -> bool:
    if not translator or not hasattr(translator, "do_llm_translate"):
        return False
    try:
        translator.do_llm_translate(None)
        return True
    except NotImplementedError:
        return False
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("translator %s failed llm detection: %s", translator, exc)
        return False


def start_parse_il(
    inf: BinaryIO,
    pages: list[int] | None = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    resfont: str = "",
    noto: Font = None,
    cancellation_event: asyncio.Event = None,
    il_creater: ILCreater = None,
    translation_config: TranslationConfig = None,
    **kwarg: Any,
) -> None:
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        service,
        resfont,
        noto,
        kwarg.get("envs", {}),
        kwarg.get("prompt", []),
        il_creater=il_creater,
    )
    # model = DocLayoutModel.load_available()

    assert device is not None
    assert il_creater is not None
    assert translation_config is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch, il_creater)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    il_creater.on_total_pages(total_pages)

    parser = PDFParser(inf)
    doc = PDFDocument(parser)

    for pageno, page in enumerate(PDFPage.create_pages(doc)):
        if cancellation_event and cancellation_event.is_set():
            raise CancelledError("task cancelled")
        if pages and (pageno not in pages):
            continue
        page.pageno = pageno

        if not translation_config.should_translate_page(pageno + 1):
            continue

        height, width = (
            page.cropbox[3] - page.cropbox[1],
            page.cropbox[2] - page.cropbox[0],
        )
        if height > 1200 or width > 2000:
            logger.warning(f"page {pageno + 1} is too large, maybe unable to translate")
            # continue

        translation_config.raise_if_cancelled()
        # The current program no longer relies on
        # the following layout recognition results,
        # but in order to facilitate the migration of pdf2zh,
        # the relevant code is temporarily retained.
        # pix = doc_zh[page.pageno].get_pixmap()
        # image = np.frombuffer(pix.samples, np.uint8).reshape(
        #     pix.height, pix.width, 3
        # )[:, :, ::-1]
        # page_layout = model.predict(
        #     image, imgsz=int(pix.height / 32) * 32)[0]
        # # kdtree 是不可能 kdtree 的，不如直接渲染成图片，用空间换时间
        # box = np.ones((pix.height, pix.width))
        # h, w = box.shape
        # vcls = ["abandon", "figure", "table",
        #         "isolate_formula", "formula_caption"]
        # for i, d in enumerate(page_layout.boxes):
        #     if page_layout.names[int(d.cls)] not in vcls:
        #         x0, y0, x1, y1 = d.xyxy.squeeze()
        #         x0, y0, x1, y1 = (
        #             np.clip(int(x0 - 1), 0, w - 1),
        #             np.clip(int(h - y1 - 1), 0, h - 1),
        #             np.clip(int(x1 + 1), 0, w - 1),
        #             np.clip(int(h - y0 + 1), 0, h - 1),
        #         )
        #         box[y0:y1, x0:x1] = i + 2
        # for i, d in enumerate(page_layout.boxes):
        #     if page_layout.names[int(d.cls)] in vcls:
        #         x0, y0, x1, y1 = d.xyxy.squeeze()
        #         x0, y0, x1, y1 = (
        #             np.clip(int(x0 - 1), 0, w - 1),
        #             np.clip(int(h - y1 - 1), 0, h - 1),
        #             np.clip(int(x1 + 1), 0, w - 1),
        #             np.clip(int(h - y0 + 1), 0, h - 1),
        #         )
        #         box[y0:y1, x0:x1] = 0
        # layout[page.pageno] = box
        # 新建一个 xref 存放新指令流
        # page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
        # doc_zh.update_object(page.page_xref, "<<>>")
        # doc_zh.update_stream(page.page_xref, b"")
        # doc_zh[page.pageno].set_contents(page.page_xref)
        ops_base = interpreter.process_page(page)
        il_creater.on_page_base_operation(ops_base)
        il_creater.on_page_end()
    il_creater.on_finish()
    device.close()


def translate(translation_config: TranslationConfig) -> TranslateResult:
    with ProgressMonitor(get_translation_stage(translation_config)) as pm:
        return do_translate(pm, translation_config)


def get_translation_stage(
    translation_config: TranslationConfig,
) -> list[tuple[str, float]]:
    result = copy.deepcopy(TRANSLATE_STAGES)
    should_remove = []

    # If only parsing and generating PDF, skip all translation-related stages
    if translation_config.only_parse_generate_pdf:
        should_remove.extend(
            [
                DetectScannedFile.stage_name,
                LayoutParser.stage_name,
                TableParser.stage_name,
                ParagraphFinder.stage_name,
                StylesAndFormulas.stage_name,
                AutomaticTermExtractor.stage_name,
                ILTranslator.stage_name,
                Typesetting.stage_name,
            ]
        )
    else:
        # Original logic for selective removal
        if not translation_config.table_model:
            should_remove.append(TableParser.stage_name)
        if translation_config.skip_scanned_detection:
            should_remove.append(DetectScannedFile.stage_name)
        if not translation_config.auto_extract_glossary:
            should_remove.append(AutomaticTermExtractor.stage_name)
        if translation_config.skip_translation:
            should_remove.append(ILTranslator.stage_name)

    result = [x for x in result if x[0] not in should_remove]
    return result


async def async_translate(translation_config: TranslationConfig):
    """Asynchronously translate a PDF file with real-time progress reporting.

    This function yields progress events that can be used to update progress bars
    or other UI elements. The events are dictionaries with the following structure:

    - progress_start: {
        "type": "progress_start",
        "stage": str,              # Stage name
        "stage_progress": float,   # Always 0.0
        "stage_current": int,      # Current count (0)
        "stage_total": int         # Total items in stage
    }
    - progress_update: {
        "type": "progress_update",
        "stage": str,              # Stage name
        "stage_progress": float,   # Stage progress (0-100)
        "stage_current": int,      # Current items processed
        "stage_total": int,        # Total items in stage
        "overall_progress": float  # Overall progress (0-100)
    }
    - progress_end: {
        "type": "progress_end",
        "stage": str,              # Stage name
        "stage_progress": float,   # Always 100.0
        "stage_current": int,      # Equal to stage_total
        "stage_total": int,        # Total items processed
        "overall_progress": float  # Overall progress (0-100)
    }
    - finish: {
        "type": "finish",
        "translate_result": TranslateResult
    }
    - error: {
        "type": "error",
        "error": str
    }

    Args:
        translation_config: Configuration for the translation process

    Yields:
        dict: Progress events during translation

    Raises:
        CancelledError: If the translation is cancelled
        Exception: Any other errors during translation
    """
    loop = asyncio.get_running_loop()
    callback = asynchronize.AsyncCallback()

    finish_event = asyncio.Event()
    cancel_event = threading.Event()
    with ProgressMonitor(
        get_translation_stage(translation_config),
        progress_change_callback=callback.step_callback,
        finish_callback=callback.finished_callback,
        finish_event=finish_event,
        cancel_event=cancel_event,
        loop=loop,
        report_interval=translation_config.report_interval,
    ) as pm:
        future = loop.run_in_executor(None, do_translate, pm, translation_config)
        try:
            async for event in callback:
                event = event.kwargs
                yield event
                if event["type"] == "error":
                    break
        except CancelledError:
            cancel_event.set()
        except KeyboardInterrupt:
            logger.info("Translation cancelled by user through keyboard interrupt")
            cancel_event.set()
    if cancel_event.is_set():
        future.cancel()
    logger.info("Waiting for translation to finish...")
    await finish_event.wait()


class MemoryMonitor:
    """Monitor memory usage of current process and all child processes."""

    def __init__(self, interval=0.1):
        """Initialize memory monitor.

        Args:
            interval: Monitoring interval in seconds, defaults to 0.1s (100ms)
        """
        self.interval = interval
        self.peak_memory_usage = 0
        self.monitor_thread = None
        self.stop_event = None

        try:
            import psutil

            self.psutil = psutil
        except ImportError:
            logger.warning("psutil not installed, memory monitoring disabled")
            self.psutil = None

    def __enter__(self):
        """Start memory monitoring."""
        if not self.psutil:
            return self

        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory_usage, daemon=True
        )
        self.monitor_thread.start()
        logger.debug("Memory monitoring started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log peak memory usage."""
        if not self.psutil or not self.monitor_thread:
            return

        self.stop_event.set()
        self.monitor_thread.join(timeout=2.0)
        logger.info(f"Peak memory usage: {self.peak_memory_usage:.2f} MB")

    def _monitor_memory_usage(self):
        """Background thread that periodically checks memory usage."""
        while not self.stop_event.is_set():
            try:
                current_process = self.psutil.Process()
                # Get memory usage of current process and all children
                total_memory = current_process.memory_info().rss
                for child in current_process.children(recursive=True):
                    try:
                        total_memory += child.memory_info().rss
                    except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                        pass

                # Convert to MB for better readability
                total_memory_mb = total_memory / (1024 * 1024)
                if total_memory_mb > self.peak_memory_usage:
                    self.peak_memory_usage = total_memory_mb
            except Exception as e:
                logger.warning(f"Error monitoring memory: {e}")

            time.sleep(self.interval)


def fix_null_page_content(doc: Document) -> list[int]:
    invalid_page = []
    for x in range(len(doc)):
        xref = doc[x].xref
        if doc.xref_object(xref) == "null":
            invalid_page.append(x)
    for x in invalid_page:
        doc.delete_page(x)
        doc.insert_page(x)
    return invalid_page


def fix_null_xref(doc: Document) -> None:
    """Fix null xref in PDF file by replacing them with empty arrays.

    Args:
        doc: PyMuPDF Document object to fix
    """
    for i in range(1, doc.xref_length()):
        try:
            obj = doc.xref_object(i)
            if obj == "null":
                doc.update_object(i, "[]")
            elif obj and "/ASCII85Decode" in obj:  # make pdfminer happy
                data = doc.xref_stream(i)
                doc.update_stream(i, data)
            elif obj and "/LZWDecode" in obj:
                data = doc.xref_stream(i)
                doc.update_stream(i, data)
            elif obj and "/Annots" in obj:
                doc.xref_set_key(i, "Annots", "null")
        except Exception:
            doc.update_object(i, "[]")


def fix_filter(doc):
    page_contents = []
    for page in doc:
        page_contents.extend(page.get_contents())
    for page_piece in page_contents:
        f = doc.xref_get_key(page_piece, "Filter")
        if f[0] == "xref":
            data = doc.xref_stream(page_piece)
            doc.update_stream(page_piece, data)
    for page in doc:
        contents = page.get_contents()
        t, v = doc.xref_get_key(page.xref, "Rotate")
        rotate = -int(v) if t == "int" else 0
        if len(contents) > 1 or rotate:
            page_streams = [doc.xref_stream(i) for i in contents]
            r = doc.get_new_xref()
            page_prefix = b""
            page_suffix = b""
            if rotate:
                m0 = pymupdf.Matrix(rotate)
                b0 = page.mediabox * m0
                m1 = m0 * pymupdf.Matrix(1, 0, 0, 1, b0.x0, -b0.y0)
                page_prefix = (
                    f" {m1.a} {m1.b} {m1.c} {m1.d} {m1.e} {m1.f} cm q ".encode()
                )
                page_suffix = b" Q "
                update_page_bbox(doc, page, page.cropbox * m1, "CropBox")
                update_page_bbox(doc, page, page.mediabox * m1, "MediaBox")
                update_page_bbox(doc, page, page.artbox * m1, "ArtBox")
                update_page_bbox(doc, page, page.bleedbox * m1, "BleedBox")
                doc.xref_set_key(page.xref, "Rotate", "0")
            doc.update_object(r, "<<>>")
            doc.update_stream(r, page_prefix + b" ".join(page_streams) + page_suffix)
            doc.xref_set_key(page.xref, "Contents", f"{r} 0 R")


def update_page_bbox(doc, page, box, key):
    if doc.xref_get_key(page.xref, key)[0] == "array":
        doc.xref_set_key(page.xref, key, f"[{box.x0} {box.y0} {box.x1} {box.y1}]")


def do_translate(
    pm: ProgressMonitor, translation_config: TranslationConfig
) -> TranslateResult:
    try:
        translation_config.progress_monitor = pm
        original_pdf_path = translation_config.input_file
        logger.info(f"start to translate: {original_pdf_path}")
        start_time = time.time()
        peak_memory_usage = 0
        with MemoryMonitor() as memory_monitor:
            # Check if split translation is enabled
            if not translation_config.split_strategy:
                result = _do_translate_single(pm, translation_config)
            else:
                # Initialize split manager and determine split points
                split_manager = SplitManager(translation_config)
                split_points = split_manager.determine_split_points(translation_config)

                if not split_points:
                    logger.warning(
                        "No split points determined, falling back to single translation"
                    )
                    result = _do_translate_single(pm, translation_config)
                else:
                    logger.info(f"Split points determined: {len(split_points)} parts")

                    if len(split_points) == 1:
                        logger.info("Only one part, use single translation")
                        result = _do_translate_single(pm, translation_config)
                    else:
                        pm.total_parts = len(split_points)

                        # Process parts serially
                        results: dict[int, TranslateResult | None] = {}
                        original_watermark_mode = (
                            translation_config.watermark_output_mode
                        )
                        original_doc = Document(original_pdf_path)
                        for i, split_point in enumerate(split_points):
                            try:
                                # Create a copy of config for this part
                                part_config = copy.copy(translation_config)
                                part_config.skip_clean = True
                                should_translate_pages = []
                                for page in range(
                                    split_point.start_page, split_point.end_page + 1
                                ):
                                    if translation_config.should_translate_page(
                                        page + 1
                                    ):
                                        should_translate_pages.append(
                                            page - split_point.start_page + 1
                                        )
                                part_config.pages = None
                                part_config.page_ranges = [
                                    (x, x) for x in should_translate_pages
                                ]
                                if (
                                    translation_config.only_include_translated_page
                                    and not should_translate_pages
                                ):
                                    results[i] = None
                                    continue

                                # Only first part should do scanned detection if enabled
                                if i > 0:
                                    part_config.skip_scanned_detection = True

                                part_config.working_dir = (
                                    translation_config.get_part_working_dir(i)
                                )
                                part_config.output_dir = (
                                    translation_config.get_part_output_dir(i)
                                )

                                assert id(
                                    part_config.shared_context_cross_split_part
                                ) == id(
                                    translation_config.shared_context_cross_split_part
                                ), "shared_context_cross_split_part must be the same"

                                part_temp_input_path = (
                                    part_config.get_working_file_path(
                                        f"input.part{i}.pdf"
                                    )
                                )
                                part_config.input_file = part_temp_input_path

                                temp_doc = Document()
                                for x in range(
                                    split_point.start_page, split_point.end_page + 1
                                ):
                                    xref = original_doc[x].xref
                                    if (
                                        original_doc.xref_get_key(xref, "Annots")[0]
                                        != "null"
                                    ):
                                        original_doc.xref_set_key(
                                            xref, "Annots", "null"
                                        )
                                temp_doc.insert_pdf(
                                    original_doc,
                                    from_page=split_point.start_page,
                                    to_page=split_point.end_page,
                                )
                                safe_save(temp_doc, part_temp_input_path)
                                assert (
                                    temp_doc.page_count
                                    == split_point.end_page - split_point.start_page + 1
                                )

                                # Only first part should have watermark
                                if i > 0:
                                    part_config.watermark_output_mode = (
                                        WatermarkOutputMode.NoWatermark
                                    )

                                # Create progress monitor for this part
                                part_monitor = pm.create_part_monitor(
                                    i, len(split_points)
                                )

                                # Process this part
                                result = _do_translate_single(
                                    part_monitor,
                                    part_config,
                                )
                                results[i] = result

                            except Exception as e:
                                logger.error(f"Error in part {i}: {e}")
                                pm.translate_error(e)
                                raise
                            finally:
                                # Clean up part working directory
                                translation_config.cleanup_part_working_dir(i)

                        # Restore original watermark mode
                        translation_config.watermark_output_mode = (
                            original_watermark_mode
                        )

                        # Merge results
                        merger = ResultMerger(translation_config)
                        logger.info("start merge results")
                        result = merger.merge_results(results)
                        logger.info("finish merge results")
            peak_memory_usage = memory_monitor.peak_memory_usage

        finish_time = time.time()
        result.total_seconds = finish_time - start_time

        logger.info(
            f"finish translate: {original_pdf_path}, cost: {finish_time - start_time} s",
        )
        result.original_pdf_path = translation_config.input_file
        result.peak_memory_usage = peak_memory_usage

        fix_cmap(result, translation_config)
        try:
            migrate_toc(translation_config, result)
        except Exception as e:
            logger.error(
                f"Failed to migrate TOC from {translation_config.input_file}: {e}"
            )
        pm.translate_done(result)
        return result

    except Exception as e:
        if translation_config.debug:
            logger.exception("translate error:")
        else:
            logger.error(f"translate error: {e}")
        pm.disable = False
        pm.translate_error(e)
        raise
    finally:
        logger.debug("do_translate finally")
        pm.on_finish()
        translation_config.cleanup_temp_files()


def migrate_toc(
    translation_config: TranslationConfig, translate_result: TranslateResult
):
    if translation_config.use_alternating_pages_dual:
        logger.info('skipping TOC migration for "use_alternating_pages_dual" mode')
        return
    old_doc = Document(translation_config.input_file)
    if not old_doc:
        return
    try:
        fix_filter(old_doc)
        fix_null_xref(old_doc)
    except Exception:
        logger.exception("auto fix failed, please check the pdf file")

    toc_data = old_doc.get_toc()

    if not toc_data:
        logger.info("No TOC found in the original PDF, skipping migration.")
        return

    if translation_config.only_include_translated_page:
        total_page = set(range(0, len(old_doc)))

        pages_to_translate = {
            i for i in len(old_doc) if translation_config.should_translate_page(i + 1)
        }

        should_removed_page = list(total_page - pages_to_translate)

    files = {
        translate_result.dual_pdf_path,
        # translate_result.mono_pdf_path,
        translate_result.no_watermark_dual_pdf_path,
        # translate_result.no_watermark_mono_pdf_path
    }

    for f in files:
        if not f:
            continue
        mig_toc_temp_input = translation_config.get_working_file_path(
            "mig_toc_temp.pdf"
        )
        shutil.copy(f, mig_toc_temp_input)
        new_doc = Document(mig_toc_temp_input.as_posix())
        if not new_doc:
            continue

        new_doc.set_toc(toc_data)
        PDFCreater.save_pdf_with_timeout(
            new_doc,
            f.as_posix(),
            translation_config=translation_config,
            clean=not translation_config.skip_clean,
            tag="mig_toc",
        )


# mediabox -> '[0 nul 792]'
def fix_media_box(doc: Document) -> None:
    mediabox_data = {}
    for x in range(1, doc.xref_length()):
        t = doc.xref_get_key(x, "Type")
        box_set = {}
        if t[1] in ["/Pages", "/Page"]:
            mediabox = doc.xref_get_key(x, "MediaBox")
            if mediabox[0] == "array":
                try:
                    _, _, x1, y1 = (
                        mediabox[1].replace("[", "").replace("]", "").split(" ")
                    )
                    doc.xref_set_key(x, "MediaBox", f"[0 0 {x1} {y1}]")
                    box_set["MediaBox"] = mediabox[1]
                except Exception:
                    logger.warning(
                        "Attempt to fix media box failed; some pages may not have been processed correctly."
                    )
            for k in ["CropBox", "BleedBox", "TrimBox", "ArtBox"]:
                box = doc.xref_get_key(x, k)
                if box[0] != "null":
                    box_set[k] = box[1]
                    doc.xref_set_key(x, k, "null")
        if box_set:
            mediabox_data[x] = box_set
    return mediabox_data


def check_cid_char(il: il_version_1.Document):
    chars = []
    for page in il.page:
        chars.extend(page.pdf_character)

    cid_count = 0
    for char in chars:
        if re.match(r"^\(cid:\d+\)$", char.char_unicode):
            cid_count += 1

    return cid_count > len(chars) * 0.8


def _do_translate_single(
    pm: ProgressMonitor,
    translation_config: TranslationConfig,
) -> TranslateResult:
    """Original translation logic for a single document or part"""
    translation_config.progress_monitor = pm

    if translation_config.shared_context_cross_split_part.auto_enabled_ocr_workaround:
        translation_config.ocr_workaround = True
        translation_config.skip_scanned_detection = True

    original_pdf_path = translation_config.input_file
    if translation_config.debug:
        doc_input = Document(original_pdf_path)
        logger.debug("debug mode, save decompressed input pdf")
        output_path = translation_config.get_working_file_path(
            "input.decompressed.pdf",
        )
        # Fix null xref in PDF file
        try:
            _ = fix_null_page_content(doc_input)
            fix_filter(doc_input)
            fix_null_xref(doc_input)
        except Exception:
            logger.exception("auto fix failed, please check the pdf file")
        safe_save(doc_input, output_path, expand=True, pretty=True)
        del doc_input

    # Continue with original processing
    temp_pdf_path = translation_config.get_working_file_path("input.pdf")
    doc_pdf2zh = Document(original_pdf_path)
    safe_save(doc_pdf2zh, temp_pdf_path)

    # Fix null xref in PDF file
    invalid_pages = []
    try:
        invalid_pages = fix_null_page_content(doc_pdf2zh)
        fix_filter(doc_pdf2zh)
        fix_null_xref(doc_pdf2zh)
    except Exception:
        logger.exception("auto fix failed, please check the pdf file")

    mediabox_data = fix_media_box(doc_pdf2zh)

    # for page in doc_pdf2zh:
    #     page.insert_font(resfont, None)

    resfont = None
    safe_save(doc_pdf2zh, temp_pdf_path)

    # if not translation_config.skip_scanned_detection and DetectScannedFile(
    #     translation_config
    # ).fast_check(doc_pdf2zh):
    #     if translation_config.auto_enable_ocr_workaround:
    #         logger.warning(
    #             "Fast scanned check hit, Turning on OCR workaround.",
    #         )
    #         translation_config.shared_context_cross_split_part.auto_enabled_ocr_workaround = True
    #         translation_config.ocr_workaround = True
    #         translation_config.skip_scanned_detection = True
    #     else:
    #         logger.warning(
    #             "Fast scanned check hit, Please check the input PDF file.",
    #         )
    #         raise ScannedPDFError("Scanned PDF detected.")

    il_creater = ILCreater(translation_config)
    il_creater.mupdf = doc_pdf2zh
    xml_converter = XMLConverter()
    logger.debug(f"start parse il from {temp_pdf_path}")
    with Path(temp_pdf_path).open("rb") as f:
        start_parse_il(
            f,
            doc_zh=doc_pdf2zh,
            resfont=resfont,
            il_creater=il_creater,
            translation_config=translation_config,
        )
    logger.debug(f"finish parse il from {temp_pdf_path}")
    docs = il_creater.create_il()
    logger.debug(f"finish create il from {temp_pdf_path}")
    del il_creater
    if translation_config.only_include_translated_page and not docs.page:
        return None

    if translation_config.debug:
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("create_il.debug.json"),
        )

    if check_cid_char(docs):
        raise ExtractTextError("The document contains too many CID chars.")

    # Skip all translation processing if only_parse_generate_pdf is enabled
    if translation_config.only_parse_generate_pdf:
        logger.debug("only_parse_generate_pdf enabled, skipping translation processing")
        # Skip directly to PDF generation
        pdf_creater = PDFCreater(temp_pdf_path, docs, translation_config, mediabox_data)
        result = pdf_creater.write(translation_config)
        result.original_pdf_path = translation_config.input_file
        return result

    # Rest of the original translation logic...
    # [Previous implementation of do_translate continues here]

    # 检测是否为扫描文件
    if translation_config.skip_scanned_detection:
        logger.debug("skipping scanned file detection")
    else:
        logger.debug("start detect scanned file")
        DetectScannedFile(translation_config).process(
            docs, temp_pdf_path, mediabox_data
        )
        logger.debug("finish detect scanned file")
        if translation_config.debug:
            xml_converter.write_json(
                docs,
                translation_config.get_working_file_path("detect_scanned_file.json"),
            )

    # Generate layouts for all pages
    logger.debug("start generating layouts")
    docs = LayoutParser(translation_config).process(docs, doc_pdf2zh)
    logger.debug("finish generating layouts")
    if translation_config.debug:
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("layout_generator.json"),
        )

    if translation_config.table_model:
        docs = TableParser(translation_config).process(docs, doc_pdf2zh)
        logger.debug("finish table parser")
        if translation_config.debug:
            xml_converter.write_json(
                docs,
                translation_config.get_working_file_path("table_parser.json"),
            )
    ParagraphFinder(translation_config).process(docs)
    logger.debug(f"finish paragraph finder from {temp_pdf_path}")
    if translation_config.debug:
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("paragraph_finder.json"),
        )
    StylesAndFormulas(translation_config).process(docs)
    logger.debug(f"finish styles and formulas from {temp_pdf_path}")
    if translation_config.debug:
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("styles_and_formulas.json"),
        )

    translate_engine = translation_config.translator
    term_extraction_engine = translation_config.get_term_extraction_translator()

    support_llm_translate = translator_supports_llm(translate_engine)
    support_llm_term_extraction = translator_supports_llm(term_extraction_engine)

    if support_llm_term_extraction and translation_config.auto_extract_glossary:
        AutomaticTermExtractor(term_extraction_engine, translation_config).procress(
            docs
        )

    if not translation_config.skip_translation:
        if support_llm_translate:
            il_translator = ILTranslatorLLMOnly(translate_engine, translation_config)
        else:
            il_translator = ILTranslator(translate_engine, translation_config)

        il_translator.translate(docs)
        del il_translator
        logger.debug(f"finish ILTranslator from {temp_pdf_path}")
    else:
        logger.info("skip ILTranslator")

    if translation_config.debug:
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("il_translated.json"),
        )

    if translation_config.debug:
        AddDebugInformation(translation_config).process(docs)
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("add_debug_information.json"),
        )
    mono_watermark_first_page_doc_bytes = None
    dual_watermark_first_page_doc_bytes = None
    try:
        if translation_config.watermark_output_mode == WatermarkOutputMode.Both:
            mono_watermark_first_page_doc_bytes, dual_watermark_first_page_doc_bytes = (
                generate_first_page_with_watermark(
                    doc_pdf2zh, translation_config, docs, mediabox_data
                )
            )
    except Exception:
        logger.warning(
            "Failed to generate watermark for first page, using no watermark"
        )
        translation_config.watermark_output_mode = WatermarkOutputMode.NoWatermark
        mono_watermark_first_page_doc_bytes = None
        dual_watermark_first_page_doc_bytes = None

    Typesetting(translation_config).typesetting_document(docs)
    logger.debug(f"finish typsetting from {temp_pdf_path}")
    if translation_config.debug:
        xml_converter.write_json(
            docs,
            translation_config.get_working_file_path("typsetting.json"),
        )

    pdf_creater = PDFCreater(temp_pdf_path, docs, translation_config, mediabox_data)
    result = pdf_creater.write(translation_config)
    try:
        if mono_watermark_first_page_doc_bytes:
            mono_watermark_pdf = merge_watermark_doc(
                result.mono_pdf_path,
                mono_watermark_first_page_doc_bytes,
                translation_config,
            )
            result.mono_pdf_path = mono_watermark_pdf
    except Exception:
        result.mono_pdf_path = result.no_watermark_mono_pdf_path
    try:
        if dual_watermark_first_page_doc_bytes:
            dual_watermark_pdf = merge_watermark_doc(
                result.dual_pdf_path,
                dual_watermark_first_page_doc_bytes,
                translation_config,
            )
            result.dual_pdf_path = dual_watermark_pdf
    except Exception:
        result.dual_pdf_path = result.no_watermark_dual_pdf_path

    result.original_pdf_path = translation_config.input_file

    return result


def generate_first_page_with_watermark(
    mupdf: Document,
    translation_config: TranslationConfig,
    doc_il: il_version_1.Document,
    mediabox_data: dict[int, Any] | None = None,
) -> (io.BytesIO, io.BytesIO):
    first_page_doc = Document()
    first_page_doc.insert_pdf(mupdf, from_page=0, to_page=0)

    il_only_first_page_doc = il_version_1.Document()
    il_only_first_page_doc.total_pages = 1
    il_only_first_page_doc.page = [copy.deepcopy(doc_il.page[0])]

    watermarked_config = copy.copy(translation_config)
    watermarked_config.watermark_output_mode = WatermarkOutputMode.Watermarked
    try:
        watermarked_config.progress_monitor.disable = True
        watermarked_temp_pdf_path = watermarked_config.get_working_file_path(
            "watermarked_temp_input.pdf"
        )
        safe_save(first_page_doc, watermarked_temp_pdf_path)

        Typesetting(watermarked_config).typsetting_document(il_only_first_page_doc)
        pdf_creater = PDFCreater(
            watermarked_temp_pdf_path.as_posix(),
            il_only_first_page_doc,
            watermarked_config,
            mediabox_data,
        )
        result = pdf_creater.write(watermarked_config)
        mono_pdf_bytes = None
        dual_pdf_bytes = None
        if result.mono_pdf_path:
            mono_pdf_bytes = io.BytesIO()
            with Path(result.mono_pdf_path).open("rb") as f:
                mono_pdf_bytes.write(f.read())
            result.mono_pdf_path.unlink()
            mono_pdf_bytes.seek(0)

        if result.dual_pdf_path:
            dual_pdf_bytes = io.BytesIO()
            with Path(result.dual_pdf_path).open("rb") as f:
                dual_pdf_bytes.write(f.read())
            result.dual_pdf_path.unlink()
            dual_pdf_bytes.seek(0)

        return mono_pdf_bytes, dual_pdf_bytes
    finally:
        watermarked_config.progress_monitor.disable = False


def merge_watermark_doc(
    no_watermark_pdf_path: pathlib.PosixPath,
    watermark_first_page_pdf_bytes: io.BytesIO,
    translation_config: TranslationConfig,
) -> pathlib.PosixPath:
    if not no_watermark_pdf_path.exists():
        raise FileNotFoundError(
            f"no_watermark_pdf_path not found: {no_watermark_pdf_path}"
        )
    if not watermark_first_page_pdf_bytes:
        raise FileNotFoundError(
            f"watermark_first_page_pdf_bytes not found: {watermark_first_page_pdf_bytes}"
        )

    no_watermark_pdf = Document(no_watermark_pdf_path.as_posix())
    no_watermark_pdf.delete_page(0)

    watermark_first_page_pdf = Document("pdf", watermark_first_page_pdf_bytes)
    no_watermark_pdf.insert_pdf(
        watermark_first_page_pdf, from_page=0, to_page=0, start_at=0
    )

    new_save_path = no_watermark_pdf_path.with_name(
        no_watermark_pdf_path.name.replace(".no_watermark", "")
    )

    PDFCreater.save_pdf_with_timeout(
        no_watermark_pdf,
        new_save_path.as_posix(),
        translation_config=translation_config,
        clean=not translation_config.skip_clean,
    )
    return new_save_path


def download_font_assets():
    warmup()


def create_cache_folder():
    try:
        logger.debug(f"create cache folder at {CACHE_FOLDER}")
        Path(CACHE_FOLDER).mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.critical(
            f"Failed to create cache folder at {CACHE_FOLDER}",
            exc_info=True,
        )
        exit(1)


def init():
    create_cache_folder()
