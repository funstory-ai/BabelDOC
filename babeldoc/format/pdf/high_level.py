import asyncio
import copy
import hashlib
import json
import logging
import re
import shutil
import threading
import time
from asyncio import CancelledError
from pathlib import Path

import pymupdf
from pymupdf import Document

from babeldoc import asynchronize
from babeldoc.assets.assets import warmup
from babeldoc.babeldoc_exception.BabelDOCException import (
    InputFileGeneratedByBabelDOCError,
)
from babeldoc.const import CACHE_FOLDER
from babeldoc.const import WATERMARK_VERSION
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.backend.pdf_creater import SAVE_PDF_STAGE_NAME
from babeldoc.format.pdf.document_il.backend.pdf_creater import SUBSET_FONT_STAGE_NAME
from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater
from babeldoc.format.pdf.document_il.backend.pdf_creater import reproduce_cmap
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.result_merger import ResultMerger
from babeldoc.format.pdf.split_manager import SplitManager
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.format.pdf.translation_config import TranslationConfig
from babeldoc.format.pdf.translation_config import WatermarkOutputMode
from babeldoc.progress_monitor import ProgressMonitor
from babeldoc.rust_callback_bridge import CallbackRun
from babeldoc.rust_debug_service import RustDebugSerializerService
from babeldoc.rust_layout_service import LayoutService
from babeldoc.rust_run_context import TranslatorProbeService
from babeldoc.rust_run_context import create_document_run_context
from babeldoc.rust_scan_service import ScanFileService
from babeldoc.rust_translation_service import RustTranslatorService
from babeldoc.rust_writer_service import WriterFileService
from babeldoc.utils import memory

logger = logging.getLogger(__name__)

_DETECT_SCANNED_STAGE = "DetectScannedFile"
_LAYOUT_STAGE = "Parse Page Layout"
_TABLE_STAGE = "Parse Table"
_PARAGRAPH_STAGE = "Parse Paragraphs"
_FORMULA_STAGE = "Parse Formulas and Styles"
_TERM_STAGE = "Automatic Term Extraction"
_TRANSLATION_STAGE = "Translate Paragraphs"
_TYPESETTING_STAGE = "Typesetting"
_DRAWING_STAGE = "Generate drawing instructions"

_DETERMINISTIC_PDF_ID = (
    "[<424142454c444f432d44455445524d31><424142454c444f432d44455445524d32>]"
)

TRANSLATE_STAGES = [
    ("Parse PDF and Create Intermediate Representation", 14.12),
    (_DETECT_SCANNED_STAGE, 2.45),
    (_LAYOUT_STAGE, 14.03),
    (_TABLE_STAGE, 1.0),
    (_PARAGRAPH_STAGE, 6.26),
    (_FORMULA_STAGE, 1.66),
    (_TERM_STAGE, 30.0),
    (_TRANSLATION_STAGE, 46.96),
    (_TYPESETTING_STAGE, 4.71),
    (FontMapper.stage_name, 0.61),  # Add Fonts
    (_DRAWING_STAGE, 1.96),
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


def rebuild_pdf_by_inserting_pages(
    pdf_path: str | Path,
    output_pdf_path: str | Path,
) -> None:
    output_pdf_path = Path(output_pdf_path)
    rebuilt_pdf_path = output_pdf_path.with_name(
        f"{output_pdf_path.stem}.rebuilt{output_pdf_path.suffix}"
    )
    if rebuilt_pdf_path.exists():
        rebuilt_pdf_path.unlink()

    source_doc = Document(pdf_path)
    rebuilt_doc = Document()
    try:
        rebuilt_doc.insert_pdf(source_doc)
        rebuilt_doc.save(rebuilt_pdf_path)
    finally:
        rebuilt_doc.close()
        source_doc.close()

    rebuilt_pdf_path.replace(output_pdf_path)


def open_pdf_with_save_fallback(
    pdf_path: str | Path,
    output_pdf_path: str | Path,
) -> Document:
    doc_pdf = Document(pdf_path)
    try:
        safe_save(doc_pdf, output_pdf_path)
        return doc_pdf
    except Exception:
        doc_pdf.close()
        rebuild_pdf_by_inserting_pages(pdf_path, output_pdf_path)
        return Document(output_pdf_path)


def save_pdf_with_same_path_fallback(
    doc_pdf: Document,
    output_pdf_path: str | Path,
) -> Document:
    output_pdf_path = Path(output_pdf_path)
    doc_name = Path(getattr(doc_pdf, "name", "") or "")
    if doc_name.resolve() != output_pdf_path.resolve():
        safe_save(doc_pdf, output_pdf_path)
        return doc_pdf

    saved_pdf_path = output_pdf_path.with_name(
        f"{output_pdf_path.stem}.saved{output_pdf_path.suffix}"
    )
    if saved_pdf_path.exists():
        saved_pdf_path.unlink()
    try:
        safe_save(doc_pdf, saved_pdf_path)
    finally:
        doc_pdf.close()
    saved_pdf_path.replace(output_pdf_path)
    return Document(output_pdf_path)


def check_metadata(pdf: Document):
    meta = pdf.metadata
    if not meta:
        return
    producer = meta.get("producer", None)
    if (
        producer
        and "BabelDOC" in producer
        and "Translation_generated_by_AI,please_carefully_discern" in producer
    ):
        raise InputFileGeneratedByBabelDOCError(
            "Input file is generated by BabelDOC, Cannot translate files that have already been translated."
        )


def add_metadata(
    translate_result: TranslateResult, translate_config: TranslationConfig
):
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
        meta = pdf.metadata
        if not meta:
            meta = {}
        creator = meta.get("creator", None)
        producer = meta.get("producer", None)
        if producer:
            if not creator:
                creator = producer
            else:
                creator += f", {producer}"

        producer_timestamp = (
            0.0 if translate_config.deterministic_output else time.time()
        )
        translated_by = f"BabelDOC{WATERMARK_VERSION}_{producer_timestamp}_Translation_generated_by_AI,please_carefully_discern"
        if translate_config.metadata_extra_data:
            translated_by += f"_{translate_config.metadata_extra_data}"
        meta["producer"] = translated_by
        meta["creator"] = creator

        for k, v in meta.items():
            if v:
                # 使用正则替换掉 surrogate 范围内的字符
                meta[k] = re.sub(r"[\uD800-\uDFFF]", "", v)

        pdf.set_metadata(meta)
        safe_save(pdf, temp_path)
        shutil.move(temp_path, path)


def finalize_deterministic_pdf_outputs(translate_result: TranslateResult) -> None:
    processed = []
    for attr in (
        "mono_pdf_path",
        "dual_pdf_path",
        "no_watermark_mono_pdf_path",
        "no_watermark_dual_pdf_path",
    ):
        output_path = getattr(translate_result, attr)
        if not output_path:
            continue
        output_path = Path(output_path)
        if output_path in processed:
            continue
        processed.append(output_path)

        temp_path = output_path.with_name(
            f"{output_path.stem}.deterministic.tmp{output_path.suffix}"
        )
        if temp_path.exists():
            temp_path.unlink()
        pdf = pymupdf.open(output_path)
        try:
            pdf.xref_set_key(-1, "ID", _DETERMINISTIC_PDF_ID)
            pdf.save(temp_path, no_new_id=True)
        finally:
            pdf.close()
        shutil.move(temp_path, output_path)


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


def translate(translation_config: TranslationConfig) -> TranslateResult:
    _require_stage_capture_feature(translation_config)
    with ProgressMonitor(get_translation_stage(translation_config)) as pm:
        return do_translate(pm, translation_config)


def _require_stage_capture_feature(translation_config: TranslationConfig) -> None:
    if not translation_config.capture_stages:
        return
    from babeldoc import _rust_il

    if not _rust_il.stage_capture_enabled():
        raise RuntimeError(
            "stage capture was requested, but this Rust extension was built without the `stage-capture` Cargo feature"
        )


def get_translation_stage(
    translation_config: TranslationConfig,
) -> list[tuple[str, float]]:
    result = copy.deepcopy(TRANSLATE_STAGES)
    should_remove = []

    # If only parsing and generating PDF, skip all translation-related stages
    if translation_config.only_parse_generate_pdf:
        should_remove.extend(
            [
                _DETECT_SCANNED_STAGE,
                _LAYOUT_STAGE,
                _TABLE_STAGE,
                _PARAGRAPH_STAGE,
                _FORMULA_STAGE,
                _TERM_STAGE,
                _TRANSLATION_STAGE,
                _TYPESETTING_STAGE,
            ]
        )
    else:
        # Original logic for selective removal
        if not translation_config.table_model:
            should_remove.append(_TABLE_STAGE)
        if translation_config.skip_scanned_detection:
            should_remove.append(_DETECT_SCANNED_STAGE)
        if not translation_config.auto_extract_glossary:
            should_remove.append(_TERM_STAGE)
        if translation_config.skip_translation:
            should_remove.append(_TRANSLATION_STAGE)

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
    _require_stage_capture_feature(translation_config)
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
        self.last_pss_check_time = None

    def __enter__(self):
        """Start memory monitoring."""
        self.stop_event = threading.Event()
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory_usage, daemon=True
        )
        self.monitor_thread.start()
        logger.debug("Memory monitoring started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop monitoring and log peak memory usage."""
        if not self.monitor_thread:
            return

        self.stop_event.set()
        self.monitor_thread.join(timeout=2.0)
        logger.info(f"Peak memory usage: {self.peak_memory_usage:.2f} MB")

    def _monitor_memory_usage(self):
        """Background thread that periodically checks memory usage."""
        while not self.stop_event.is_set():
            try:
                # Use throttled memory check with 2-second PSS throttle
                total_memory, self.last_pss_check_time = (
                    memory.get_memory_usage_with_throttle(
                        include_children=True,
                        prefer_pss=True,
                        last_pss_check_time=self.last_pss_check_time,
                        pss_throttle_seconds=2.0,
                    )
                )

                # Convert to MB for better readability
                total_memory_mb = total_memory / (1024 * 1024)
                if total_memory_mb > self.peak_memory_usage:
                    self.peak_memory_usage = total_memory_mb
            except Exception as e:
                logger.warning(f"Error monitoring memory: {e}")

            self.stop_event.wait(self.interval)

    def get_peek_memory_psutil(self):
        """Get peak memory usage using psutil (for backwards compatibility)."""
        return memory.get_memory_usage_bytes(include_children=True, prefer_pss=True)


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
        if len(contents) > 1:
            page_streams = [doc.xref_stream(i) for i in contents]
            r = doc.get_new_xref()
            doc.update_object(r, "<<>>")
            doc.update_stream(r, b" ".join(page_streams))
            doc.xref_set_key(page.xref, "Contents", f"{r} 0 R")
    return
    # skip rotate for now
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
                update_page_bbox(doc, page, page.artbox * m1, "ArtBox")
                update_page_bbox(doc, page, page.bleedbox * m1, "BleedBox")
                update_page_bbox(doc, page, page.mediabox * m1, "MediaBox")
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
    _require_stage_capture_feature(translation_config)
    run_context = None
    rust_module = None
    rust_logging_started = False
    try:
        translation_config.progress_monitor = pm
        original_pdf_path = translation_config.input_file
        logger.info(f"start to translate: {original_pdf_path}")
        try:
            check_metadata(Document(original_pdf_path))
        except InputFileGeneratedByBabelDOCError as e:
            logger.error(
                f"input file {original_pdf_path} is generated by BabelDOC, Cannot translate files that have already been translated."
            )
            raise e
        except Exception as e:
            logger.warning(f"Error in check metadata, continue: {e}")
        from babeldoc import _rust_il

        rust_module = _rust_il
        rust_module.acquire_logging_run()
        rust_logging_started = True
        run_context = create_document_run_context(translation_config)
        start_time = time.time()
        peak_memory_usage = 0
        with MemoryMonitor() as memory_monitor:
            # Check if split translation is enabled
            if not translation_config.split_strategy:
                result = _do_translate_single(pm, translation_config, run_context)
            else:
                # Initialize split manager and determine split points
                split_manager = SplitManager(translation_config)
                split_points = split_manager.determine_split_points(translation_config)

                if not split_points:
                    logger.warning(
                        "No split points determined, falling back to single translation"
                    )
                    result = _do_translate_single(pm, translation_config, run_context)
                else:
                    logger.info(f"Split points determined: {len(split_points)} parts")

                    if len(split_points) == 1:
                        logger.info("Only one part, use single translation")
                        result = _do_translate_single(
                            pm, translation_config, run_context
                        )
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
                                part_config._capture_part_index = i
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
                                    run_context,
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
        # Populate aggregate valid text statistics into result
        try:
            sc = translation_config.shared_context_cross_split_part
            result.total_valid_character_count = getattr(
                sc, "valid_char_count_total", 0
            )
            token_total = getattr(sc, "total_valid_text_token_count", None)
            result.total_valid_text_token_count = (
                token_total if isinstance(token_total, int) else 0
            )
        except Exception as e:
            logger.warning("Failed to populate valid text statistics: %s", e)
            try:
                result.total_valid_character_count = 0
                result.total_valid_text_token_count = 0
            except Exception:
                pass
        result.original_pdf_path = translation_config.input_file
        result.peak_memory_usage = peak_memory_usage

        fix_cmap(result, translation_config)
        add_metadata(result, translation_config)
        try:
            migrate_toc(translation_config, result)
        except Exception as e:
            logger.error(
                f"Failed to migrate TOC from {translation_config.input_file}: {e}"
            )
        if translation_config.deterministic_output:
            finalize_deterministic_pdf_outputs(result)
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
        try:
            if run_context is not None:
                run_context.close()
        finally:
            try:
                if rust_logging_started:
                    try:
                        rust_module.flush_logging_run()
                    finally:
                        rust_module.stop_logging_run()
            finally:
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


def _rust_stage_configs(config, run_context):
    style_fields = {
        name: getattr(config, name)
        for name in (
            "skip_formula_offset_calculation",
            "ocr_workaround",
            "remove_non_formula_lines",
            "figure_table_protection_threshold",
            "non_formula_line_iou_threshold",
            "formular_font_pattern",
            "formular_char_pattern",
        )
    }
    paragraph = {
        "debug": config.debug,
        "ocr_workaround": config.ocr_workaround,
        "merge_alternating_line_numbers": config.merge_alternating_line_numbers,
        "split_short_lines": config.split_short_lines,
        "short_line_split_factor": config.short_line_split_factor,
        "formular_font_pattern": config.formular_font_pattern,
    }
    styles = {
        "config_fields": style_fields,
    }
    typesetting = {
        "config_fields": {"lang_out": config.lang_out},
        "font_source_styles": run_context.font_source_styles,
        "progress_monitor_present": True,
    }
    return {"paragraph": paragraph, "styles": styles, "typesetting": typesetting}


def _rust_writer_inputs(config, run_context, temp_pdf_path):
    from babeldoc.assets.embedding_assets_metadata import EMBEDDING_FONT_METADATA

    basename = Path(config.input_file).stem
    debug_suffix = ".debug" if config.debug else ""
    if config.watermark_output_mode != WatermarkOutputMode.Watermarked:
        debug_suffix += ".no_watermark"
    mono_path = (
        None
        if config.no_mono
        else str(
            config.get_output_file_path(
                f"{basename}{debug_suffix}.{config.lang_out}.mono.pdf"
            )
        )
    )
    dual_path = (
        None
        if config.no_dual
        else str(
            config.get_output_file_path(
                f"{basename}{debug_suffix}.{config.lang_out}.dual.pdf"
            )
        )
    )
    writer_config = {
        "source": str(temp_pdf_path),
        "translated": str(config.get_working_file_path("translated.pdf")),
        "debug_original": str(config.get_working_file_path("debug_original.pdf")),
        "mono": mono_path,
        "dual": dual_path,
        "debug": config.debug,
        "ocr_workaround": config.ocr_workaround,
        "skip_clean": config.skip_clean,
        "skip_form_render": config.skip_form_render,
        "skip_curve_render": config.skip_curve_render,
        "only_include_translated_page": config.only_include_translated_page,
        "use_alternating_pages_dual": config.use_alternating_pages_dual,
        "dual_translate_first": config.dual_translate_first,
    }
    font_ids = list(
        dict.fromkeys(
            font_id
            for group in ("normal", "script", "fallback", "base")
            for font_id in run_context.font_family[group]
        )
    )
    writer_assets = [
        {
            "id": font_id,
            "metadata": {
                key: EMBEDDING_FONT_METADATA[font_id][key]
                for key in ("font_name", "ascent", "descent", "encoding_length")
            },
        }
        for font_id in font_ids
    ]
    writer_assets.append(
        {
            "id": "base",
            "metadata": {
                key: EMBEDDING_FONT_METADATA[run_context.font_family["base"][0]][key]
                for key in ("font_name", "ascent", "descent", "encoding_length")
            },
        }
    )
    paths = {
        "scan": str(config.get_working_file_path("detect_scanned_file.json")),
        "layout": str(config.get_working_file_path("layout_generator.json")),
        "table": str(config.get_working_file_path("table_parser.json")),
        "paragraph": str(config.get_working_file_path("paragraph_finder.json")),
        "styles": str(config.get_working_file_path("styles_and_formulas.json")),
        "translation": str(config.get_working_file_path("il_translated.json")),
        "debug_information": str(
            config.get_working_file_path("add_debug_information.json")
        ),
        "typesetting": str(config.get_working_file_path("typsetting.json")),
    }
    return writer_config, writer_assets, paths


def _do_translate_single(
    pm: ProgressMonitor,
    translation_config: TranslationConfig,
    run_context,
) -> TranslateResult:
    """Parse and retain native IR through the Rust production entry."""
    _require_stage_capture_feature(translation_config)
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
    doc_pdf2zh = open_pdf_with_save_fallback(original_pdf_path, temp_pdf_path)

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

    doc_pdf2zh = save_pdf_with_same_path_fallback(doc_pdf2zh, temp_pdf_path)

    try:
        from babeldoc import _rust_il

        cancel_event = pm.cancel_event or threading.Event()
        with CallbackRun({}, cancel_event, _rust_il, record_requests=False) as run:
            pm.raise_if_cancelled()
            parser_config = {
                "disable_parser_resource_limits": translation_config.disable_parser_resource_limits,
                "pages": [
                    number
                    for number in range(doc_pdf2zh.page_count)
                    if translation_config.should_translate_page(number + 1)
                ],
                "enable_graphic_element_process": translation_config.enable_graphic_element_process,
                "ocr_workaround": translation_config.ocr_workaround,
                "visual_roundtrip_mode": getattr(
                    translation_config,
                    "visual_roundtrip_mode",
                    False,
                ),
            }
            capture_path = translation_config.get_stage_capture_file_path("parser")
            debug_path = (
                translation_config.get_working_file_path("create_il.debug.json")
                if translation_config.debug
                else None
            )
            logger.debug(f"start parse il from {temp_pdf_path}")
            rust_document = _rust_il.parse_prepared_pdf_product(
                str(temp_pdf_path),
                json.dumps(parser_config),
                run_context.font_mapper,
                run_context.shared_context,
                run.flag,
                pm,
                only_include_translated_page=translation_config.only_include_translated_page,
                capture_path=None if capture_path is None else str(capture_path),
                debug_path=None if debug_path is None else str(debug_path),
            )
            pm.raise_if_cancelled()
            logger.debug(f"finish parse il from {temp_pdf_path}")
            if rust_document is None:
                return None
            stage_config = (
                {}
                if translation_config.only_parse_generate_pdf
                else _rust_stage_configs(translation_config, run_context)
            )
            writer_config, writer_assets, debug_paths = _rust_writer_inputs(
                translation_config, run_context, temp_pdf_path
            )
            capture_paths = {}
            if translation_config.capture_stages:
                for stage in (
                    "scan",
                    "layout",
                    "paragraph",
                    "styles",
                    "translation",
                    "typesetting",
                ):
                    path = translation_config.get_stage_capture_file_path(stage)
                    if path is not None:
                        capture_paths[stage] = str(path)
            services = {}
            try:
                translator = translation_config.translator
                term_translator = translation_config.get_term_extraction_translator()
                translation_service = RustTranslatorService(
                    translator, translation_config, run_context.shared_context
                )
                run_context.dispatcher_a.service = translation_service
                run_context.dispatcher_b.service = translation_service
                term_service = RustTranslatorService(
                    term_translator, translation_config, run_context.shared_context
                )
                services["scan"] = (ScanFileService(), 1)
                services["layout"] = (
                    LayoutService(
                        translation_config, doc_pdf2zh, run_context.font_mapper
                    ),
                    1,
                )
                services["translation"] = (
                    translation_service,
                    translation_config.pool_max_workers,
                )
                services["term"] = (
                    term_service,
                    translation_config.term_pool_max_workers,
                )
                services["writer"] = (
                    WriterFileService(translation_config, mediabox_data, temp_pdf_path),
                    1,
                )
                services["debug"] = (RustDebugSerializerService(), 1)
                services["probe_terms"] = (
                    TranslatorProbeService(term_translator, translator_supports_llm),
                    1,
                )
                services["probe_translate"] = (
                    TranslatorProbeService(translator, translator_supports_llm),
                    1,
                )
                run.add_services(services, record_requests=False)
                run_context.dispatcher_a.cancel_flag = run.flag
                run_context.dispatcher_b.cancel_flag = run.flag
                services["layout"][0].flag = run.flag
                writer_service = services["writer"][0]
                basename = Path(translation_config.input_file).stem
                debug_suffix = ".debug" if translation_config.debug else ""
                if (
                    translation_config.watermark_output_mode
                    != WatermarkOutputMode.Watermarked
                ):
                    debug_suffix += ".no_watermark"
                if translation_config.save_auto_extracted_glossary:
                    writer_service.glossary_path = translation_config.get_output_file_path(
                        f"{basename}{debug_suffix}.{translation_config.lang_out}.glossary.csv"
                    )
                mono_path, dual_path, glossary_path = rust_document.run_part(
                    run_context.native,
                    translation_config,
                    run.executors,
                    json.dumps(stage_config),
                    json.dumps(debug_paths),
                    json.dumps(writer_config),
                    json.dumps(writer_assets),
                    run.flag,
                    pm,
                    translation_config.capture_stages,
                    json.dumps(capture_paths),
                )
            finally:
                # The Rust stages normally request these cleanups themselves. Keep
                # service teardown idempotent for failures before those callbacks.
                for service_name, method in (
                    ("scan", "close"),
                    ("layout", "close"),
                    ("layout", "close_process_pool"),
                    ("writer", "close"),
                ):
                    cleanup = getattr(
                        services.get(service_name, (None, 0))[0], method, None
                    )
                    if cleanup is not None:
                        cleanup()

        result = TranslateResult(
            mono_pdf_path=None if mono_path is None else Path(mono_path),
            dual_pdf_path=None if dual_path is None else Path(dual_path),
            auto_extracted_glossary_path=(
                None if glossary_path is None else Path(glossary_path)
            ),
        )
        # only-parse keeps 0.6.4's single no-watermark group even in Both mode.
        if not translation_config.only_parse_generate_pdf:
            _copy_both_mode_outputs(result, translation_config)
        result.original_pdf_path = translation_config.input_file
        return result

    finally:
        if not doc_pdf2zh.is_closed:
            doc_pdf2zh.close()


def _copy_both_mode_outputs(
    result: TranslateResult, translation_config: TranslationConfig
) -> None:
    if translation_config.watermark_output_mode != WatermarkOutputMode.Both:
        return

    for no_watermark_attr, output_attr in (
        ("no_watermark_mono_pdf_path", "mono_pdf_path"),
        ("no_watermark_dual_pdf_path", "dual_pdf_path"),
    ):
        no_watermark_path = getattr(result, no_watermark_attr)
        if no_watermark_path and no_watermark_path.exists():
            output_path = no_watermark_path.with_name(
                no_watermark_path.name.replace(".no_watermark", "")
            )
            shutil.copyfile(no_watermark_path, output_path)
            setattr(result, output_attr, output_path)


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
