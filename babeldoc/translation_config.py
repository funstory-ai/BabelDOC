import enum
import logging
import shutil
import tempfile
import threading
from pathlib import Path

from babeldoc.const import CACHE_FOLDER
from babeldoc.document_il.translator.translator import BaseTranslator
from babeldoc.docvision.doclayout import DocLayoutModel
from babeldoc.progress_monitor import ProgressMonitor
from babeldoc.split_manager import BaseSplitStrategy
from babeldoc.split_manager import PageCountStrategy

logger = logging.getLogger(__name__)


class WatermarkOutputMode(enum.Enum):
    Watermarked = "watermarked"
    NoWatermark = "no_watermark"
    Both = "both"


class SharedContextCrossSplitPart:
    def __init__(self):
        self.first_paragraph = None
        self.recent_title_paragraph = None


class TranslationConfig:
    @staticmethod
    def create_max_pages_per_part_split_strategy(max_pages_per_part: int):
        return PageCountStrategy(max_pages_per_part)

    def __init__(
        self,
        translator: BaseTranslator,
        input_file: str | Path,
        lang_in: str,
        lang_out: str,
        doc_layout_model: DocLayoutModel,
        # for backward compatibility
        font: str | Path | None = None,
        pages: str | None = None,
        output_dir: str | Path | None = None,
        debug: bool = False,
        working_dir: str | Path | None = None,
        no_dual: bool = False,
        no_mono: bool = False,
        formular_font_pattern: str | None = None,
        formular_char_pattern: str | None = None,
        qps: int = 1,
        split_short_lines: bool = False,
        short_line_split_factor: float = 0.8,
        use_rich_pbar: bool = True,
        progress_monitor: ProgressMonitor | None = None,
        skip_clean: bool = False,
        dual_translate_first: bool = False,
        disable_rich_text_translate: bool = False,
        enhance_compatibility: bool = False,
        report_interval: float = 0.1,
        min_text_length: int = 5,
        use_side_by_side_dual: bool = True,  # Deprecated: 是否使用拼版式双语 PDF（并排显示原文和译文）向下兼容选项，已停用。
        use_alternating_pages_dual: bool = False,
        watermark_output_mode: WatermarkOutputMode = WatermarkOutputMode.Watermarked,
        # Add split-related parameters
        split_strategy: BaseSplitStrategy | None = None,
        table_model=None,
        show_char_box: bool = False,
        skip_scanned_detection: bool = False,
        ocr_workaround: bool = False,
    ):
        self.translator = translator

        self.input_file = input_file
        self.lang_in = lang_in
        self.lang_out = lang_out
        # just ignore font
        self.font = None

        self.pages = pages
        self.page_ranges = self.parse_pages(pages) if pages else None
        self.debug = debug
        self.watermark_output_mode = watermark_output_mode

        self.output_dir = output_dir
        self.working_dir = working_dir
        self.no_dual = no_dual
        self.no_mono = no_mono

        self.formular_font_pattern = formular_font_pattern
        self.formular_char_pattern = formular_char_pattern
        self.qps = qps
        self.split_short_lines = split_short_lines

        self.short_line_split_factor = short_line_split_factor
        self.use_rich_pbar = use_rich_pbar
        self.progress_monitor = progress_monitor
        self.doc_layout_model = doc_layout_model

        self.skip_clean = skip_clean or enhance_compatibility
        self.skip_scanned_detection = skip_scanned_detection

        self.dual_translate_first = dual_translate_first or enhance_compatibility
        self.disable_rich_text_translate = (
            disable_rich_text_translate or enhance_compatibility
        )

        self.report_interval = report_interval
        self.min_text_length = min_text_length
        self.use_alternating_pages_dual = use_alternating_pages_dual
        self.ocr_workaround = ocr_workaround

        if self.ocr_workaround:
            self.skip_scanned_detection = True

        # for backward compatibility
        if use_side_by_side_dual is False and use_alternating_pages_dual is False:
            self.use_alternating_pages_dual = True

        if progress_monitor and progress_monitor.cancel_event is None:
            progress_monitor.cancel_event = threading.Event()

        if working_dir is None:
            if debug:
                working_dir = Path(CACHE_FOLDER) / "working" / Path(input_file).stem
                self._is_temp_dir = False
            else:
                working_dir = tempfile.mkdtemp()
                self._is_temp_dir = True
        self.working_dir = working_dir

        Path(working_dir).mkdir(parents=True, exist_ok=True)

        if output_dir is None:
            output_dir = Path.cwd()
        self.output_dir = output_dir

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if not doc_layout_model:
            doc_layout_model = DocLayoutModel.load_available()
        self.doc_layout_model = doc_layout_model

        self.shared_context_cross_split_part = SharedContextCrossSplitPart()

        # Initialize split-related attributes
        self.split_strategy = split_strategy

        # Create a unique working directory for each part
        self._part_working_dirs: dict[int, Path] = {}
        self._part_output_dirs: dict[int, Path] = {}

        self.table_model = table_model
        self.show_char_box = show_char_box

    def parse_pages(self, pages_str: str | None) -> list[tuple[int, int]] | None:
        """解析页码字符串，返回页码范围列表

        Args:
            pages_str: 形如 "1-,2,-3,4" 的页码字符串

        Returns:
            包含 (start, end) 元组的列表，其中 -1 表示无限制
        """
        if not pages_str:
            return None

        ranges: list[tuple[int, int]] = []
        for part in pages_str.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                start_as_int = int(start) if start else 1
                end_as_int = int(end) if end else -1
                ranges.append((start_as_int, end_as_int))
            else:
                page = int(part)
                ranges.append((page, page))
        return ranges

    def should_translate_page(self, page_number: int) -> bool:
        """判断指定页码是否需要翻译
        Args:
            page_number: 页码
        Returns:
            是否需要翻译该页
        """
        if isinstance(self.page_ranges, list) and len(self.page_ranges) == 0:
            return False
        if not self.page_ranges:
            return True

        for start, end in self.page_ranges:
            if start <= page_number and (end == -1 or page_number <= end):
                return True
        return False

    def get_output_file_path(self, filename: str) -> Path:
        return Path(self.output_dir) / filename

    def get_working_file_path(self, filename: str) -> Path:
        return Path(self.working_dir) / filename

    def get_part_working_dir(self, part_index: int) -> Path:
        """Get working directory for a specific part"""
        if part_index not in self._part_working_dirs:
            if self.working_dir:
                part_dir = Path(self.working_dir) / f"part_{part_index}"
            else:
                part_dir = Path(tempfile.mkdtemp()) / f"part_{part_index}"
            part_dir.mkdir(parents=True, exist_ok=True)
            self._part_working_dirs[part_index] = part_dir
        return self._part_working_dirs[part_index]

    def get_part_output_dir(self, part_index: int) -> Path:
        """Get output directory for a specific part"""
        if part_index not in self._part_output_dirs:
            part_dir = Path(self.working_dir) / f"part_{part_index}_output"
            part_dir.mkdir(parents=True, exist_ok=True)
            self._part_output_dirs[part_index] = part_dir
        return self._part_output_dirs[part_index]

    def cleanup_part_output_dir(self, part_index: int):
        """Clean up output directory for a specific part"""
        if part_index in self._part_output_dirs:
            part_dir = self._part_output_dirs[part_index]
            if part_dir.exists():
                shutil.rmtree(part_dir)
            del self._part_output_dirs[part_index]

    def cleanup_part_working_dir(self, part_index: int):
        """Clean up working directory for a specific part"""
        if part_index in self._part_working_dirs:
            part_dir = self._part_working_dirs[part_index]
            if part_dir.exists():
                shutil.rmtree(part_dir)
            del self._part_working_dirs[part_index]

    def cleanup_temp_files(self):
        """Clean up all temporary files including part working directories"""
        try:
            for part_index in list(self._part_working_dirs.keys()):
                self.cleanup_part_working_dir(part_index)
            if self._is_temp_dir:
                logger.info(f"cleanup temp files: {self.working_dir}")
                shutil.rmtree(self.working_dir)
        except Exception:
            logger.exception("Error cleaning up temporary files")

    def raise_if_cancelled(self):
        if self.progress_monitor is not None:
            self.progress_monitor.raise_if_cancelled()

    def cancel_translation(self):
        if self.progress_monitor is not None:
            self.progress_monitor.cancel()


class TranslateResult:
    original_pdf_path: str
    total_seconds: float
    mono_pdf_path: str | None
    dual_pdf_path: str | None
    no_watermark_mono_pdf_path: str | None
    no_watermark_dual_pdf_path: str | None
    peak_memory_usage: int | None
    no_mono: bool
    no_dual: bool

    def __init__(self, mono_pdf_path: str | None, dual_pdf_path: str | None, no_mono: bool = False, no_dual: bool = False):
        self.mono_pdf_path = mono_pdf_path
        self.dual_pdf_path = dual_pdf_path
        self.no_mono = no_mono
        self.no_dual = no_dual

        # For compatibility considerations, if only a non-watermarked PDF is generated,
        # the values of mono_pdf_path and no_watermark_mono_pdf_path are the same.
        self.no_watermark_mono_pdf_path = mono_pdf_path
        self.no_watermark_dual_pdf_path = dual_pdf_path

    def __str__(self):
        """Return a human-readable string representation of the translation result."""
        result = []
        if hasattr(self, "original_pdf_path") and self.original_pdf_path:
            result.append(f"\tOriginal PDF: {self.original_pdf_path}")

        if hasattr(self, "total_seconds") and self.total_seconds:
            result.append(f"\tTotal time: {self.total_seconds:.2f} seconds")

        if self.mono_pdf_path and not self.no_mono:
            result.append(f"\tMonolingual PDF: {self.mono_pdf_path}")

        if self.dual_pdf_path and not self.no_dual:
            result.append(f"\tDual-language PDF: {self.dual_pdf_path}")

        if (
            hasattr(self, "no_watermark_mono_pdf_path")
            and self.no_watermark_mono_pdf_path
            and self.no_watermark_mono_pdf_path != self.mono_pdf_path
            and not self.no_mono
        ):
            result.append(
                f"\tNo-watermark Monolingual PDF: {self.no_watermark_mono_pdf_path}"
            )

        if (
            hasattr(self, "no_watermark_dual_pdf_path")
            and self.no_watermark_dual_pdf_path
            and self.no_watermark_dual_pdf_path != self.dual_pdf_path
            and not self.no_dual
        ):
            result.append(
                f"\tNo-watermark Dual-language PDF: {self.no_watermark_dual_pdf_path}"
            )

        if hasattr(self, "peak_memory_usage") and self.peak_memory_usage:
            result.append(f"\tPeak memory usage: {self.peak_memory_usage} MB")

        if result:
            result.insert(0, "Translation results:")

        return "\n".join(result) if result else "No translation results available"
