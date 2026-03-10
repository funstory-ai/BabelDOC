import logging
from pathlib import Path

from pymupdf import Document

from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater
from babeldoc.format.pdf.translation_config import TranslateResult
from babeldoc.format.pdf.translation_config import TranslationConfig

logger = logging.getLogger(__name__)


class ResultMerger:
    """Handles merging of split translation results"""

    def __init__(self, translation_config: TranslationConfig):
        self.config = translation_config

    def merge_results(
        self, results: dict[int, TranslateResult | None]
    ) -> TranslateResult:
        """Merge multiple translation results into one"""
        if not results:
            raise ValueError("No results to merge")

        basename = Path(self.config.input_file).stem
        debug_suffix = ".debug" if self.config.debug else ""

        mono_file_name = f"{basename}{debug_suffix}.{self.config.lang_out}.mono.pdf"
        dual_file_name = f"{basename}{debug_suffix}.{self.config.lang_out}.dual.pdf"

        debug_suffix += ".no_watermark"

        mono_file_name_no_watermark = (
            f"{basename}{debug_suffix}.{self.config.lang_out}.mono.pdf"
        )
        dual_file_name_no_watermark = (
            f"{basename}{debug_suffix}.{self.config.lang_out}.dual.pdf"
        )
        results = {k: v for k, v in results.items() if v is not None}
        # Sort results by part index
        sorted_results = dict(sorted(results.items()))
        first_result = next(iter(sorted_results.values()))

        # Initialize paths for merged files
        merged_mono_path = None
        merged_dual_path = None
        merged_no_watermark_mono_path = None
        merged_no_watermark_dual_path = None
        try:
            # Merge monolingual PDFs if they exist
            if (
                any(r.mono_pdf_path for r in results.values())
                and not self.config.no_mono
            ):
                merged_mono_path = self._merge_pdfs(
                    [
                        r.mono_pdf_path
                        for r in sorted_results.values()
                        if r.mono_pdf_path
                    ],
                    mono_file_name,
                    tag="merged_mono",
                )
        except Exception as e:
            logger.error(f"Error merging monolingual PDFs: {e}")
            merged_mono_path = None

        try:
            # Merge dual-language PDFs if they exist
            if (
                any(r.dual_pdf_path for r in results.values())
                and not self.config.no_dual
            ):
                merged_dual_path = self._merge_pdfs(
                    [
                        r.dual_pdf_path
                        for r in sorted_results.values()
                        if r.dual_pdf_path
                    ],
                    dual_file_name,
                    tag="merged_dual",
                )
        except Exception as e:
            logger.error(f"Error merging dual-language PDFs: {e}")
            merged_dual_path = None

        if any(
            r.dual_pdf_path != r.no_watermark_dual_pdf_path
            or r.mono_pdf_path != r.no_watermark_mono_pdf_path
            for r in results.values()
        ):
            try:
                # Merge no-watermark PDFs if they exist
                if (
                    any(r.no_watermark_mono_pdf_path for r in results.values())
                    and not self.config.no_mono
                ):
                    merged_no_watermark_mono_path = self._merge_pdfs(
                        [
                            r.no_watermark_mono_pdf_path
                            for r in sorted_results.values()
                            if r.no_watermark_mono_pdf_path
                        ],
                        mono_file_name_no_watermark,
                        tag="merged_no_watermark_mono",
                    )
            except Exception as e:
                logger.error(f"Error merging no-watermark PDFs: {e}")
                merged_no_watermark_mono_path = None

            try:
                if (
                    any(r.no_watermark_dual_pdf_path for r in results.values())
                    and not self.config.no_dual
                ):
                    merged_no_watermark_dual_path = self._merge_pdfs(
                        [
                            r.no_watermark_dual_pdf_path
                            for r in sorted_results.values()
                            if r.no_watermark_dual_pdf_path
                        ],
                        "merged_no_watermark_dual.pdf",
                        tag="merged_no_watermark_dual",
                    )
            except Exception as e:
                logger.error(f"Error merging no-watermark PDFs: {e}")
                merged_no_watermark_dual_path = None

        auto_extracted_glossary_path = None
        if (
            self.config.save_auto_extracted_glossary
            and self.config.shared_context_cross_split_part.auto_extracted_glossary
        ):
            auto_extracted_glossary_path = self.config.get_output_file_path(
                f"{basename}{debug_suffix}.{self.config.lang_out}.glossary.csv"
            )
            with auto_extracted_glossary_path.open("w", encoding="utf-8-sig") as f:
                logger.info(
                    f"save auto extracted glossary to {auto_extracted_glossary_path}"
                )
                f.write(
                    self.config.shared_context_cross_split_part.auto_extracted_glossary.to_csv()
                )

        # Create merged result
        merged_result = TranslateResult(
            mono_pdf_path=merged_mono_path,
            dual_pdf_path=merged_dual_path,
            auto_extracted_glossary_path=auto_extracted_glossary_path,
        )
        merged_result.no_watermark_mono_pdf_path = merged_no_watermark_mono_path
        merged_result.no_watermark_dual_pdf_path = merged_no_watermark_dual_path

        if merged_result.no_watermark_mono_pdf_path is None:
            merged_result.no_watermark_mono_pdf_path = merged_mono_path
        elif merged_result.mono_pdf_path is None:
            merged_result.mono_pdf_path = merged_no_watermark_mono_path

        if merged_result.no_watermark_dual_pdf_path is None:
            merged_result.no_watermark_dual_pdf_path = merged_dual_path
        elif merged_result.dual_pdf_path is None:
            merged_result.dual_pdf_path = merged_no_watermark_dual_path

        # Calculate total time
        total_time = sum(
            r.total_seconds for r in results.values() if hasattr(r, "total_seconds")
        )
        merged_result.total_seconds = total_time

        return merged_result

    def _merge_pdfs(
        self, pdf_paths: list[str | Path], output_name: str, tag: str
    ) -> Path:
        """Merge multiple PDFs into one"""
        if not pdf_paths:
            return None

        output_path = self.config.get_output_file_path(output_name)
        merged_doc = Document()

        for pdf_path in pdf_paths:
            doc = Document(str(pdf_path))
            merged_doc.insert_pdf(doc)

        merged_doc = PDFCreater.subset_fonts_in_subprocess(
            merged_doc, self.config, tag=tag
        )
        PDFCreater.save_pdf_with_timeout(
            merged_doc, str(output_path), translation_config=self.config
        )

        return output_path
