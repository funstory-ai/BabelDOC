"""Thin, sequential file service for the Rust writer stages.

The document objects persist between calls. Branches and retry belong to Rust;
PDF operations retain the existing writer's subprocess and fallback behavior.
"""

import logging
from pathlib import Path
from types import SimpleNamespace

import pymupdf

from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater

logger = logging.getLogger(__name__)


class WriterFileService:
    cleanup_methods = frozenset({"close"})

    def __init__(self, config, mediabox_data, original_path):
        self.config = config
        self.mediabox_data = mediabox_data
        self.creator = PDFCreater.__new__(PDFCreater)
        self.creator.original_pdf_path = original_path
        self.creator.translation_config = config
        self.documents = []
        self.translated = self.original = self.dual = None
        self.glossary_path = None

    def _own(self, pdf):
        self.documents.append(pdf)
        return pdf

    def open_translated(self, path):
        self.translated = self._own(pymupdf.open(path))

    def open_original(self, path):
        self.original = self._own(pymupdf.open(path))

    def subset_mono(self):
        self.translated = self._own(
            self.creator.subset_fonts_in_subprocess(
                self.translated, self.config, tag="mono"
            )
        )

    def subset_debug(self):
        self.original = self._own(
            self.creator.subset_fonts_in_subprocess(
                self.original, self.config, tag="debug"
            )
        )

    def restore_media_box(self):
        try:
            self.creator.restore_media_box(self.translated, self.mediabox_data)
        except Exception:
            logger.exception("restore media box failed")

    def keep_translated(self, pages):
        self.translated.delete_pages(
            list(set(range(len(self.translated))) - set(pages))
        )

    def keep_original(self, pages):
        self.original.delete_pages(list(set(range(len(self.original))) - set(pages)))

    def save_decompressed_mono(self, path):
        self.translated.save(path, expand=True, pretty=True)

    def save_decompressed_dual(self, path):
        self.dual.save(path, expand=True, pretty=True)

    def _save(self, pdf, path, garbage, clean, tag):
        return self.creator.save_pdf_with_timeout(
            pdf,
            path,
            self.config,
            garbage=garbage,
            deflate=True,
            clean=clean,
            deflate_fonts=True,
            linear=False,
            tag=tag,
        )

    def save_mono(self, path, garbage, clean):
        self._save(self.translated, path, garbage, clean, "mono")

    def save_dual(self, path, garbage, clean):
        self._save(self.dual, path, garbage, clean, "dual")

    def make_alternating(self, first):
        self.dual = self._own(
            self.creator.create_alternating_pages_dual_pdf(
                self.original,
                self.translated,
                SimpleNamespace(
                    dual_translate_first=first, input_file=self.config.input_file
                ),
            )
        )

    def make_side_by_side(self, first, path):
        self.dual = self._own(
            self.creator.create_side_by_side_dual_pdf(
                self.original,
                self.translated,
                path,
                SimpleNamespace(
                    dual_translate_first=first, input_file=self.config.input_file
                ),
            )
        )

    def glossary_hook(self):
        glossary = getattr(
            self.config.shared_context_cross_split_part,
            "auto_extracted_glossary",
            None,
        )
        if self.glossary_path is None or glossary is None:
            return None
        self.glossary_path.parent.mkdir(parents=True, exist_ok=True)
        with Path(self.glossary_path).open("w", encoding="utf-8-sig") as output:
            output.write(glossary.to_csv())
        return str(self.glossary_path)

    def close(self):
        documents, self.documents = self.documents, []
        self.translated = self.original = self.dual = None
        for pdf in reversed(documents):
            if not pdf.is_closed:
                pdf.close()
