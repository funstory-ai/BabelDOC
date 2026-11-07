"""Sequential PyMuPDF render and stream-application service for scan detection."""

import hashlib
import logging

import cv2
import pymupdf
from skimage.metrics import structural_similarity

from babeldoc.format.pdf.document_il.utils.raster_geometry import with_pixel_budget

logger = logging.getLogger("babeldoc.format.pdf.document_il.midend.detect_scanned_file")


class ScanFileService:
    cleanup_methods = frozenset({"close"})

    def __init__(self):
        self.pdf = None
        self.before = {}
        self.records = []
        self.applications = []

    def begin(self, path):
        self.pdf = pymupdf.open(path)
        self.before.clear()

    def render_before(self, page_number):
        geometry = with_pixel_budget(
            self.pdf[page_number], 72, normalize_rotation=False
        )
        image = geometry.image[:, :, ::-1]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.before[page_number] = (geometry, gray)
        self.records.append(
            {
                "page": page_number,
                "before_sha256": hashlib.sha256(gray.tobytes()).hexdigest(),
            }
        )

    @classmethod
    def _replace_stream(cls, pdf, xref, stream):
        pdf.update_stream(xref, stream)

    @classmethod
    def _ensure_resources(cls, pdf, xref, stream, candidates):
        # Keep the same order and helper behavior as PDFCreater's 0.6.4 writer.
        from babeldoc.format.pdf.document_il.backend.pdf_creater import PDFCreater

        PDFCreater._ensure_stream_extgstate_resources(pdf, xref, stream, candidates)
        PDFCreater._ensure_stream_shading_resources(pdf, xref, stream, candidates)

    def apply_and_compare(self, page_number, stream, forms):
        page = self.pdf[page_number]
        candidate_xrefs = [page.xref, *(form["xref_id"] for form in forms)]
        for form in forms:
            xref = form["xref_id"]
            try:
                form_stream = form["stream"]
                self._ensure_resources(self.pdf, xref, form_stream, candidate_xrefs)
                self._replace_stream(self.pdf, xref, form_stream)
            except Exception:
                logger.warning(f"update xref {xref} stream fail, continue")
        container = self.pdf.get_new_xref()
        self.pdf.update_object(container, "<<>>")
        page_stream = stream
        self._ensure_resources(self.pdf, page.xref, page_stream, candidate_xrefs)
        self._replace_stream(self.pdf, container, page_stream)
        page.set_contents(container)
        self.applications.append(self._application_signature(page, forms))
        geometry, before = self.before.pop(page_number)
        after = geometry.render_at_dpi(page, normalize_rotation=False)[:, :, ::-1]
        after = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
        similarity = float(structural_similarity(before, after))
        self.records[-1].update(
            after_sha256=hashlib.sha256(after.tobytes()).hexdigest(), ssim=similarity
        )
        return similarity

    def _application_signature(self, page, forms):
        page_contents = page.get_contents()
        page_stream = self.pdf.xref_stream(page_contents[-1]) if page_contents else b""
        return {
            "page": page.number,
            "page_stream_sha256": hashlib.sha256(page_stream).hexdigest(),
            "page_resources": self.pdf.xref_get_key(page.xref, "Resources"),
            "forms": [
                {
                    "xref_id": form["xref_id"],
                    "stream_sha256": hashlib.sha256(
                        self.pdf.xref_stream(form["xref_id"]) or b""
                    ).hexdigest(),
                    "resources": self.pdf.xref_get_key(form["xref_id"], "Resources"),
                }
                for form in forms
            ],
        }

    def close(self):
        # Match the source's reference-release behavior; do not call Document.close().
        self.pdf = None
        self.before.clear()
