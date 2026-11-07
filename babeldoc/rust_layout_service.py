"""Layout service that owns Python model resources, not writable IL pages."""

import sys
import threading
import weakref
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from babeldoc.const import close_process_pool
from babeldoc.rust_callback_bridge import CallbackRun


@dataclass(frozen=True, slots=True)
class BoxProjection:
    x: object
    y: object
    x2: object
    y2: object


@dataclass(frozen=True, slots=True)
class VisualProjection:
    box: object


@dataclass(frozen=True, slots=True)
class CharacterProjection:
    visual_bbox: object
    char_unicode: object
    vertical: object


_PAGE_PROJECTION_OPTIONS = (
    {"slots": True, "weakref_slot": True} if sys.version_info >= (3, 11) else {}
)


@dataclass(frozen=True, eq=False, **_PAGE_PROJECTION_OPTIONS)
class PageProjection:
    page_number: object


@dataclass(frozen=True, slots=True, eq=False)
class CharacterPageProjection(PageProjection):
    pdf_character: tuple


class ModelConfig:
    __slots__ = ("_config", "_flag")

    def __init__(self, config, flag):
        self._config, self._flag = config, flag

    def raise_if_cancelled(self):
        self._flag.checkpoint()

    def get_working_file_path(self, name):
        return self._config.get_working_file_path(name)


def needs_characters(model):
    modules = {cls.__module__ for cls in type(model).__mro__}
    return (
        "babeldoc.docvision.rpc_doclayout6" in modules
        or "babeldoc.docvision.rpc_doclayout7" in modules
        or (
            "babeldoc.docvision.rpc_doclayout8" in modules
            and model.requires_line_extraction
        )
    )


def _save_debug_image(config, image: np.ndarray, layout, page_number: int):
    """Save a debug image with layout boxes when debug output is enabled."""
    if not config.debug:
        return

    debug_dir = Path(config.get_working_file_path("ocr-box-image"))
    debug_dir.mkdir(parents=True, exist_ok=True)

    debug_image = image.copy()
    for box in layout.boxes:
        x0, y0, x1, y1 = box.xyxy
        cv2.rectangle(
            debug_image,
            (int(x0), int(y0)),
            (int(x1), int(y1)),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            debug_image,
            layout.names[box.cls],
            (int(x0), int(y0) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    img_bgr = cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR)
    output_path = debug_dir / f"{page_number}.jpg"
    cv2.imwrite(str(output_path), img_bgr)


def make_pages(inputs):
    pages = []
    for number, chars in inputs:
        if chars is None:
            pages.append(PageProjection(number))
        else:
            projected = []
            for present, bounds, unicode, vertical in chars:
                visual = (
                    VisualProjection(None if bounds is None else BoxProjection(*bounds))
                    if present
                    else None
                )
                projected.append(CharacterProjection(visual, unicode, vertical))
            pages.append(CharacterPageProjection(number, tuple(projected)))
    return pages


def project_result(page, layouts, pdf, index):
    """Lossless V2 projection. An empty box list never touches MuPDF."""
    boxes = []
    mediabox = None
    for layout in layouts.boxes:
        xyxy = layout.xyxy
        if len(xyxy) != 4:
            raise ValueError("expected four coordinates")
        size = pdf[page.page_number].mediabox_size
        mediabox = (size.x, size.y)
        boxes.append(
            (
                tuple(
                    (32 if isinstance(c, np.float32) else 64, float(c)) for c in xyxy
                ),
                layout.conf.item(),
                layouts.names[layout.cls],
            )
        )
    return index, mediabox, boxes


class LayoutService:
    cleanup_methods = frozenset({"close", "close_process_pool"})

    def __init__(self, config, pdf, font_mapper=None):
        self.config, self.pdf = config, pdf
        self.model = config.doc_layout_model
        self.font_mapper = font_mapper
        self.characters = needs_characters(self.model)
        self.generator = None
        self.pages = []
        self.identities = {}
        self._original_mapper = None
        self._mapper_replaced = False
        self.flag = None

    def begin(self, inputs):
        self.pages = make_pages(inputs)
        self.identities = {
            id(page): (weakref.ref(page), index)
            for index, page in enumerate(self.pages)
        }
        if self.characters:
            if self.font_mapper is None:
                raise ValueError(
                    "layout character backend requires Rust FontMapperHandle"
                )
            self._original_mapper = self.model.font_mapper
            self.model.font_mapper = self.font_mapper
            self._mapper_replaced = True
        self.generator = iter(
            self.model.handle_document(
                self.pages,
                self.pdf,
                ModelConfig(self.config, self.flag),
                lambda image, layout, page_number: _save_debug_image(
                    self.config, image, layout, page_number
                ),
            )
        )

    def next_page(self):
        try:
            page, layouts = next(self.generator)
        except StopIteration:
            self.generator = None
            return None
        identity = self.identities.get(id(page))
        index = identity[1] if identity is not None and identity[0]() is page else None
        result = project_result(page, layouts, self.pdf, index)
        # Model backends retain their own page references when needed. Drop this
        # service's copy after publication, without invalidating the identity map.
        if index is not None:
            self.pages[index] = None
        return result

    def close(self):
        try:
            if self.generator is not None:
                self.generator.close()
        finally:
            self.generator = None
            self.pages = []
            self.identities = {}
            if self._mapper_replaced:
                self.model.font_mapper = self._original_mapper
                self._mapper_replaced = False

    def close_process_pool(self):
        close_process_pool()


def run_layout(
    document, config, pdf, native, font_mapper=None, cancel_event=None, dump_path=None
):
    """Standalone callback entry for layout testing and embedding."""
    service = LayoutService(config, pdf, font_mapper)
    with CallbackRun(
        {"layout": (service, 1)}, cancel_event or threading.Event(), native
    ) as run:
        service.flag = run.flag
        try:
            return native.run_layout_parser(
                document,
                run.executors["layout"],
                run.flag,
                config.progress_monitor,
                config.debug,
                service.characters,
                None if dump_path is None else str(dump_path),
            )
        finally:
            # Submit on the same executor even if the Rust entry failed before begin.
            executor = run.executors["layout"]
            executor.pool.submit(service.close).result()
            executor.pool.submit(service.close_process_pool).result()
