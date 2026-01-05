import base64
import json
import logging
import threading
import unicodedata
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple, Dict, Union

import cv2
import httpx
import msgpack
import numpy as np
import pymupdf
from tenacity import retry
from tenacity import retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential

import babeldoc
from babeldoc.docvision.base_doclayout import DocLayoutModel, YoloBox, YoloResult
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.extract_char import (
    convert_page_to_char_boxes,
    process_page_chars_to_lines,
    Line,
)
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.layout_helper import SPACE_REGEX
from babeldoc.format.pdf.document_il.utils.mupdf_helper import (
    get_no_rotation_img,
    get_no_rotation_img_multiprocess,
)

logger = logging.getLogger(__name__)

class LayoutVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    V4 = "v4"
    V5 = "v5"
    V6 = "v6"
    V7 = "v7"

def encode_image(image: Union[str, np.ndarray]) -> bytes:
    """Read and encode image to bytes."""
    if isinstance(image, str):
        if not Path(image).exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        img = cv2.imread(image)
        if img is None:
            raise ValueError(f"Failed to read image: {image}")
    else:
        img = image

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    encoded = cv2.imencode(".jpg", img)[1].tobytes()
    return encoded

def resize_and_pad_image(image: np.ndarray, new_shape: Union[int, Tuple[int, int]]) -> np.ndarray:
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = image.shape[:2]
    new_h, new_w = new_shape

    r = min(new_h / h, new_w / w)
    resized_h, resized_w = int(round(h * r)), int(round(w * r))

    image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_h = new_h - resized_h
    pad_w = new_w - resized_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    image = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return image

def scale_boxes(img1_shape, boxes, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
    pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)
    boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / gain
    return boxes

class ResultContainer:
    def __init__(self):
        self.result = YoloResult(boxes_data=np.array([]), names=[])

# Base Strategy Interface
class LayoutStrategy(ABC):
    def __init__(self, host: str):
        self.host = host
        self.dpi = 150
        self.max_workers = 1
        self.lock = threading.Lock()

    @abstractmethod
    def predict_image(self, image: np.ndarray, **kwargs) -> YoloResult:
        pass

    def predict_page(
        self,
        page: il_version_1.Page,
        mupdf_doc: pymupdf.Document,
        translate_config: Any,
        save_debug_image: Any,
    ) -> Tuple[il_version_1.Page, YoloResult]:
        translate_config.raise_if_cancelled()
        with self.lock:
            pix = get_no_rotation_img(mupdf_doc[page.page_number], dpi=self.dpi)
        image = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width, 3
        )[:, :, ::-1]

        # Default behavior for most strategies
        predict_result = self.predict_image(image)
        save_debug_image(image, predict_result, page.page_number + 1)
        return page, predict_result

    def handle_document(
        self,
        pages: List[il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config: Any,
        save_debug_image: Any,
    ) -> Generator[Tuple[il_version_1.Page, YoloResult], None, None]:
        # Default implementation using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            yield from executor.map(
                self.predict_page,
                pages,
                (mupdf_doc for _ in range(len(pages))),
                (translate_config for _ in range(len(pages))),
                (save_debug_image for _ in range(len(pages))),
            )

    def init_font_mapper(self, translation_config):
        """Hook for initializing font mapper if needed (used in v6)."""
        pass

class StandardStrategy(LayoutStrategy):
    """V1, V2, V4 implementation (MsgPack /inference)"""
    def __init__(self, host: str, dpi: int = 150, max_workers: int = 1, fixed_resize: bool = False):
        super().__init__(host)
        self.dpi = dpi
        self.max_workers = max_workers
        self.fixed_resize = fixed_resize # If True, resize to 800x800, else resize to orig (but padded to stride?) - Wait, v1 resized to 800x800 and padded.

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
    )
    def _predict_request(self, image_bytes: bytes) -> Any:
        data = {"image": [image_bytes]}
        if self.fixed_resize:
             data["imgsz"] = 800
        # V1 sends imgsz, V2/V4 usually don't send imgsz in the dict or send it differently?
        # Checking V1: data={"image": image_data, "imgsz": imgsz}
        # Checking V2: data={"image": image_data}

        packed_data = msgpack.packb(data, use_bin_type=True)
        response = httpx.post(
            f"{self.host}/inference",
            data=packed_data,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            timeout=480, # V1 was 300, V2 480
            follow_redirects=True,
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} {response.text}")
        return msgpack.unpackb(response.content, raw=False)

    def predict_image(self, image: np.ndarray, **kwargs) -> YoloResult:
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (800, 800) if self.fixed_resize else (orig_h, orig_w)

        if image.shape[0] != target_imgsz[0] or image.shape[1] != target_imgsz[1]:
            image_processed = resize_and_pad_image(image, new_shape=target_imgsz)
        else:
            image_processed = image

        image_bytes = encode_image(image_processed)
        preds = self._predict_request(image_bytes)

        if self.fixed_resize:
            # V1 logic: scale back to orig
            scale_target = (orig_h, orig_w)
            scale_src = target_imgsz
        else:
            # V2/V4 logic: scale back to PDF points (72 dpi)
            scale_target = (orig_h / self.dpi * 72, orig_w / self.dpi * 72)
            scale_src = target_imgsz

        result_container = ResultContainer()

        # FIX: Handle single dict response (V2/V4)
        if isinstance(preds, dict):
            preds = [preds]

        if len(preds) > 0:
            idx = 0
            id_lookup = {}
            names = {}
            for pred in preds:
                # Let's use a unified unpacking
                raw_boxes = pred.get("boxes", [])
                raw_names = pred.get("names", {})

                final_boxes = []

                for box in raw_boxes:
                    score = box.get("score", box.get("conf"))
                    if score is not None and score < 0.7 and not self.fixed_resize: # V2/V4 filter
                         continue

                    xyxy = box.get("xyxy", box.get("coordinate"))
                    label = box.get("label")
                    cls_id = box.get("cls")

                    if label and cls_id is None:
                         # V2 logic mapping label to ID
                         if label not in names:
                             # Reverse lookup? No, just assign new ID if missing
                             pass # Ideally we use what server sent or build our own

                    # Assume server sends standard format or we normalize it.
                    # V1: boxes have xyxy, conf, cls. names is dict {id: label}
                    # V2: boxes have coordinate, score, label.

                    if "coordinate" in box: # V2 style
                         xyxy = box["coordinate"]
                         conf = box["score"]
                         label = box["label"]

                         # Map label to ID
                         if label not in id_lookup:
                             idx += 1
                             id_lookup[label] = idx
                             names[idx] = label
                         cls_id = id_lookup[label]
                    else: # V1 style
                         conf = box["conf"]
                         cls_id = box["cls"]
                         if raw_names:
                             names = {int(k): v for k, v in raw_names.items()}

                    scaled_xyxy = scale_boxes(scale_src, np.array(xyxy), scale_target)
                    final_boxes.append(YoloBox(None, scaled_xyxy, np.array(conf), cls_id))

                result_container.result = YoloResult(boxes=final_boxes, names=names)

        return result_container.result

    def predict_page(self, page, mupdf_doc, translate_config, save_debug_image):
        if self.fixed_resize: # V1: uses get_no_rotation_img with default dpi (72)
             translate_config.raise_if_cancelled()
             with self.lock:
                 pix = get_no_rotation_img(mupdf_doc[page.page_number]) # Default dpi=72
             image = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)[:, :, ::-1]
             predict_result = self.predict_image(image)
             save_debug_image(image, predict_result, page.page_number + 1)
             return page, predict_result
        else:
             return super().predict_page(page, mupdf_doc, translate_config, save_debug_image)


class AnalyzeStrategy(LayoutStrategy):
    """V3, V5 implementation (JSON /analyze or /analyze_hybrid)"""
    def __init__(self, host: str, endpoint: str, max_workers: int = 4):
        super().__init__(host)
        self.endpoint = endpoint
        self.max_workers = max_workers
        self.dpi = 150

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
    )
    def _predict_request(self, image_bytes: bytes) -> Any:
        response = httpx.post(
            f"{self.host}/{self.endpoint}?min_sim=0.7&early_stop=0.99&timeout=1800",
            files={"file": ("image.jpg", image_bytes, "image/jpeg")},
            headers={"Accept": "application/json"},
            timeout=1800,
            follow_redirects=True,
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} {response.text}")
        return json.loads(response.text)

    def predict_image(self, image: np.ndarray, **kwargs) -> YoloResult:
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (orig_h, orig_w)

        # V3/V5 resize logic: target_imgsz = orig, then resize/pad to it?
        # Actually in V3/V5 code: target_imgsz = (orig_h, orig_w). resize_and_pad_image called with this.
        # This usually means no change unless padding is needed for stride.
        image_processed = resize_and_pad_image(image, new_shape=target_imgsz)
        image_bytes = encode_image(image_processed)

        result_data = self._predict_request(image_bytes)

        scale_target = (orig_h / self.dpi * 72, orig_w / self.dpi * 72)

        boxes = []
        names = {}
        idx = 0
        id_lookup = {}

        # Unpack JSON result
        # V3: boxes key, ocr_match_score, coords
        # V5: clusters key, box, label

        items = []
        if "boxes" in result_data: # V3
            items = result_data["boxes"]
            key_score = "ocr_match_score"
            key_coords = "coords"
        elif "clusters" in result_data: # V5
            items = result_data["clusters"]
            key_score = None # Conf = 1
            key_coords = "box"

        for item in items:
            score = item[key_score] if key_score else 1.0
            if key_score and score < 0.7: continue

            coords = item[key_coords]
            label = item["label"]

            if label not in id_lookup:
                idx += 1
                id_lookup[label] = idx
                names[idx] = label
            cls_id = id_lookup[label]

            scaled_xyxy = scale_boxes(target_imgsz, np.array(coords), scale_target)
            boxes.append(YoloBox(None, scaled_xyxy, np.array(score), cls_id))

        return YoloResult(boxes=boxes, names=names)


class VlmStrategy(LayoutStrategy):
    """V7 implementation"""
    def __init__(self, host: str):
        super().__init__(host)
        self.dpi = 150
        self.max_workers = 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
    )
    def _predict_request(self, image_bytes: bytes, lines: List[Line], image_shape: Tuple[int, int]) -> Any:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        def convert_line(line):
            boxes = [c[0] for c in line.chars]
            if not boxes: return None
            min_x = min([b.x for b in boxes])
            max_x = max([b.x2 for b in boxes])
            min_y = min([b.y for b in boxes])
            max_y = max([b.y2 for b in boxes])

            min_x = min_x / 72 * self.dpi
            max_x = max_x / 72 * self.dpi
            min_y = min_y / 72 * self.dpi
            max_y = max_y / 72 * self.dpi

            image_height = image_shape[0]
            # Flip Y
            min_y, max_y = image_height - max_y, image_height - min_y

            return {"box": [min_x, min_y, max_x, max_y], "text": line.text}

        formatted_results = [convert_line(l) for l in lines]
        formatted_results = [r for r in formatted_results if r]

        request_data = {
            "image": image_b64,
            "ocr_results": formatted_results,
            "image_size": list(image_shape)[::-1],
        }

        response = httpx.post(
            f"{self.host}/inference",
            json=request_data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=1800,
            follow_redirects=True,
        )
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.status_code} {response.text}")
        return json.loads(response.text)

    def predict_image(self, image: np.ndarray, lines: Optional[List[Line]] = None, **kwargs) -> YoloResult:
        if lines is None: lines = []
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (orig_h, orig_w)
        image_processed = resize_and_pad_image(image, new_shape=target_imgsz)
        image_bytes = encode_image(image_processed)

        result_data = self._predict_request(image_bytes, lines, image.shape[:2])

        scale_target = (orig_h / self.dpi * 72, orig_w / self.dpi * 72)
        boxes = []
        names = {}
        idx = 0
        id_lookup = {}

        if "clusters" in result_data:
            for item in result_data["clusters"]:
                coords = item["box"]
                label = item["label"]
                if label not in id_lookup:
                    idx += 1
                    id_lookup[label] = idx
                    names[idx] = label
                cls_id = id_lookup[label]
                scaled_xyxy = scale_boxes(target_imgsz, np.array(coords), scale_target)
                boxes.append(YoloBox(None, scaled_xyxy, np.array(1.0), cls_id))

        return YoloResult(boxes=boxes, names=names)

    def predict_page(self, page, mupdf_doc, translate_config, save_debug_image):
        translate_config.raise_if_cancelled()
        with self.lock:
            pix = get_no_rotation_img(mupdf_doc[page.page_number], dpi=self.dpi)
        image = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height, pix.width, 3
        )[:, :, ::-1]

        char_boxes = convert_page_to_char_boxes(page)
        lines = process_page_chars_to_lines(char_boxes)

        predict_result = self.predict_image(image, lines=lines)
        save_debug_image(image, predict_result, page.page_number + 1)
        return page, predict_result


class DualStrategy(LayoutStrategy):
    """V6 implementation (VLM + Paddle/Mosec hybrid)"""
    def __init__(self, host: str):
        if ";" not in host:
             raise ValueError("Host must be two hosts separated by ';'")
        self.host1, self.host2 = [h.strip() for h in host.split(";", 1)]
        super().__init__(host)
        self.dpi = 150
        self.max_workers = 32
        self.font_mapper = None

    def init_font_mapper(self, translation_config):
        self.font_mapper = FontMapper(translation_config)

    def _predict_vlm(self, image: np.ndarray, lines: List[Line]) -> Any:
        # Re-using logic similar to VLM but with filter_text and clip_num
        image_bytes = encode_image(image)
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_h, image_w = image.shape[:2]

        def clip_num(num, min_v, max_v):
            return max(min_v, min(num, max_v))

        def filter_text(txt):
            normalize = unicodedata.normalize("NFKC", txt)
            if self.font_mapper:
                unicodes = [c for c in normalize if self.font_mapper.has_char(c)]
                normalize = "".join(unicodes)
            return SPACE_REGEX.sub(" ", normalize).strip()

        formatted = []
        for line in lines:
            if not line.text: continue
            boxes = [c[0] for c in line.chars]
            min_x = min(b.x for b in boxes)
            max_x = max(b.x2 for b in boxes)
            min_y = min(b.y for b in boxes)
            max_y = max(b.y2 for b in boxes)

            min_x = min_x / 72 * self.dpi
            max_x = max_x / 72 * self.dpi
            min_y = min_y / 72 * self.dpi
            max_y = max_y / 72 * self.dpi

            min_y, max_y = image_h - max_y, image_h - min_y

            if (max_x - min_x) * (max_y - min_y) < 1: continue

            min_x = clip_num(min_x, 0, image_w - 1)
            max_x = clip_num(max_x, 0, image_w - 1)
            min_y = clip_num(min_y, 0, image_h - 1)
            max_y = clip_num(max_y, 0, image_h - 1)

            filtered = filter_text(line.text)
            if not filtered: continue

            formatted.append({"box": [min_x, min_y, max_x, max_y], "text": filtered})

        if not formatted: return None

        request_data = {
            "image": image_b64,
            "ocr_results": formatted,
            "image_size": [image_h, image_w],
        }

        try:
            response = httpx.post(
                f"{self.host1}/inference",
                json=request_data,
                headers={"Accept": "application/json", "Content-Type": "application/json"},
                timeout=30,
                follow_redirects=True
            )
            if response.status_code == 200:
                return json.loads(response.text)
        except Exception as e:
            logger.warning(f"VLM request failed: {e}")
            raise
        return None

    def _predict_paddle(self, image: np.ndarray) -> Any:
        # Using StandardStrategy logic for the second host
        try:
            s = StandardStrategy(self.host2, dpi=self.dpi, fixed_resize=False)
            # Re-implement call to get raw data
            image_bytes = encode_image(image)
            return s._predict_request(image_bytes)
        except Exception as e:
            logger.warning(f"Paddle request failed: {e}")
            raise

    def predict_image(self, image: np.ndarray, lines: Optional[List[Line]] = None, **kwargs) -> YoloResult:
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (orig_h, orig_w)
        image_proc = resize_and_pad_image(image, new_shape=target_imgsz)

        preds1 = None
        preds2 = None

        with ThreadPoolExecutor(max_workers=2) as ex:
             f1 = ex.submit(self._predict_vlm, image_proc, lines) if lines else None
             f2 = ex.submit(self._predict_paddle, image_proc)
             if f1: preds1 = f1.result()
             preds2 = f2.result()

        pdf_h, pdf_w = orig_h / self.dpi * 72, orig_w / self.dpi * 72

        merged_boxes = []
        names = {}

        def process(preds, id_offset, suffix):
            if not preds: return
            # Normalize structure
            # VLM returns {"clusters": ...}
            # Paddle returns msgpack unpack result (list of dicts or dict)

            items = []
            if isinstance(preds, dict) and "clusters" in preds:
                for c in preds["clusters"]:
                    items.append({
                        "xyxy": c["box"],
                        "conf": 1.0,
                        "cls": 0, # Placeholder
                        "label": c["label"]
                    })
            elif isinstance(preds, dict) and "boxes" in preds:
                 # Standard msgpack structure
                 raw_names = preds.get("names", {})
                 for b in preds["boxes"]:
                      if b.get("score", 0) < 0.7: continue
                      items.append({
                          "xyxy": b.get("coordinate", b.get("xyxy")),
                          "conf": b.get("score", b.get("conf")),
                          "cls": b.get("cls"),
                          "label": b.get("label", raw_names.get(b.get("cls")))
                      })

            for item in items:
                scaled = scale_boxes(target_imgsz, np.array(item["xyxy"]), (pdf_h, pdf_w))

                # Assign ID
                # We need a stable mapping.
                # Just use hash or auto increment?
                # V6 uses offset.

                label = item["label"]
                if suffix: label += suffix

                # Find or create ID for this label
                # In V6 original code, it generated IDs dynamically for each response then mapped.
                # Here we need a global map for this result?
                # Actually V6 just used "new_cls_id = box['cls'] + id_offset" which implies box['cls'] is integer.
                # But VLM (clusters) doesn't have integer cls.
                # V6 logic:
                # For VLM: id_lookup[label] = idx.
                # For Paddle: id_lookup[label] = idx.
                # Then +offset.

                # Simplified:
                # We just need unique ID for unique label in this YoloResult.
                # names dict stores ID -> Label.

                # We can't easily rely on upstream IDs being consistent.
                pass

        # Re-implementing V6 merging logic more precisely

        # ... (Merging logic from V6) ...
        # Since this is complex, I will copy-paste the logic but adapted.

        boxes_list = []

        # Helper to process
        def _process_preds(preds, id_offset, label_suffix):
             if not preds: return

             # Extract boxes and labels
             # VLM
             if isinstance(preds, dict) and "clusters" in preds:
                  for i, box in enumerate(preds["clusters"]):
                       label = box["label"]
                       if label_suffix: label += label_suffix
                       xyxy = box["box"]
                       conf = 1.0
                       scaled_xyxy = scale_boxes(target_imgsz, np.array(xyxy), (pdf_h, pdf_w))
                       boxes_list.append((scaled_xyxy, conf, label, id_offset + i)) # Use arbitrary ID base

             # Paddle
             elif isinstance(preds, dict) and "boxes" in preds:
                  raw_names = preds.get("names", {})
                  # Need to rebuild ID map if missing?
                  # V6 original code assumes boxes have 'cls' or 'label'.
                  # If 'label' exists, it builds id_lookup.

                  local_names = {}
                  local_idx = 0
                  local_lookup = {}

                  for box in preds["boxes"]:
                       if box.get("score", 0) < 0.7: continue

                       label = box.get("label")
                       if not label and "cls" in box:
                            label = raw_names.get(box["cls"], str(box["cls"]))

                       if label not in local_lookup:
                            local_idx += 1
                            local_lookup[label] = local_idx

                       cls_id = local_lookup[label]

                       if label_suffix: label += label_suffix

                       xyxy = box.get("coordinate", box.get("xyxy"))
                       conf = box.get("score", box.get("conf"))
                       scaled_xyxy = scale_boxes(target_imgsz, np.array(xyxy), (pdf_h, pdf_w))

                       boxes_list.append((scaled_xyxy, conf, label, cls_id + id_offset))

        _process_preds(preds1, 1000, "_hybrid")
        _process_preds(preds2, 2000, None)

        # Now unify IDs
        final_boxes = []
        final_names = {}
        label_to_id = {}
        next_id = 1

        for xyxy, conf, label, _ in boxes_list:
             if label not in label_to_id:
                  label_to_id[label] = next_id
                  final_names[next_id] = label
                  next_id += 1
             final_boxes.append(YoloBox(None, xyxy, np.array(conf), label_to_id[label]))

        final_boxes.sort(key=lambda b: b.conf, reverse=True)

        # Post process (V6 specific)
        self.post_process_boxes(final_boxes, final_names)

        return YoloResult(boxes=final_boxes, names=final_names)

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)
        if x2_inter <= x1_inter or y2_inter <= y1_inter: return 0.0
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def is_subset(self, inner, outer):
        return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]

    def expand_box_to_contain(self, expand, contain):
        return [min(expand[0], contain[0]), min(expand[1], contain[1]), max(expand[2], contain[2]), max(expand[3], contain[3])]

    def post_process_boxes(self, merged_boxes: List[YoloBox], names: Dict[int, str]):
        for i, text_box in enumerate(merged_boxes):
            text_label = names.get(text_box.cls, "")
            if "text" not in text_label: continue
            for j, para_box in enumerate(merged_boxes):
                if i == j: continue
                para_label = names.get(para_box.cls, "")
                if "paragraph_hybrid" not in para_label: continue
                iou = self.calculate_iou(text_box.xyxy, para_box.xyxy)
                if iou > 0.95 and not self.is_subset(para_box.xyxy, text_box.xyxy):
                    expanded = self.expand_box_to_contain(text_box.xyxy, para_box.xyxy)
                    merged_boxes[i] = YoloBox(None, np.array(expanded), text_box.conf, text_box.cls)

    def handle_document(self, pages, mupdf_doc, translate_config, save_debug_image):
        # V6 uses get_no_rotation_img_multiprocess and layout.temp.pdf
        layout_temp_path = translate_config.get_working_file_path("layout.temp.pdf")
        mupdf_doc.save(layout_temp_path.as_posix())

        def process_page_v6(page):
            translate_config.raise_if_cancelled()
            image = get_no_rotation_img_multiprocess(
                layout_temp_path.as_posix(), page.page_number, dpi=self.dpi
            )
            char_boxes = convert_page_to_char_boxes(page)
            lines = process_page_chars_to_lines(char_boxes)
            predict_result = self.predict_image(image, lines=lines)
            save_debug_image(image, predict_result, page.page_number + 1)
            return page, predict_result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            yield from executor.map(process_page_v6, pages)

class RpcDocLayoutModel(DocLayoutModel):
    def __init__(self, host: str, version: str = LayoutVersion.V1):
        self.host = host
        self.version = version
        self.strategy = self._create_strategy(version, host)
        self._stride = 32

    def _create_strategy(self, version: str, host: str) -> LayoutStrategy:
        if version == LayoutVersion.V1:
            return StandardStrategy(host, dpi=72, fixed_resize=True, max_workers=16)
        elif version == LayoutVersion.V2:
            return StandardStrategy(host, dpi=150, max_workers=16)
        elif version == LayoutVersion.V4:
            return StandardStrategy(host, dpi=150, max_workers=1)
        elif version == LayoutVersion.V3:
            return AnalyzeStrategy(host, "analyze")
        elif version == LayoutVersion.V5:
            return AnalyzeStrategy(host, "analyze_hybrid", max_workers=1)
        elif version == LayoutVersion.V6:
            return DualStrategy(host)
        elif version == LayoutVersion.V7:
            return VlmStrategy(host)
        else:
            raise ValueError(f"Unknown layout version: {version}")

    @property
    def stride(self) -> int:
        return self._stride

    def handle_document(
        self,
        pages: List[il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config: Any,
        save_debug_image: Any,
    ) -> Generator[Tuple[il_version_1.Page, YoloResult], None, None]:
        return self.strategy.handle_document(pages, mupdf_doc, translate_config, save_debug_image)

    def init_font_mapper(self, translation_config):
        self.strategy.init_font_mapper(translation_config)

    def predict(self, image, **kwargs):
        # Used for testing/CLI debug mainly
        if isinstance(image, np.ndarray) and len(image.shape) == 3:
            return [self.strategy.predict_image(image, **kwargs)]
        elif isinstance(image, list):
             # Restore parallelism for batch prediction
             result_containers = [None] * len(image)
             def _pred(i, img):
                 result_containers[i] = self.strategy.predict_image(img, **kwargs)

             with ThreadPoolExecutor(max_workers=self.strategy.max_workers) as executor:
                  list(executor.map(_pred, range(len(image)), image))
             return result_containers
        return []

    @staticmethod
    def from_host(host: str, version: str = LayoutVersion.V1) -> "RpcDocLayoutModel":
        return RpcDocLayoutModel(host=host, version=version)
