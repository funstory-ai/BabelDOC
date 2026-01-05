import abc
import base64
import json
import logging
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Generator

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
from babeldoc.docvision.base_doclayout import DocLayoutModel
from babeldoc.docvision.base_doclayout import YoloBox
from babeldoc.docvision.base_doclayout import YoloResult
from babeldoc.format.pdf.document_il import il_version_1
from babeldoc.format.pdf.document_il.utils.extract_char import (
    convert_page_to_char_boxes,
)
from babeldoc.format.pdf.document_il.utils.extract_char import (
    process_page_chars_to_lines,
)
from babeldoc.format.pdf.document_il.utils.fontmap import FontMapper
from babeldoc.format.pdf.document_il.utils.layout_helper import SPACE_REGEX
from babeldoc.format.pdf.document_il.utils.mupdf_helper import (
    get_no_rotation_img,
    get_no_rotation_img_multiprocess,
)

logger = logging.getLogger(__name__)

DPI = 150

def encode_image(image) -> bytes:
    """Read and encode image to bytes

    Args:
        image: Can be either a file path (str) or numpy array
    """
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


class ResultContainer:
    def __init__(self):
        self.result = YoloResult(boxes_data=np.array([]), names=[])


class LayoutStrategy(abc.ABC):
    def __init__(self, host: str):
        self.host = host
        self.lock = threading.Lock()

    @abc.abstractmethod
    def handle_document(
        self,
        pages: list[il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ) -> Generator[tuple[il_version_1.Page, YoloResult], None, None]:
        pass

    def init_font_mapper(self, translation_config):
        pass

    def resize_and_pad_image(self, image, new_shape):
        """
        Resize and pad the image to the specified size,
        ensuring dimensions are multiples of stride.
        """
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h, w = image.shape[:2]
        new_h, new_w = new_shape

        # Calculate scaling ratio
        r = min(new_h / h, new_w / w)
        resized_h, resized_w = int(round(h * r)), int(round(w * r))

        # Resize image
        image = cv2.resize(
            image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )

        # Calculate padding size
        pad_h = new_h - resized_h
        pad_w = new_w - resized_w
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        # Add padding
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return image

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        """
        Rescales bounding boxes.
        """
        # Calculate scaling ratio
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

        # Calculate padding size
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)

        # Remove padding and scale boxes
        boxes = (boxes - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes


class BaseDocLayoutStrategy(LayoutStrategy):
    """Base class for strategies that share predict_page logic"""

    def __init__(self, host: str, dpi: int, max_workers: int):
        super().__init__(host)
        self.dpi = dpi
        self.max_workers = max_workers

    def predict_image(
        self,
        image,
        host: str | None = None,
        result_container: ResultContainer | None = None,
        imgsz: int = 1024,
    ) -> ResultContainer:
        raise NotImplementedError

    def predict_page(
        self, page, mupdf_doc: pymupdf.Document, translate_config, save_debug_image
    ):
        translate_config.raise_if_cancelled()
        with self.lock:
            # get_no_rotation_img uses default DPI if not specified.
            # BaseDocLayoutStrategy subclasses pass DPI explicitly.
            pix = get_no_rotation_img(mupdf_doc[page.page_number], dpi=self.dpi)

        image = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height,
            pix.width,
            3,
        )[:, :, ::-1]

        # All existing versions use 800 for prediction call within predict_image logic or similar
        predict_result = self.predict_image(image, self.host, None, 800)
        save_debug_image(image, predict_result, page.page_number + 1)
        return page, predict_result

    def handle_document(
        self,
        pages: list[il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            yield from executor.map(
                self.predict_page,
                pages,
                (mupdf_doc for _ in range(len(pages))),
                (translate_config for _ in range(len(pages))),
                (save_debug_image for _ in range(len(pages))),
            )


class LayoutStrategyV1(BaseDocLayoutStrategy):
    def __init__(self, host: str):
        super().__init__(host, dpi=72, max_workers=16)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Request failed, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/3)"
        ),
    )
    def _predict_layout(self, image, host: str, imgsz: int = 1024):
        if not isinstance(image, list):
            image = [image]
        image_data = [encode_image(image) for image in image]
        data = {
            "image": image_data,
            "imgsz": imgsz,
        }
        packed_data = msgpack.packb(data, use_bin_type=True)
        response = httpx.post(
            f"{host}/inference",
            data=packed_data,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            timeout=300,
            follow_redirects=True,
        )

        if response.status_code == 200:
            try:
                result = msgpack.unpackb(response.content, raw=False)
                return result
            except Exception as e:
                logger.exception(f"Failed to unpack response: {e!s}")
                raise
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response content: {response.content}")
            raise Exception(
                f"Request failed with status {response.status_code}: {response.text}",
            )

    def predict_image(
        self,
        image,
        host: str | None = None,
        result_container: ResultContainer | None = None,
        imgsz: int = 1024,
    ) -> ResultContainer:
        if result_container is None:
            result_container = ResultContainer()

        target_imgsz = (800, 800)
        orig_h, orig_w = image.shape[:2]
        if image.shape[0] != target_imgsz[0] or image.shape[1] != target_imgsz[1]:
            image = self.resize_and_pad_image(image, new_shape=target_imgsz)

        preds = self._predict_layout([image], host=self.host, imgsz=800)

        if len(preds) > 0:
            for pred in preds:
                boxes = [
                    YoloBox(
                        None,
                        self.scale_boxes(
                            (800, 800), np.array(x["xyxy"]), (orig_h, orig_w)
                        ),
                        np.array(x["conf"]),
                        x["cls"],
                    )
                    for x in pred["boxes"]
                ]
                result_container.result = YoloResult(
                    boxes=boxes,
                    names={int(k): v for k, v in pred["names"].items()},
                )
        return result_container.result


class LayoutStrategyV2(BaseDocLayoutStrategy):
    def __init__(self, host: str, max_workers=16):
        super().__init__(host, dpi=DPI, max_workers=max_workers)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Request failed, retrying in {getattr(retry_state.next_action, 'sleep', 'unknown')} seconds... "
            f"(Attempt {retry_state.attempt_number}/3)"
        ),
    )
    def _predict_layout(self, image, host: str, imgsz: int = 1024):
        if not isinstance(image, list):
            image = [image]
        image_data = [encode_image(image) for image in image]
        data = {
            "image": image_data,
        }
        packed_data = msgpack.packb(data, use_bin_type=True)
        response = httpx.post(
            f"{host}/inference",
            data=packed_data,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            timeout=480,
            follow_redirects=True,
        )

        idx = 0
        id_lookup = {}
        if response.status_code == 200:
            try:
                result = msgpack.unpackb(response.content, raw=False)
                useful_result = []
                if isinstance(result, dict):
                    names = {}
                    for box in result["boxes"]:
                        if box["score"] < 0.7:
                            continue

                        box["xyxy"] = box["coordinate"]
                        box["conf"] = box["score"]
                        if box["label"] not in names:
                            idx += 1
                            names[idx] = box["label"]
                            box["cls_id"] = idx
                            id_lookup[box["label"]] = idx
                        else:
                            box["cls_id"] = id_lookup[box["label"]]
                        names[box["cls_id"]] = box["label"]
                        box["cls"] = box["cls_id"]
                        useful_result.append(box)
                    if "names" not in result:
                        result["names"] = names
                    result["boxes"] = useful_result
                    result = [result]
                return result
            except Exception as e:
                logger.exception(f"Failed to unpack response: {e!s}")
                raise
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response content: {response.content}")
            raise Exception(
                f"Request failed with status {response.status_code}: {response.text}",
            )

    def predict_image(
        self,
        image,
        host: str | None = None,
        result_container: ResultContainer | None = None,
        imgsz: int = 1024,
    ) -> ResultContainer:
        if result_container is None:
            result_container = ResultContainer()
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (orig_h, orig_w)
        # In V2, image is resized if dimensions don't match target_imgsz, but target_imgsz is set to orig_h, orig_w...
        # So resizing happens only if image.shape != (orig_h, orig_w), which is never true unless image changed.
        # Wait, reading V2 code:
        # orig_h, orig_w = image.shape[:2]
        # target_imgsz = (orig_h, orig_w)
        # if image.shape[0] != target_imgsz[0] or image.shape[1] != target_imgsz[1]:
        #    image = self.resize_and_pad_image(image, new_shape=target_imgsz)
        # This seems redundant in original code, but I'll keep logic.

        preds = self._predict_layout(image, host=self.host)

        # Scale back to 72 DPI
        # orig_h/w are in 150 DPI (from predict_page), so convert to 72
        orig_h_72, orig_w_72 = orig_h / DPI * 72, orig_w / DPI * 72

        if len(preds) > 0:
            for pred in preds:
                boxes = [
                    YoloBox(
                        None,
                        self.scale_boxes(
                            target_imgsz, np.array(x["xyxy"]), (orig_h_72, orig_w_72)
                        ),
                        np.array(x["conf"]),
                        x["cls"],
                    )
                    for x in pred["boxes"]
                ]
                result_container.result = YoloResult(
                    boxes=boxes,
                    names={int(k): v for k, v in pred["names"].items()},
                )
        return result_container.result


class LayoutStrategyV3(BaseDocLayoutStrategy):
    def __init__(self, host: str, max_workers=4):
        super().__init__(host, dpi=DPI, max_workers=max_workers)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Request failed, retrying in {getattr(retry_state.next_action, 'sleep', 'unknown')} seconds... "
            f"(Attempt {retry_state.attempt_number}/3)"
        ),
    )
    def _predict_layout(self, image, host: str, imgsz: int = 1024):
        image_data = encode_image(image)
        response = httpx.post(
            f"{host}/analyze?min_sim=0.7&early_stop=0.99&timeout=1800",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            headers={
                "Accept": "application/json",
            },
            timeout=1800,
            follow_redirects=True,
        )

        idx = 0
        id_lookup = {}
        if response.status_code == 200:
            try:
                result = json.loads(response.text)
                useful_result = []
                if isinstance(result, dict):
                    names = {}
                    for box in result["boxes"]:
                        if box["ocr_match_score"] < 0.7:
                            continue

                        box["xyxy"] = box["coords"]
                        box["conf"] = box["ocr_match_score"]
                        if box["label"] not in names:
                            idx += 1
                            names[idx] = box["label"]
                            box["cls_id"] = idx
                            id_lookup[box["label"]] = idx
                        else:
                            box["cls_id"] = id_lookup[box["label"]]
                        names[box["cls_id"]] = box["label"]
                        box["cls"] = box["cls_id"]
                        useful_result.append(box)
                    if "names" not in result:
                        result["names"] = names
                    result["boxes"] = useful_result
                    result = [result]
                return result
            except Exception as e:
                logger.exception(f"Failed to unpack response: {e!s}")
                raise
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response content: {response.content}")
            raise Exception(
                f"Request failed with status {response.status_code}: {response.text}",
            )

    def predict_image(
        self,
        image,
        host: str | None = None,
        result_container: ResultContainer | None = None,
        imgsz: int = 1024,
    ) -> ResultContainer:
        if result_container is None:
            result_container = ResultContainer()
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (orig_h, orig_w)

        preds = self._predict_layout(image, host=self.host)

        orig_h_72, orig_w_72 = orig_h / DPI * 72, orig_w / DPI * 72

        if len(preds) > 0:
            for pred in preds:
                boxes = [
                    YoloBox(
                        None,
                        self.scale_boxes(
                            target_imgsz, np.array(x["xyxy"]), (orig_h_72, orig_w_72)
                        ),
                        np.array(x["conf"]),
                        x["cls"],
                    )
                    for x in pred["boxes"]
                ]
                result_container.result = YoloResult(
                    boxes=boxes,
                    names={int(k): v for k, v in pred["names"].items()},
                )
        return result_container.result


class LayoutStrategyV4(LayoutStrategyV2):
    def __init__(self, host: str):
        # V4 shares predict_layout with V2 but has max_workers=1
        super().__init__(host, max_workers=1)


class LayoutStrategyV5(LayoutStrategyV3):
    def __init__(self, host: str):
        # V5 has max_workers=1
        super().__init__(host, max_workers=1)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Request failed, retrying in {getattr(retry_state.next_action, 'sleep', 'unknown')} seconds... "
            f"(Attempt {retry_state.attempt_number}/3)"
        ),
    )
    def _predict_layout(self, image, host: str, imgsz: int = 1024):
        image_data = encode_image(image)
        # URL changes from analyze to analyze_hybrid
        response = httpx.post(
            f"{host}/analyze_hybrid?min_sim=0.7&early_stop=0.99&timeout=1800",
            files={"file": ("image.jpg", image_data, "image/jpeg")},
            headers={
                "Accept": "application/json",
            },
            timeout=1800,
            follow_redirects=True,
        )

        idx = 0
        id_lookup = {}
        if response.status_code == 200:
            try:
                result = json.loads(response.text)
                useful_result = []
                if isinstance(result, dict):
                    names = {}
                    clusters = result["clusters"] # V5 uses clusters
                    for box in clusters:
                        box["xyxy"] = box["box"]
                        box["conf"] = 1
                        if box["label"] not in names:
                            idx += 1
                            names[idx] = box["label"]
                            box["cls_id"] = idx
                            id_lookup[box["label"]] = idx
                        else:
                            box["cls_id"] = id_lookup[box["label"]]
                        names[box["cls_id"]] = box["label"]
                        box["cls"] = box["cls_id"]
                        useful_result.append(box)
                    if "names" not in result:
                        result["names"] = names
                    result["boxes"] = useful_result
                    result = [result]
                return result
            except Exception as e:
                logger.exception(f"Failed to unpack response: {e!s}")
                raise
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response content: {response.text}")
            raise Exception(
                f"Request failed with status {response.status_code}: {response.text}",
            )

class LayoutStrategyV6(LayoutStrategy):
    def __init__(self, host: str):
        super().__init__(host)
        if ";" not in host:
            raise ValueError(
                "LayoutStrategyV6 host must be two hosts separated by ';' (e.g. 'http://h1;http://h2')"
            )
        self.host1, self.host2 = [h.strip() for h in host.split(";", 1)]
        self.font_mapper = None

    def init_font_mapper(self, translation_config):
        self.font_mapper = FontMapper(translation_config)

    def clip_num(self, num: float, min_value: float, max_value: float) -> float:
        """Clip a number to a specified range."""
        if num < min_value:
            return min_value
        elif num > max_value:
            return max_value
        return num

    def filter_text(self, txt: str):
        normalize = unicodedata.normalize("NFKC", txt)
        unicodes = []
        for c in normalize:
            if self.font_mapper and self.font_mapper.has_char(c):
                unicodes.append(c)
        normalize = "".join(unicodes)
        result = SPACE_REGEX.sub(" ", normalize).strip()
        return result

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Request failed VLM, retrying in {getattr(retry_state.next_action, 'sleep', 'unknown')} seconds... "
            f"(Attempt {retry_state.attempt_number}/5)"
        ),
    )
    def _predict_layout_vlm(self, image, host: str, imgsz: int, lines):
        if lines is None:
            lines = []
        image_data = encode_image(image)

        def convert_line(line):
            if not line.text:
                return None
            boxes = [c[0] for c in line.chars]
            min_x = min(b.x for b in boxes)
            max_x = max(b.x2 for b in boxes)
            min_y = min(b.y for b in boxes)
            max_y = max(b.y2 for b in boxes)

            image_height, image_width = image.shape[:2]

            # Transform to image pixel coordinates
            min_x = min_x / 72 * DPI
            max_x = max_x / 72 * DPI
            min_y = min_y / 72 * DPI
            max_y = max_y / 72 * DPI

            min_y, max_y = image_height - max_y, image_height - min_y

            box_volume = (max_x - min_x) * (max_y - min_y)
            if box_volume < 1:
                return None

            min_x = self.clip_num(min_x, 0, image_width - 1)
            max_x = self.clip_num(max_x, 0, image_width - 1)
            min_y = self.clip_num(min_y, 0, image_height - 1)
            max_y = self.clip_num(max_y, 0, image_height - 1)

            filtered_text = self.filter_text(line.text)
            if not filtered_text:
                return None

            return {"box": [min_x, min_y, max_x, max_y], "text": filtered_text}

        formatted_results = [convert_line(l) for l in lines]
        formatted_results = [r for r in formatted_results if r is not None]
        if not formatted_results:
            return None

        image_b64 = base64.b64encode(image_data).decode("utf-8")

        request_data = {
            "image": image_b64,
            "ocr_results": formatted_results,
            "image_size": list(image.shape[:2])[::-1],  # (height, width)
        }

        response = httpx.post(
            f"{host}/inference",
            json=request_data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=30,
            follow_redirects=True,
        )

        idx = 0
        id_lookup = {}
        if response.status_code == 200:
            try:
                result = json.loads(response.text)
                useful_result = []
                if isinstance(result, dict):
                    names = {}
                    clusters = result["clusters"]
                    for box in clusters:
                        box["xyxy"] = box["box"]
                        box["conf"] = 1
                        if box["label"] not in names:
                            idx += 1
                            names[idx] = box["label"]
                            box["cls_id"] = idx
                            id_lookup[box["label"]] = idx
                        else:
                            box["cls_id"] = id_lookup[box["label"]]
                        names[box["cls_id"]] = box["label"]
                        box["cls"] = box["cls_id"]
                        useful_result.append(box)
                    if "names" not in result:
                        result["names"] = names
                    result["boxes"] = useful_result
                    result = [result]
                return result
            except Exception as e:
                logger.exception(f"Failed to unpack response: {e!s}")
                raise
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response content: {response.text}")
            raise Exception(
                f"Request failed with status {response.status_code}: {response.text}",
            )

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Request failed PADDLE, retrying in {getattr(retry_state.next_action, 'sleep', 'unknown')} seconds... "
            f"(Attempt {retry_state.attempt_number}/5)"
        ),
    )
    def _predict_layout_paddle(self, image, host: str, imgsz: int):
        if not isinstance(image, list):
            image = [image]
        image_data = [encode_image(image) for image in image]
        data = {
            "image": image_data,
        }
        packed_data = msgpack.packb(data, use_bin_type=True)
        response = httpx.post(
            f"{host}/inference",
            data=packed_data,
            headers={
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            },
            timeout=30,
            follow_redirects=True,
        )

        idx = 0
        id_lookup = {}
        if response.status_code == 200:
            try:
                result = msgpack.unpackb(response.content, raw=False)
                useful_result = []
                if isinstance(result, dict):
                    names = {}
                    for box in result["boxes"]:
                        if box["score"] < 0.7:
                            continue
                        box["xyxy"] = box["coordinate"]
                        box["conf"] = box["score"]
                        if box["label"] not in names:
                            idx += 1
                            names[idx] = box["label"]
                            box["cls_id"] = idx
                            id_lookup[box["label"]] = idx
                        else:
                            box["cls_id"] = id_lookup[box["label"]]
                        names[box["cls_id"]] = box["label"]
                        box["cls"] = box["cls_id"]
                        useful_result.append(box)
                    if "names" not in result:
                        result["names"] = names
                    result["boxes"] = useful_result
                    result = [result]
                return result
            except Exception as e:
                logger.exception(f"Failed to unpack response: {e!s}")
                raise
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response content: {response.content}")
            raise Exception(
                f"Request failed with status {response.status_code}: {response.text}",
            )

    def calculate_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0

    def is_subset(self, inner_box, outer_box):
        x1_inner, y1_inner, x2_inner, y2_inner = inner_box
        x1_outer, y1_outer, x2_outer, y2_outer = outer_box
        return (
            x1_inner >= x1_outer
            and y1_inner >= y1_outer
            and x2_inner <= x2_outer
            and y2_inner <= y2_outer
        )

    def expand_box_to_contain(self, box_to_expand, box_to_contain):
        x1_expand, y1_expand, x2_expand, y2_expand = box_to_expand
        x1_contain, y1_contain, x2_contain, y2_contain = box_to_contain
        return [
            min(x1_expand, x1_contain),
            min(y1_expand, y1_contain),
            max(x2_expand, x2_contain),
            max(y2_expand, y2_contain),
        ]

    def post_process_boxes(self, merged_boxes: list[YoloBox], names: dict[int, str]):
        for i, text_box in enumerate(merged_boxes):
            text_label = names.get(text_box.cls, "")
            if "text" not in text_label:
                continue

            for j, para_box in enumerate(merged_boxes):
                if i == j:
                    continue
                para_label = names.get(para_box.cls, "")
                if "paragraph_hybrid" not in para_label:
                    continue

                iou = self.calculate_iou(text_box.xyxy, para_box.xyxy)
                if iou > 0.95 and not self.is_subset(para_box.xyxy, text_box.xyxy):
                    expanded_box = self.expand_box_to_contain(
                        text_box.xyxy, para_box.xyxy
                    )
                    merged_boxes[i] = YoloBox(
                        None,
                        np.array(expanded_box),
                        text_box.conf,
                        text_box.cls,
                    )

    def predict_image(
        self,
        image,
        imgsz: int = 1024,
        lines=None,
    ) -> YoloResult:
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (orig_h, orig_w)
        # V6 original code calls resize_and_pad_image if size != target_imgsz, which it is.
        # But target_imgsz is set to image.shape. So it doesn't resize.

        with ThreadPoolExecutor(max_workers=2) as ex:
            if lines:
                future1 = ex.submit(
                    self._predict_layout_vlm,
                    image,
                    self.host1,
                    imgsz,
                    lines,
                )
            else:
                future1 = None

            future2 = ex.submit(self._predict_layout_paddle, image, self.host2, imgsz)

            preds1 = future1.result() if future1 else None
            preds2 = future2.result()

        pdf_h, pdf_w = orig_h / DPI * 72, orig_w / DPI * 72
        merged_boxes: list[YoloBox] = []
        names: dict[int, str] = {}

        def _process_preds(preds, id_offset: int, label_suffix: str | None):
            for pred in preds or []:
                for box in pred["boxes"]:
                    scaled_xyxy = self.scale_boxes(
                        target_imgsz, np.array(box["xyxy"]), (pdf_h, pdf_w)
                    )
                    new_cls_id = box["cls"] + id_offset
                    label = pred["names"].get(box["cls"], str(box["cls"]))
                    if label_suffix:
                        label = f"{label}{label_suffix}"
                    names[new_cls_id] = label
                    merged_boxes.append(
                        YoloBox(
                            None,
                            scaled_xyxy,
                            np.array(box.get("conf", box.get("score", 1.0))),
                            new_cls_id,
                        )
                    )

        if preds1:
            _process_preds(preds1, 1000, "_hybrid")
        _process_preds(preds2, 2000, None)

        merged_boxes.sort(key=lambda b: b.conf, reverse=True)
        self.post_process_boxes(merged_boxes, names)
        return YoloResult(boxes=merged_boxes, names=names)

    def predict_page(self, page, pdf_bytes: Path, translate_config, save_debug_image):
        translate_config.raise_if_cancelled()
        image = get_no_rotation_img_multiprocess(
            pdf_bytes.as_posix(), page.page_number, dpi=DPI
        )
        char_boxes = convert_page_to_char_boxes(page)
        lines = process_page_chars_to_lines(char_boxes)
        predict_result = self.predict_image(image, 800, lines)
        save_debug_image(image, predict_result, page.page_number + 1)
        return page, predict_result

    def handle_document(
        self,
        pages: list[il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ):
        layout_temp_path = translate_config.get_working_file_path("layout.temp.pdf")
        mupdf_doc.save(layout_temp_path.as_posix())
        with ThreadPoolExecutor(max_workers=32) as executor:
            yield from executor.map(
                self.predict_page,
                pages,
                (layout_temp_path for _ in range(len(pages))),
                (translate_config for _ in range(len(pages))),
                (save_debug_image for _ in range(len(pages))),
            )


class LayoutStrategyV7(LayoutStrategy):
    def __init__(self, host: str):
        super().__init__(host)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, Exception)),
        before_sleep=lambda retry_state: logger.warning(
            f"Request failed, retrying in {getattr(retry_state.next_action, 'sleep', 'unknown')} seconds... "
            f"(Attempt {retry_state.attempt_number}/3)"
        ),
    )
    def _predict_layout(self, image, host: str, lines=None):
        if lines is None:
            lines = []
        image_data = encode_image(image)

        def convert_line(line):
            boxes = [c[0] for c in line.chars]
            min_x = min([b.x for b in boxes])
            max_x = max([b.x2 for b in boxes])
            min_y = min([b.y for b in boxes])
            max_y = max([b.y2 for b in boxes])

            min_x = min_x / 72 * DPI
            max_x = max_x / 72 * DPI
            min_y = min_y / 72 * DPI
            max_y = max_y / 72 * DPI

            image_height = image.shape[0]
            min_y, max_y = image_height - max_y, image_height - min_y

            return {"box": [min_x, min_y, max_x, max_y], "text": line.text}

        formatted_results = [convert_line(l) for l in lines]
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        request_data = {
            "image": image_b64,
            "ocr_results": formatted_results,
            "image_size": list(image.shape[:2])[::-1],
        }

        response = httpx.post(
            f"{host}/inference",
            json=request_data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=1800,
            follow_redirects=True,
        )

        idx = 0
        id_lookup = {}
        if response.status_code == 200:
            try:
                result = json.loads(response.text)
                useful_result = []
                if isinstance(result, dict):
                    names = {}
                    clusters = result["clusters"]
                    for box in clusters:
                        box["xyxy"] = box["box"]
                        box["conf"] = 1
                        if box["label"] not in names:
                            idx += 1
                            names[idx] = box["label"]
                            box["cls_id"] = idx
                            id_lookup[box["label"]] = idx
                        else:
                            box["cls_id"] = id_lookup[box["label"]]
                        names[box["cls_id"]] = box["label"]
                        box["cls"] = box["cls_id"]
                        useful_result.append(box)
                    if "names" not in result:
                        result["names"] = names
                    result["boxes"] = useful_result
                    result = [result]
                return result
            except Exception as e:
                logger.exception(f"Failed to unpack response: {e!s}")
                raise
        else:
            logger.error(f"Request failed with status {response.status_code}")
            logger.error(f"Response content: {response.text}")
            raise Exception(
                f"Request failed with status {response.status_code}: {response.text}",
            )

    def predict_image(
        self,
        image,
        host: str | None = None,
        result_container: ResultContainer | None = None,
        imgsz: int = 1024,
        page: il_version_1.Page | None = None,
    ) -> YoloResult:
        if result_container is None:
            result_container = ResultContainer()
        orig_h, orig_w = image.shape[:2]
        target_imgsz = (orig_h, orig_w)
        # Original V7 code checks for resize
        if image.shape[0] != target_imgsz[0] or image.shape[1] != target_imgsz[1]:
            image = self.resize_and_pad_image(image, new_shape=target_imgsz)

        char_boxes = convert_page_to_char_boxes(page)
        lines = process_page_chars_to_lines(char_boxes)

        preds = self._predict_layout(image, host=self.host, lines=lines)

        orig_h_72, orig_w_72 = orig_h / DPI * 72, orig_w / DPI * 72
        if len(preds) > 0:
            for pred in preds:
                boxes = [
                    YoloBox(
                        None,
                        self.scale_boxes(
                            target_imgsz, np.array(x["xyxy"]), (orig_h_72, orig_w_72)
                        ),
                        np.array(x["conf"]),
                        x["cls"],
                    )
                    for x in pred["boxes"]
                ]
                result_container.result = YoloResult(
                    boxes=boxes,
                    names={int(k): v for k, v in pred["names"].items()},
                )
        return result_container.result

    def predict_page(
        self, page, mupdf_doc: pymupdf.Document, translate_config, save_debug_image
    ):
        translate_config.raise_if_cancelled()
        with self.lock:
            pix = get_no_rotation_img(mupdf_doc[page.page_number], dpi=DPI)
        image = np.frombuffer(pix.samples, np.uint8).reshape(
            pix.height,
            pix.width,
            3,
        )[:, :, ::-1]
        predict_result = self.predict_image(image, self.host, None, 800, page)
        save_debug_image(image, predict_result, page.page_number + 1)
        return page, predict_result

    def handle_document(
        self,
        pages: list[il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ):
        with ThreadPoolExecutor(max_workers=1) as executor:
            yield from executor.map(
                self.predict_page,
                pages,
                (mupdf_doc for _ in range(len(pages))),
                (translate_config for _ in range(len(pages))),
                (save_debug_image for _ in range(len(pages))),
            )


class RpcDocLayoutModel(DocLayoutModel):
    """DocLayoutModel implementation that uses RPC service."""

    def __init__(self, host: str = "http://localhost:8000", version: str = "v1"):
        """Initialize RPC model with host address and version."""
        self.host = host
        self.version = version
        self._stride = 32
        self.strategy = self._get_strategy(version, host)

    def _get_strategy(self, version: str, host: str) -> LayoutStrategy:
        version = version.lower()
        if version == "v1":
            return LayoutStrategyV1(host)
        elif version == "v2":
            return LayoutStrategyV2(host)
        elif version == "v3":
            return LayoutStrategyV3(host)
        elif version == "v4":
            return LayoutStrategyV4(host)
        elif version == "v5":
            return LayoutStrategyV5(host)
        elif version == "v6":
            return LayoutStrategyV6(host)
        elif version == "v7":
            return LayoutStrategyV7(host)
        else:
            raise ValueError(f"Unknown layout version: {version}")

    @property
    def stride(self) -> int:
        return self._stride

    def handle_document(
        self,
        pages: list[il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ):
        yield from self.strategy.handle_document(
            pages, mupdf_doc, translate_config, save_debug_image
        )

    def init_font_mapper(self, translation_config):
        self.strategy.init_font_mapper(translation_config)

    @staticmethod
    def from_host(host: str) -> "RpcDocLayoutModel":
        """Create RpcDocLayoutModel from host address."""
        # This static method is kept for backward compatibility if needed,
        # but usage in main.py will be updated.
        return RpcDocLayoutModel(host=host)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Test the service with V1 (default)
    try:
        # Use a default test image if example/1.png doesn't exist
        image_path = "example/1.png"
        if not Path(image_path).exists():
            print(f"Warning: {image_path} not found.")
            print("Please provide the path to a test image:")
            image_path = input("> ")

        logger.info(f"Processing image: {image_path}")
        # Note: predict_layout function is removed from global scope, must use class
        model = RpcDocLayoutModel(host="http://localhost:8000", version="v1")
        # To test, we would need to mock page/mupdf_doc or expose predict_image in strategy
        # For simplicity, we just print a message that direct test is not supported without refactor.
        print("Please run full application to test.")

    except Exception as e:
        print(f"Error: {e!s}")
