import ast
import logging
import platform
import re
import threading
from collections.abc import Generator

import cv2
import numpy as np

from babeldoc.docvision.base_doclayout import DocLayoutModel
from babeldoc.docvision.base_doclayout import YoloResult
from babeldoc.format.pdf.document_il.utils.mupdf_helper import get_no_rotation_img

try:
    import onnx
    import onnxruntime
except ImportError as e:
    if "DLL load failed" in str(e):
        raise OSError(
            "Microsoft Visual C++ Redistributable is not installed. "
            "Download it at https://aka.ms/vs/17/release/vc_redist.x64.exe"
        ) from e
    raise
import pymupdf

import babeldoc.format.pdf.document_il.il_version_1
from babeldoc.assets.assets import get_doclayout_onnx_model_path

# from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


# 检测操作系统类型
os_name = platform.system()


class OnnxModel(DocLayoutModel):
    _FIXED_IMGSZ = 1024  # fixed input size for static-shape CoreML

    def __init__(self, model_path: str):
        self.model_path = model_path

        model = onnx.load(model_path)
        metadata = {d.key: d.value for d in model.metadata_props}
        self._stride = ast.literal_eval(metadata["stride"])
        self._names = ast.literal_eval(metadata["names"])
        providers = []

        available_providers = onnxruntime.get_available_providers()
        use_coreml = os_name == "Darwin" and any(
            re.match(r"coreml", p, re.IGNORECASE) for p in available_providers
        )
        if use_coreml:
            # Fix input shapes to [1, 3, 1024, 1024] so CoreML can take over
            # 96%+ of the graph nodes (658/681) instead of just 3/823.
            model = self._make_static_model(model, self._FIXED_IMGSZ)
            providers = [
                "CoreMLExecutionProvider",
                "CPUExecutionProvider",
            ]
            logger.info(
                "Using CoreMLExecutionProvider with static input "
                f"[1, 3, {self._FIXED_IMGSZ}, {self._FIXED_IMGSZ}]"
            )
        else:
            for provider in available_providers:
                if re.match(r"cpu", provider, re.IGNORECASE):
                    logger.info(f"Available Provider: {provider}")
                    providers.append(provider)
        self.model = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=providers,
        )
        self.lock = threading.Lock()

    @staticmethod
    def _make_static_model(model, imgsz):
        """Rewrite dynamic input dims to fixed [1, 3, imgsz, imgsz] and run
        shape inference so all intermediate shapes become static."""
        from onnx import shape_inference

        for inp in model.graph.input:
            if inp.name == "images":
                dim = inp.type.tensor_type.shape.dim
                for i, val in enumerate([1, 3, imgsz, imgsz]):
                    dim[i].ClearField("dim_param")
                    dim[i].dim_value = val
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass  # best-effort; CoreML still benefits from fixed input
        return model

    @staticmethod
    def from_pretrained():
        pth = get_doclayout_onnx_model_path()
        return OnnxModel(pth)

    @property
    def stride(self):
        return self._stride

    def resize_and_pad_image(self, image, new_shape):
        """
        Resize and pad the image to the specified size, ensuring dimensions are multiples of stride.

        Parameters:
        - image: Input image
        - new_shape: Target size (integer or (height, width) tuple)
        - stride: Padding alignment stride, default 32

        Returns:
        - Processed image
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
            image,
            (resized_w, resized_h),
            interpolation=cv2.INTER_LINEAR,
        )

        # Pad to full target size (enables static-shape CoreML acceleration)
        pad_w = new_w - resized_w
        pad_h = new_h - resized_h
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2

        # Add padding
        image = cv2.copyMakeBorder(
            image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        return image

    def scale_boxes(self, img1_shape, boxes, img0_shape):
        """
        Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
        specified in (img1_shape) to the shape of a different image (img0_shape).

        Args:
            img1_shape (tuple): The shape of the image that the bounding boxes are for,
                in the format of (height, width).
            boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
            img0_shape (tuple): the shape of the target image, in the format of (height, width).

        Returns:
            boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
        """

        # Calculate scaling ratio
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])

        # Calculate padding size
        pad_x = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1)
        pad_y = round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1)

        # Remove padding and scale boxes
        boxes[..., :4] = (boxes[..., :4] - [pad_x, pad_y, pad_x, pad_y]) / gain
        return boxes

    def predict(self, image, imgsz=800, batch_size=16, **kwargs):
        """
        Predict the layout of document pages.

        Args:
            image: A single image or a list of images of document pages.
            imgsz: Resize the image to this size. Must be a multiple of the stride.
            batch_size: Number of images to process in one batch.
            **kwargs: Additional arguments.

        Returns:
            A list of YoloResult objects, one for each input image.
        """
        # Handle single image input
        if isinstance(image, np.ndarray) and len(image.shape) == 3:
            image = [image]

        total_images = len(image)
        results = []
        batch_size = 1

        # Process images in batches
        for i in range(0, total_images, batch_size):
            batch_images = image[i : i + batch_size]
            batch_size_actual = len(batch_images)

            # Calculate target size based on the maximum height in the batch
            max_height = max(img.shape[0] for img in batch_images)
            target_imgsz = 1024

            # Preprocess batch
            processed_batch = []
            orig_shapes = []
            for img in batch_images:
                orig_h, orig_w = img.shape[:2]
                orig_shapes.append((orig_h, orig_w))

                pix = self.resize_and_pad_image(img, new_shape=target_imgsz)
                pix = np.transpose(pix, (2, 0, 1))  # CHW
                pix = pix.astype(np.float32) / 255.0  # Normalize to [0, 1]
                processed_batch.append(pix)

            # Stack batch
            batch_input = np.stack(processed_batch, axis=0)  # BCHW
            new_h, new_w = batch_input.shape[2:]

            # Run inference
            batch_preds = self.model.run(None, {"images": batch_input})[0]

            # Process each prediction in the batch
            for j in range(batch_size_actual):
                preds = batch_preds[j]
                preds = preds[preds[..., 4] > 0.25]
                if len(preds) > 0:
                    preds[..., :4] = self.scale_boxes(
                        (new_h, new_w),
                        preds[..., :4],
                        orig_shapes[j],
                    )
                results.append(YoloResult(boxes_data=preds, names=self._names))

        return results

    def handle_document(
        self,
        pages: list[babeldoc.format.pdf.document_il.il_version_1.Page],
        mupdf_doc: pymupdf.Document,
        translate_config,
        save_debug_image,
    ) -> Generator[
        tuple[babeldoc.format.pdf.document_il.il_version_1.Page, YoloResult], None, None
    ]:
        for page in pages:
            translate_config.raise_if_cancelled()
            with self.lock:
                # pix = mupdf_doc[page.page_number].get_pixmap(dpi=72)
                pix = get_no_rotation_img(mupdf_doc[page.page_number])
            image = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height,
                pix.width,
                3,
            )[:, :, ::-1]
            predict_result = self.predict(image)[0]
            save_debug_image(
                image,
                predict_result,
                page.page_number + 1,
            )
            yield page, predict_result
