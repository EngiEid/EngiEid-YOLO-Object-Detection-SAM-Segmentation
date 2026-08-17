from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO, SAM


class YOLOSAMPipeline:
    """YOLO object detection + SAM promptable segmentation."""

    def __init__(
        self,
        yolo_weights: str = "yolo26n.pt",
        sam_weights: str = "sam_b.pt",
        device: Optional[str] = None,
    ):
        self.device = device
        self.yolo = YOLO(yolo_weights)
        self.sam = SAM(sam_weights)

    def detect(self, image: np.ndarray, conf: float = 0.25):
        """Run YOLO object detection and return the Ultralytics results."""
        kwargs = {"source": image, "conf": conf, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        return self.yolo.predict(**kwargs)

    def segment_from_box(
        self,
        image: np.ndarray,
        box: Tuple[float, float, float, float],
    ):
        """Segment the object selected by an XYXY bounding box."""
        x1, y1, x2, y2 = map(float, box)
        kwargs = {"source": image, "bboxes": [[x1, y1, x2, y2]], "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        return self.sam.predict(**kwargs)

    def segment_from_point(
        self,
        image: np.ndarray,
        point: Tuple[float, float],
    ):
        """Segment the object selected by a foreground point."""
        x, y = map(float, point)
        kwargs = {
            "source": image,
            "points": [[x, y]],
            "labels": [1],
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        return self.sam.predict(**kwargs)

    @staticmethod
    def first_mask(results) -> Optional[np.ndarray]:
        """Extract the highest-ranked/first SAM mask as a boolean numpy array."""
        if not results:
            return None
        result = results[0]
        if result.masks is None or result.masks.data is None:
            return None
        masks = result.masks.data.detach().cpu().numpy()
        if len(masks) == 0:
            return None
        return masks[0].astype(bool)

    @staticmethod
    def overlay_mask(
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Overlay a segmentation mask and draw its bounding box."""
        output = image.copy()
        if mask.shape[:2] != output.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        # Use a fixed RGB-like overlay color in BGR format.
        overlay = np.zeros_like(output)
        overlay[:, :, 1] = 255
        output[mask] = (
            output[mask].astype(np.float32) * (1 - alpha)
            + overlay[mask].astype(np.float32) * alpha
        ).astype(np.uint8)

        ys, xs = np.where(mask)
        if len(xs):
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output

    @staticmethod
    def yolo_annotated_image(image: np.ndarray, results) -> np.ndarray:
        """Render YOLO detections on an image."""
        if not results:
            return image.copy()
        return results[0].plot()

    @staticmethod
    def image_to_pil(image: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def load_image(file_or_path) -> np.ndarray:
    """Load an uploaded file/path into a BGR OpenCV image."""
    if isinstance(file_or_path, (str, Path)):
        image = cv2.imread(str(file_or_path))
        if image is None:
            raise ValueError(f"Could not read image: {file_or_path}")
        return image

    data = file_or_path.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode the uploaded image.")
    return image
