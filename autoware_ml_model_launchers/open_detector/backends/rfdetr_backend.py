from __future__ import annotations

from typing import Dict, List, Sequence

import cv2
import numpy as np

from .base import BaseDetectorBackend
from ..types import Detection


class RFDetrBackend(BaseDetectorBackend):
    """
    Roboflow RF-DETR backend.

    The rfdetr package returns a supervision.Detections-like object with xyxy,
    confidence, and class_id arrays. This wrapper intentionally avoids importing
    supervision directly.
    """

    DEFAULT_MODEL = "small"

    def load(self) -> None:
        try:
            import rfdetr
            from PIL import Image
        except ImportError as exc:
            raise ImportError(
                "RF-DETR backend selected but rfdetr is not installed. "
                "Install with: python3 -m pip install rfdetr"
            ) from exc

        self.Image = Image
        self.coco_classes = self._load_coco_classes()
        variant = (self.config.model or self.DEFAULT_MODEL).strip().lower().replace("-", "_")
        self.model = self._instantiate_model(rfdetr, variant)
        self.loaded = True

    def _load_coco_classes(self) -> Dict[int, str]:
        try:
            from rfdetr.util.coco_classes import COCO_CLASSES
        except Exception:
            return {}
        if isinstance(COCO_CLASSES, dict):
            return {int(k): str(v) for k, v in COCO_CLASSES.items()}
        if isinstance(COCO_CLASSES, Sequence) and not isinstance(COCO_CLASSES, (str, bytes)):
            return {i: str(v) for i, v in enumerate(COCO_CLASSES)}
        return {}

    @staticmethod
    def _instantiate_model(rfdetr_module, variant: str):
        names = {
            "n": "RFDETRNano",
            "nano": "RFDETRNano",
            "s": "RFDETRSmall",
            "small": "RFDETRSmall",
            "base": "RFDETRBase",
            "b": "RFDETRBase",
            "m": "RFDETRMedium",
            "medium": "RFDETRMedium",
            "l": "RFDETRLarge",
            "large": "RFDETRLarge",
            "xlarge": "RFDETRXLarge",
            "xl": "RFDETRXLarge",
            "2xlarge": "RFDETR2XLarge",
            "2xl": "RFDETR2XLarge",
        }
        class_name = names.get(variant, variant)
        if not hasattr(rfdetr_module, class_name):
            supported = ", ".join(sorted(names))
            raise ValueError(f"Unsupported RF-DETR model variant {variant!r}; supported aliases: {supported}")
        return getattr(rfdetr_module, class_name)()

    def _label_for_class_id(self, class_id: int) -> str:
        if class_id in self.coco_classes:
            return self.coco_classes[class_id]
        # Some COCO helpers are 1-indexed; tolerate that without assuming it.
        if class_id - 1 in self.coco_classes:
            return self.coco_classes[class_id - 1]
        return str(class_id)

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        self.require_loaded()

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = self.Image.fromarray(rgb)
        detections = self.model.predict(pil_image, threshold=float(self.config.conf_thres))

        xyxy = np.asarray(getattr(detections, "xyxy", []), dtype=float)
        confidence = np.asarray(getattr(detections, "confidence", []), dtype=float)
        class_ids = np.asarray(getattr(detections, "class_id", []), dtype=int)

        out: List[Detection] = []
        for box, score, class_id in zip(xyxy, confidence, class_ids):
            class_id_int = int(class_id)
            out.append(
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    score=float(score),
                    label=self._label_for_class_id(class_id_int),
                    class_id=class_id_int,
                    source="rfdetr",
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        if self.config.max_det and self.config.max_det > 0:
            out = out[: int(self.config.max_det)]
        return out
