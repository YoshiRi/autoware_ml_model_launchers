from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .base import BaseDetectorBackend
from ..types import Detection


class UltralyticsYoloBackend(BaseDetectorBackend):
    """
    Ultralytics YOLO backend.

    Heavy dependency is imported in load(), not at package import time.
    """

    DEFAULT_MODEL = "yolo26s.pt"

    def load(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics backend selected but ultralytics is not installed. "
                "Install with: python3 -m pip install ultralytics"
            ) from exc

        model_name = self.config.model or self.DEFAULT_MODEL
        self.model = YOLO(model_name)
        self.names = self._extract_names(getattr(self.model, "names", {}))
        self.loaded = True

    @staticmethod
    def _extract_names(names_obj: object) -> Dict[int, str]:
        if isinstance(names_obj, dict):
            return {int(k): str(v) for k, v in names_obj.items()}
        if isinstance(names_obj, Sequence) and not isinstance(names_obj, (str, bytes)):
            return {i: str(v) for i, v in enumerate(names_obj)}
        return {}

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self.loaded:
            self.load()

        results = self.model.predict(
            source=image_bgr,
            imgsz=int(self.config.imgsz),
            conf=float(self.config.conf_thres),
            iou=float(self.config.iou_thres),
            max_det=int(self.config.max_det),
            device=(self.config.device if self.config.device else None),
            half=bool(self.config.half),
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names = self._extract_names(getattr(result, "names", self.names)) or self.names
        xyxy = boxes.xyxy.detach().cpu().numpy()
        conf = boxes.conf.detach().cpu().numpy()
        cls = boxes.cls.detach().cpu().numpy().astype(int)

        out: List[Detection] = []
        for box, score, class_id in zip(xyxy, conf, cls):
            class_id = int(class_id)
            label = names.get(class_id, str(class_id))
            out.append(
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    score=float(score),
                    label=label,
                    class_id=class_id,
                    source="ultralytics",
                )
            )
        return out
