from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .base import BaseDetectorBackend
from ..filtering import parse_string_list
from ..types import BackendConfig, Detection


class YoloWorldBackend(BaseDetectorBackend):
    """
    Ultralytics YOLO-World backend for open-vocabulary 2D detection.

    Prompt classes can be provided through `BackendConfig.extra["classes"]` or
    `BackendConfig.extra["prompt_classes"]`.
    """

    DEFAULT_MODEL = "yolov8s-world.pt"

    def __init__(self, config: BackendConfig, *, autoload: bool = True) -> None:
        super().__init__(config)
        self.prompt_classes: List[str] = []
        if autoload:
            self.load()

    def load(self) -> None:
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise ImportError(
                "YOLO-World backend selected but ultralytics is not installed. "
                "Install with: python3 -m pip install ultralytics"
            ) from exc

        model_name = self.config.model or self.DEFAULT_MODEL
        self.model = YOLOWorld(model_name)
        self.prompt_classes = self._prompt_classes_from_extra(self.config.extra)
        if self.prompt_classes:
            self.model.set_classes(self.prompt_classes)
        self.names = self._extract_names(getattr(self.model, "names", {}))
        self.loaded = True

    @staticmethod
    def _prompt_classes_from_extra(extra: object) -> List[str]:
        if not isinstance(extra, dict):
            return []
        value = extra.get("classes", extra.get("prompt_classes", ""))
        return parse_string_list(value)

    @staticmethod
    def _extract_names(names_obj: object) -> Dict[int, str]:
        if isinstance(names_obj, dict):
            return {int(k): str(v) for k, v in names_obj.items()}
        if isinstance(names_obj, Sequence) and not isinstance(names_obj, (str, bytes)):
            return {i: str(v) for i, v in enumerate(names_obj)}
        return {}

    def _label_for_class_id(self, class_id: int, names: Dict[int, str]) -> str:
        if class_id in names:
            return names[class_id]
        if 0 <= class_id < len(self.prompt_classes):
            return self.prompt_classes[class_id]
        return str(class_id)

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        self.require_loaded()

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
            out.append(
                Detection(
                    x1=float(box[0]),
                    y1=float(box[1]),
                    x2=float(box[2]),
                    y2=float(box[3]),
                    score=float(score),
                    label=self._label_for_class_id(class_id, names),
                    class_id=class_id,
                    source="yolo_world",
                    metadata={"prompt_classes": list(self.prompt_classes)},
                )
            )
        return out
