from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

from .base import BaseDetectorBackend
from ..types import Detection


class DFineBackend(BaseDetectorBackend):
    """Hugging Face Transformers D-FINE backend."""

    DEFAULT_MODEL = "ustc-community/dfine-small-obj2coco"

    def load(self) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise ImportError(
                "D-FINE backend selected but required packages are missing. "
                "Install with: python3 -m pip install torch pillow transformers"
            ) from exc

        self.torch = torch
        self.Image = Image
        model_name = self.config.model or self.DEFAULT_MODEL
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)

        device = self._resolve_device(self.config.device, torch)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = self._id2label(getattr(self.model.config, "id2label", {}))
        self.loaded = True

    @staticmethod
    def _resolve_device(device: str, torch_module) -> str:
        if device:
            if device.isdigit():
                return f"cuda:{device}"
            return device
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"

    @staticmethod
    def _id2label(mapping: object) -> Dict[int, str]:
        if isinstance(mapping, dict):
            out: Dict[int, str] = {}
            for k, v in mapping.items():
                try:
                    out[int(k)] = str(v)
                except (TypeError, ValueError):
                    continue
            return out
        return {}

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self.loaded:
            self.load()

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = self.Image.fromarray(rgb)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = self.torch.tensor([(image.height, image.width)], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=float(self.config.conf_thres),
        )
        if not results:
            return []

        result = results[0]
        scores = result.get("scores", [])
        labels = result.get("labels", [])
        boxes = result.get("boxes", [])

        out: List[Detection] = []
        for score, label_id, box in zip(scores, labels, boxes):
            score_f = float(score.detach().cpu().item() if hasattr(score, "detach") else score)
            class_id = int(label_id.detach().cpu().item() if hasattr(label_id, "detach") else label_id)
            box_list = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
            label = self.id2label.get(class_id, str(class_id))
            out.append(
                Detection(
                    x1=float(box_list[0]),
                    y1=float(box_list[1]),
                    x2=float(box_list[2]),
                    y2=float(box_list[3]),
                    score=score_f,
                    label=label,
                    class_id=class_id,
                    source="dfine",
                )
            )
        out.sort(key=lambda d: d.score, reverse=True)
        if self.config.max_det and self.config.max_det > 0:
            out = out[: int(self.config.max_det)]
        return out
