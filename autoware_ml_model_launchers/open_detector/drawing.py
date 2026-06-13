from __future__ import annotations

from typing import Iterable, Tuple

import cv2
import numpy as np

from .types import Detection


def color_for_class(class_id: int | None, label: str = "") -> Tuple[int, int, int]:
    """Deterministic BGR color. No global state, no random colors."""
    if class_id is None:
        seed = sum(ord(ch) for ch in str(label))
    else:
        seed = int(class_id) * 37
    return ((seed + 53) % 255, (seed * 3 + 97) % 255, (seed * 7 + 193) % 255)


def draw_detections(image_bgr: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    drawn = image_bgr.copy()
    height, width = drawn.shape[:2]
    for det in detections:
        det = det.clipped((height, width))
        x1, y1, x2, y2 = [int(round(v)) for v in (det.x1, det.y1, det.x2, det.y2)]
        if x2 <= x1 or y2 <= y1:
            continue
        color = color_for_class(det.class_id, det.label)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        label = f"{det.label} {det.score:.2f}"
        text_org = (x1, max(0, y1 - 6))
        cv2.putText(drawn, label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return drawn
