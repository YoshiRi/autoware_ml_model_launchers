from __future__ import annotations

from typing import List

import numpy as np

from .base import BaseDetectorBackend
from ..types import BackendConfig, Detection


class FakeBackend(BaseDetectorBackend):
    """Dependency-free backend for CLI/ROS plumbing tests."""

    def __init__(self, config: BackendConfig, *, autoload: bool = True) -> None:
        super().__init__(config)
        if autoload:
            self.load()

    def infer(self, image_bgr: np.ndarray) -> List[Detection]:
        self.require_loaded()
        height, width = image_bgr.shape[:2]
        return [
            Detection(
                x1=0.18 * width,
                y1=0.20 * height,
                x2=0.52 * width,
                y2=0.72 * height,
                score=0.91,
                label="car",
                class_id=2,
                source="fake",
            ),
            Detection(
                x1=0.58 * width,
                y1=0.18 * height,
                x2=0.78 * width,
                y2=0.62 * height,
                score=0.77,
                label="person",
                class_id=0,
                source="fake",
            ),
        ]
