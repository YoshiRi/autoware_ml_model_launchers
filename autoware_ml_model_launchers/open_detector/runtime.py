from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Mapping, Sequence, Set

import numpy as np

from .filtering import apply_filter_and_mapping
from .types import Detection


@dataclass(frozen=True)
class TimingStats:
    runs: int
    mean: float | None
    median: float | None
    min_ms: float | None
    max_ms: float | None

    @classmethod
    def from_values(cls, values_ms: Sequence[float]) -> "TimingStats":
        values = list(values_ms)
        return cls(
            runs=len(values),
            mean=statistics.mean(values) if values else None,
            median=statistics.median(values) if values else None,
            min_ms=min(values) if values else None,
            max_ms=max(values) if values else None,
        )

    def to_dict(self) -> dict:
        return {
            "runs": self.runs,
            "mean": self.mean,
            "median": self.median,
            "min": self.min_ms,
            "max": self.max_ms,
        }


@dataclass(frozen=True)
class DetectorRuntimeResult:
    raw_detections: list[Detection]
    detections: list[Detection]
    infer_ms: float
    image_size_hw: tuple[int, int]


class OpenDetectorRuntime:
    """ROS-free detector runtime for one image frame."""

    def __init__(
        self,
        backend,
        *,
        class_filter: Set[str],
        label_map: Mapping[str, str],
        max_det: int,
    ) -> None:
        self.backend = backend
        self.class_filter = class_filter
        self.label_map = label_map
        self.max_det = int(max_det)

    def update(self, image_bgr: np.ndarray) -> DetectorRuntimeResult:
        start = time.perf_counter()
        raw = self.backend.infer(image_bgr)
        infer_ms = (time.perf_counter() - start) * 1000.0

        height, width = image_bgr.shape[:2]
        clipped = [det.clipped((height, width)) for det in raw]
        detections = apply_filter_and_mapping(
            clipped,
            class_filter=self.class_filter,
            label_map=self.label_map,
            min_score=0.0,
            max_det=self.max_det,
        )
        return DetectorRuntimeResult(
            raw_detections=list(raw),
            detections=detections,
            infer_ms=infer_ms,
            image_size_hw=(int(height), int(width)),
        )
