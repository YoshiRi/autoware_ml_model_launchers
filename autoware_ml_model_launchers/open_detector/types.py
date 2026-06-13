from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Detection:
    """
    Backend-neutral 2D detection in original image pixel coordinates.

    Coordinates are xyxy with origin at the top-left corner of the original image.
    Backends must return detections in this coordinate system.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    class_id: Optional[int] = None
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return self.x1 + self.width * 0.5

    @property
    def cy(self) -> float:
        return self.y1 + self.height * 0.5

    def valid(self) -> bool:
        return self.width > 0.0 and self.height > 0.0 and self.score >= 0.0 and bool(self.label)

    def clipped(self, image_shape: Tuple[int, int]) -> "Detection":
        """Clip the bbox to an image shape `(height, width)` and return a new Detection."""
        height, width = image_shape[:2]
        x1 = min(max(float(self.x1), 0.0), max(float(width - 1), 0.0))
        y1 = min(max(float(self.y1), 0.0), max(float(height - 1), 0.0))
        x2 = min(max(float(self.x2), 0.0), max(float(width - 1), 0.0))
        y2 = min(max(float(self.y2), 0.0), max(float(height - 1), 0.0))
        return Detection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            score=float(self.score),
            label=str(self.label),
            class_id=self.class_id,
            source=self.source,
            metadata=dict(self.metadata),
        )

    def with_label(self, label: str) -> "Detection":
        return Detection(
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            score=self.score,
            label=label,
            class_id=self.class_id,
            source=self.source,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox_xyxy": [float(self.x1), float(self.y1), float(self.x2), float(self.y2)],
            "bbox_cxcywh": [float(self.cx), float(self.cy), float(self.width), float(self.height)],
            "score": float(self.score),
            "label": str(self.label),
            "class_id": self.class_id,
            "source": self.source,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BackendConfig:
    """
    Common config passed to backend constructors.

    Backends can ignore fields that do not apply. Put unusual backend-specific options
    under `extra` so ROS parameters and CLI arguments do not grow without bound.
    """

    backend: str
    model: str = ""
    device: str = ""
    imgsz: int = 960
    conf_thres: float = 0.25
    iou_thres: float = 0.70
    max_det: int = 100
    half: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def detections_to_dicts(detections: Iterable[Detection]) -> List[Dict[str, Any]]:
    return [det.to_dict() for det in detections]
