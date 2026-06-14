from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Detection:
    """Backend-neutral 2D detection in original image pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    class_id: int | None = None
    source_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def is_valid(self) -> bool:
        return self.width > 0.0 and self.height > 0.0

    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Track:
    """Tracked 2D ROI emitted by a tracker backend."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    class_id: int | None = None
    source_index: int | None = None
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    is_confirmed: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def bbox_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)
