from __future__ import annotations

from collections.abc import Sequence

from autoware_ml_model_launchers.open_tracker.backends.base import BBoxTrackerBackend
from autoware_ml_model_launchers.open_tracker.types import Detection, Track


class ReusableBBoxTrackerRuntime:
    """Pure logic wrapper between ROS I/O and a tracker backend."""

    def __init__(self, backend: BBoxTrackerBackend) -> None:
        self.backend = backend
        self.frame_count = 0

    def update(
        self,
        detections: Sequence[Detection],
        stamp_sec: float | None = None,
    ) -> list[Track]:
        self.frame_count += 1
        return self.backend.update(detections, stamp_sec=stamp_sec)

    def reset(self) -> None:
        self.frame_count = 0
        self.backend.reset()
