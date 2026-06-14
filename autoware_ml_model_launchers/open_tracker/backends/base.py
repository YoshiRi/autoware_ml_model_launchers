from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from autoware_ml_model_launchers.open_tracker.types import Detection, Track


class BBoxTrackerBackend(ABC):
    """
    Tracker interface for ROI-only tracking.

    Backends must not import ROS. They receive normalized detections and return
    normalized tracks. ROS conversion belongs to the node layer.
    """

    @abstractmethod
    def update(
        self,
        detections: Sequence[Detection],
        stamp_sec: float | None = None,
    ) -> list[Track]:
        raise NotImplementedError

    def reset(self) -> None:
        """Reset tracker state. Optional for stateless or third-party backends."""
        return None
