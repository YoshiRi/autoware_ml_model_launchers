from __future__ import annotations

from typing import Any

from autoware_ml_model_launchers.open_tracker.backends.base import BBoxTrackerBackend
from autoware_ml_model_launchers.open_tracker.backends.simple_iou import SimpleIouTrackerBackend


def create_bbox_tracker_backend(name: str, **kwargs: Any) -> BBoxTrackerBackend:
    """
    Create a ROI-only tracker backend.

    Only the selected optional backend is imported. This keeps dependency
    failures local to the backend being tested.
    """
    normalized = name.lower().replace("-", "_")
    if normalized in {"simple_iou", "iou", "fake"}:
        allowed = {
            "iou_threshold",
            "max_missed",
            "min_hits",
            "score_threshold",
            "class_agnostic",
            "emit_unmatched_tracks",
        }
        return SimpleIouTrackerBackend(**{k: v for k, v in kwargs.items() if k in allowed})

    if normalized.startswith("roboflow_"):
        from autoware_ml_model_launchers.open_tracker.backends.roboflow_trackers import (
            RoboflowTrackersBackend,
        )

        tracker_type = normalized.removeprefix("roboflow_")
        return RoboflowTrackersBackend(
            tracker_type=tracker_type,
            class_agnostic=bool(kwargs.get("class_agnostic", False)),
            iou_threshold=float(kwargs.get("iou_threshold", 0.30)),
            max_missed=int(kwargs.get("max_missed", 5)),
            min_hits=int(kwargs.get("min_hits", 1)),
            score_threshold=float(kwargs.get("score_threshold", 0.0)),
        )

    raise ValueError(
        f"Unknown tracker backend '{name}'. "
        "Supported: simple_iou, roboflow_bytetrack, roboflow_ocsort, roboflow_sort"
    )
