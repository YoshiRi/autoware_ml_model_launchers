from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from autoware_ml_model_launchers.open_tracker.backends.base import BBoxTrackerBackend
from autoware_ml_model_launchers.open_tracker.types import Detection, Track


class RoboflowTrackersBackend(BBoxTrackerBackend):
    """
    Optional wrapper around roboflow/trackers.

    This backend is lazy-loaded so the package remains usable without installing
    `trackers` and `supervision`. It is intentionally isolated from ROS I/O.
    """

    def __init__(
        self,
        tracker_type: str = "bytetrack",
        class_agnostic: bool = False,
        iou_threshold: float = 0.30,
        max_missed: int = 5,
        min_hits: int = 1,
        score_threshold: float = 0.0,
    ) -> None:
        self.tracker_type = tracker_type.lower().replace("-", "_")
        self.class_agnostic = class_agnostic
        self.tracker_kwargs = {
            "minimum_iou_threshold": float(iou_threshold),
            "lost_track_buffer": int(max_missed),
            "minimum_consecutive_frames": int(min_hits),
        }
        if self.tracker_type in {"bytetrack", "byte_track"}:
            self.tracker_kwargs["high_conf_det_threshold"] = float(score_threshold)
            self.tracker_kwargs["track_activation_threshold"] = float(
                score_threshold
            )
        elif self.tracker_type in {"ocsort", "oc_sort"}:
            self.tracker_kwargs["high_conf_det_threshold"] = float(score_threshold)
        elif self.tracker_type == "sort":
            self.tracker_kwargs["track_activation_threshold"] = float(
                score_threshold
            )
        self._sv = None
        self._tracker = None
        self._label_to_class_id: dict[str, int] = {}
        self._class_id_to_label: dict[int, str] = {}
        self._load()

    def update(
        self,
        detections: Sequence[Detection],
        stamp_sec: float | None = None,
    ) -> list[Track]:
        del stamp_sec
        sv_detections = self._to_supervision_detections(detections)
        tracked = self._tracker.update(sv_detections)
        return self._from_supervision_detections(tracked, detections)

    def reset(self) -> None:
        if self._tracker is not None and hasattr(self._tracker, "reset"):
            self._tracker.reset()
        else:
            self._load()

    def _load(self) -> None:
        try:
            import supervision as sv  # type: ignore
            from trackers import ByteTrackTracker, OCSORTTracker, SORTTracker  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "Roboflow Trackers backend requires: pip install trackers supervision"
            ) from exc

        tracker_map = {
            "bytetrack": ByteTrackTracker,
            "byte_track": ByteTrackTracker,
            "ocsort": OCSORTTracker,
            "oc_sort": OCSORTTracker,
            "sort": SORTTracker,
        }
        if self.tracker_type not in tracker_map:
            supported = ", ".join(sorted(tracker_map))
            raise ValueError(f"Unsupported roboflow tracker '{self.tracker_type}'. Supported: {supported}")
        self._sv = sv
        self._tracker = tracker_map[self.tracker_type](**self.tracker_kwargs)

    def _class_id_for_label(self, label: str, fallback: int | None) -> int:
        if fallback is not None and not self.class_agnostic:
            self._class_id_to_label.setdefault(int(fallback), label)
            return int(fallback)
        if label not in self._label_to_class_id:
            class_id = len(self._label_to_class_id)
            self._label_to_class_id[label] = class_id
            self._class_id_to_label[class_id] = label
        return self._label_to_class_id[label]

    def _to_supervision_detections(self, detections: Sequence[Detection]):
        if self._sv is None:
            raise RuntimeError("Backend not loaded")
        xyxy = []
        confidence = []
        class_id = []
        data_index = []
        for idx, det in enumerate(detections):
            if not det.is_valid():
                continue
            xyxy.append(det.bbox_xyxy())
            confidence.append(float(det.score))
            class_id.append(self._class_id_for_label(det.label, det.class_id))
            data_index.append(idx)
        if not xyxy:
            return self._sv.Detections.empty()
        return self._sv.Detections(
            xyxy=np.asarray(xyxy, dtype=np.float32),
            confidence=np.asarray(confidence, dtype=np.float32),
            class_id=np.asarray(class_id, dtype=np.int32),
            data={"source_index": np.asarray(data_index, dtype=np.int32)},
        )

    def _from_supervision_detections(
        self,
        tracked,
        source_detections: Sequence[Detection],
    ) -> list[Track]:
        if len(tracked) == 0:
            return []
        xyxy = np.asarray(tracked.xyxy, dtype=np.float32)
        confidence = getattr(tracked, "confidence", None)
        class_ids = getattr(tracked, "class_id", None)
        tracker_ids = getattr(tracked, "tracker_id", None)
        source_indices = None
        if hasattr(tracked, "data") and "source_index" in tracked.data:
            source_indices = np.asarray(tracked.data["source_index"], dtype=np.int32)

        tracks: list[Track] = []
        for i, bbox in enumerate(xyxy):
            class_id = int(class_ids[i]) if class_ids is not None else None
            source_index = int(source_indices[i]) if source_indices is not None else None
            label = self._class_id_to_label.get(class_id, str(class_id)) if class_id is not None else ""
            if source_index is not None and 0 <= source_index < len(source_detections):
                source = source_detections[source_index]
                label = source.label
                class_id = source.class_id
            track_id = int(tracker_ids[i]) if tracker_ids is not None else i + 1
            if track_id < 0:
                continue
            score = float(confidence[i]) if confidence is not None else 1.0
            tracks.append(
                Track(
                    track_id=track_id,
                    x1=float(bbox[0]),
                    y1=float(bbox[1]),
                    x2=float(bbox[2]),
                    y2=float(bbox[3]),
                    score=score,
                    label=label,
                    class_id=class_id,
                    source_index=source_index,
                )
            )
        return tracks
