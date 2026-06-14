from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from autoware_ml_model_launchers.open_tracker.backends.base import BBoxTrackerBackend
from autoware_ml_model_launchers.open_tracker.geometry import bbox_iou
from autoware_ml_model_launchers.open_tracker.types import Detection, Track


@dataclass
class _TrackState:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: str
    class_id: int | None
    source_index: int | None
    age: int = 1
    hits: int = 1
    time_since_update: int = 0

    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def update_from_detection(self, det: Detection) -> None:
        self.x1 = det.x1
        self.y1 = det.y1
        self.x2 = det.x2
        self.y2 = det.y2
        self.score = det.score
        self.label = det.label
        self.class_id = det.class_id
        self.source_index = det.source_index
        self.hits += 1
        self.time_since_update = 0

    def to_track(self, min_hits: int) -> Track:
        return Track(
            track_id=self.track_id,
            x1=self.x1,
            y1=self.y1,
            x2=self.x2,
            y2=self.y2,
            score=self.score,
            label=self.label,
            class_id=self.class_id,
            source_index=self.source_index,
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
            is_confirmed=self.hits >= min_hits,
        )


class SimpleIouTrackerBackend(BBoxTrackerBackend):
    """
    Small dependency-free ROI tracker.

    This is intentionally simple. It is useful as a deterministic smoke-test
    backend and as a first comparison against Autoware ByteTrack. It does not
    use a Kalman filter, velocity model, or image features.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        max_missed: int = 5,
        min_hits: int = 1,
        score_threshold: float = 0.0,
        class_agnostic: bool = False,
        emit_unmatched_tracks: bool = False,
    ) -> None:
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self.score_threshold = float(score_threshold)
        self.class_agnostic = bool(class_agnostic)
        self.emit_unmatched_tracks = bool(emit_unmatched_tracks)
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._next_track_id = 1
        self._tracks.clear()

    def update(
        self,
        detections: Sequence[Detection],
        stamp_sec: float | None = None,
    ) -> list[Track]:
        del stamp_sec
        valid_detections = [
            det for det in detections if det.is_valid() and det.score >= self.score_threshold
        ]

        # Age existing tracks before association. New tracks created in this call
        # start at age 1.
        for track in self._tracks.values():
            track.age += 1

        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        candidate_pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for det_idx, det in enumerate(valid_detections):
                if not self.class_agnostic and not self._same_class(track, det):
                    continue
                iou = bbox_iou(track.bbox(), det.bbox_xyxy())
                if iou >= self.iou_threshold:
                    candidate_pairs.append((iou, track_id, det_idx))

        candidate_pairs.sort(key=lambda item: item[0], reverse=True)

        for _, track_id, det_idx in candidate_pairs:
            if track_id in matched_track_ids or det_idx in matched_det_indices:
                continue
            self._tracks[track_id].update_from_detection(valid_detections[det_idx])
            matched_track_ids.add(track_id)
            matched_det_indices.add(det_idx)

        for track_id, track in list(self._tracks.items()):
            if track_id not in matched_track_ids:
                track.time_since_update += 1

        for det_idx, det in enumerate(valid_detections):
            if det_idx in matched_det_indices:
                continue
            track_id = self._allocate_track_id()
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                score=det.score,
                label=det.label,
                class_id=det.class_id,
                source_index=det.source_index,
            )
            matched_track_ids.add(track_id)

        self._drop_stale_tracks()

        tracks = []
        for track in self._tracks.values():
            if track.hits < self.min_hits:
                continue
            if track.time_since_update == 0 or self.emit_unmatched_tracks:
                tracks.append(track.to_track(min_hits=self.min_hits))
        tracks.sort(key=lambda t: t.track_id)
        return tracks

    def _allocate_track_id(self) -> int:
        track_id = self._next_track_id
        self._next_track_id += 1
        return track_id

    def _drop_stale_tracks(self) -> None:
        for track_id, track in list(self._tracks.items()):
            if track.time_since_update > self.max_missed:
                del self._tracks[track_id]

    @staticmethod
    def _same_class(track: _TrackState, det: Detection) -> bool:
        if track.class_id is not None and det.class_id is not None:
            return track.class_id == det.class_id
        return track.label == det.label
