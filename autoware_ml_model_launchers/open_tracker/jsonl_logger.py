from __future__ import annotations

import json
from pathlib import Path
from typing import TextIO

from autoware_ml_model_launchers.open_tracker.types import Track


class TrackJsonlLogger:
    def __init__(self, path: str | None) -> None:
        self.path = Path(path).expanduser() if path else None
        self._file: TextIO | None = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("a", encoding="utf-8")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def write_frame(
        self,
        stamp_sec: float | None,
        frame_id: str,
        tracker_backend: str,
        tracks: list[Track],
    ) -> None:
        if self._file is None:
            return
        for track in tracks:
            record = {
                "stamp": stamp_sec,
                "frame_id": frame_id,
                "tracker_backend": tracker_backend,
                "track_id": track.track_id,
                "label": track.label,
                "class_id": track.class_id,
                "score": track.score,
                "bbox_xyxy": [track.x1, track.y1, track.x2, track.y2],
                "source_index": track.source_index,
                "age": track.age,
                "hits": track.hits,
                "time_since_update": track.time_since_update,
                "is_confirmed": track.is_confirmed,
            }
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def __enter__(self) -> "TrackJsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
