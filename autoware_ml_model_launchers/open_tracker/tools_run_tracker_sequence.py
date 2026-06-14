#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from autoware_ml_model_launchers.open_tracker.backend_factory import create_bbox_tracker_backend
from autoware_ml_model_launchers.open_tracker.runtime import ReusableBBoxTrackerRuntime
from autoware_ml_model_launchers.open_tracker.types import Detection, Track


def _load_frames(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("frames", []))
    if isinstance(data, list):
        return data
    raise ValueError("Input JSON must be a list of frames or {'frames': [...]}")


def _frame_to_detections(frame: dict) -> list[Detection]:
    detections = []
    for idx, item in enumerate(frame.get("detections", [])):
        bbox = item.get("bbox") or item.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            raise ValueError(f"Detection {idx} is missing bbox/bbox_xyxy")
        detections.append(
            Detection(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                score=float(item.get("score", 1.0)),
                label=str(item.get("label", "object")),
                class_id=item.get("class_id"),
                source_index=idx,
            )
        )
    return detections


def _track_to_dict(track: Track) -> dict:
    return {
        "track_id": track.track_id,
        "bbox": [track.x1, track.y1, track.x2, track.y2],
        "score": track.score,
        "label": track.label,
        "class_id": track.class_id,
        "source_index": track.source_index,
        "age": track.age,
        "hits": track.hits,
        "time_since_update": track.time_since_update,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ROI tracker on a JSON frame sequence without ROS.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="simple_iou")
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--max-missed", type=int, default=5)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--class-agnostic", action="store_true")
    args = parser.parse_args()

    backend = create_bbox_tracker_backend(
        args.backend,
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
        min_hits=args.min_hits,
        score_threshold=args.score_threshold,
        class_agnostic=args.class_agnostic,
    )
    runtime = ReusableBBoxTrackerRuntime(backend)
    output_frames = []
    for frame_index, frame in enumerate(_load_frames(args.input)):
        stamp_sec = frame.get("stamp", float(frame_index))
        tracks = runtime.update(_frame_to_detections(frame), stamp_sec=stamp_sec)
        output_frames.append(
            {
                "frame_index": frame_index,
                "stamp": stamp_sec,
                "tracks": [_track_to_dict(track) for track in tracks],
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"frames": output_frames}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
