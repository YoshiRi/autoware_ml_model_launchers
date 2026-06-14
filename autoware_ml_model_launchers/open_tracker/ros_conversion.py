from __future__ import annotations

import copy
import uuid
from typing import Any

from autoware_ml_model_launchers.open_tracker.geometry import xyxy_to_cxcywh
from autoware_ml_model_launchers.open_tracker.types import Detection, Track


_TRACK_UUID_NAMESPACE = uuid.UUID("53d55e20-8a4b-4b4a-b2d9-ddbcfcf845d5")


def stamp_to_sec(stamp: Any) -> float | None:
    if stamp is None:
        return None
    try:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9
    except AttributeError:
        return None


def detections_from_detection2d_array(msg: Any) -> list[Detection]:
    detections: list[Detection] = []
    for idx, ros_det in enumerate(getattr(msg, "detections", [])):
        bbox = getattr(ros_det, "bbox", None)
        if bbox is None:
            continue
        cx, cy = _read_center_xy(bbox.center)
        width = float(getattr(bbox, "size_x", 0.0))
        height = float(getattr(bbox, "size_y", 0.0))
        if width <= 0.0 or height <= 0.0:
            continue
        label, score, class_id = _read_best_hypothesis(ros_det)
        detections.append(
            Detection(
                x1=cx - 0.5 * width,
                y1=cy - 0.5 * height,
                x2=cx + 0.5 * width,
                y2=cy + 0.5 * height,
                score=score,
                label=label,
                class_id=class_id,
                source_index=idx,
            )
        )
    return detections


def tracks_to_detection2d_array(tracks: list[Track], header: Any) -> Any:
    from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

    msg = Detection2DArray()
    msg.header = header
    for track in tracks:
        ros_det = Detection2D()
        ros_det.header = header
        _write_bbox(ros_det.bbox, track)
        ros_det.id = str(track.track_id)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = str(track.label)
        hypothesis.hypothesis.score = float(track.score)
        ros_det.results.append(hypothesis)
        msg.detections.append(ros_det)
    return msg


def detections_from_autoware(msg: Any) -> list[Detection]:
    detections: list[Detection] = []
    for idx, feature_object in enumerate(getattr(msg, "feature_objects", [])):
        roi = feature_object.feature.roi
        width = float(roi.width)
        height = float(roi.height)
        if width <= 0.0 or height <= 0.0:
            continue
        class_id, label = _read_autoware_classification(feature_object.object)
        detections.append(
            Detection(
                x1=float(roi.x_offset),
                y1=float(roi.y_offset),
                x2=float(roi.x_offset) + width,
                y2=float(roi.y_offset) + height,
                score=float(feature_object.object.existence_probability),
                label=label,
                class_id=class_id,
                source_index=idx,
            )
        )
    return detections


def tracks_to_autoware(tracks: list[Track], source_msg: Any) -> tuple[Any, Any]:
    from autoware_perception_msgs.msg import ObjectClassification
    from tier4_perception_msgs.msg import (
        DetectedObjectWithFeature,
        DetectedObjectsWithFeature,
        DynamicObject,
        DynamicObjectArray,
    )

    output = DetectedObjectsWithFeature()
    output.header = source_msg.header
    track_ids = DynamicObjectArray()
    track_ids.header = source_msg.header
    source_objects = list(getattr(source_msg, "feature_objects", []))

    for track in tracks:
        if track.source_index is not None and 0 <= track.source_index < len(source_objects):
            feature_object = copy.deepcopy(source_objects[track.source_index])
        else:
            feature_object = DetectedObjectWithFeature()
            classification = ObjectClassification()
            classification.label = int(track.class_id or 0)
            classification.probability = 1.0
            feature_object.object.classification.append(classification)

        feature_object.object.existence_probability = float(track.score)
        _write_roi(feature_object.feature.roi, track)
        output.feature_objects.append(feature_object)

        dynamic_object = DynamicObject()
        dynamic_object.id.uuid = list(_track_uuid(track.track_id).bytes)
        track_ids.objects.append(dynamic_object)

    return output, track_ids


def _track_uuid(track_id: int) -> uuid.UUID:
    return uuid.uuid5(_TRACK_UUID_NAMESPACE, str(track_id))


def _read_center_xy(center: Any) -> tuple[float, float]:
    if hasattr(center, "x") and hasattr(center, "y"):
        return float(center.x), float(center.y)
    if hasattr(center, "position"):
        return float(center.position.x), float(center.position.y)
    raise AttributeError("Unsupported BoundingBox2D.center layout")


def _write_bbox(bbox: Any, track: Track) -> None:
    cx, cy, width, height = xyxy_to_cxcywh(track.x1, track.y1, track.x2, track.y2)
    center = bbox.center
    if hasattr(center, "x") and hasattr(center, "y"):
        center.x = float(cx)
        center.y = float(cy)
        if hasattr(center, "theta"):
            center.theta = 0.0
    elif hasattr(center, "position"):
        center.position.x = float(cx)
        center.position.y = float(cy)
        if hasattr(center, "orientation"):
            center.orientation.w = 1.0
    else:
        raise AttributeError("Unsupported BoundingBox2D.center layout")
    bbox.size_x = float(width)
    bbox.size_y = float(height)


def _write_roi(roi: Any, track: Track) -> None:
    x_offset = max(0, int(round(track.x1)))
    y_offset = max(0, int(round(track.y1)))
    roi.x_offset = x_offset
    roi.y_offset = y_offset
    roi.width = max(0, int(round(track.x2)) - x_offset)
    roi.height = max(0, int(round(track.y2)) - y_offset)


def _read_best_hypothesis(ros_det: Any) -> tuple[str, float, int | None]:
    results = list(getattr(ros_det, "results", []))
    if not results:
        return "", 1.0, None
    best = max(
        results,
        key=lambda item: float(getattr(getattr(item, "hypothesis", item), "score", 0.0)),
    )
    hypothesis = getattr(best, "hypothesis", best)
    raw_class_id = getattr(hypothesis, "class_id", "")
    score = float(getattr(hypothesis, "score", 1.0))
    label = str(raw_class_id)
    try:
        numeric_class_id = int(raw_class_id)
    except (TypeError, ValueError):
        numeric_class_id = None
    return label, score, numeric_class_id


def _read_autoware_classification(detected_object: Any) -> tuple[int | None, str]:
    classifications = list(getattr(detected_object, "classification", []))
    if not classifications:
        return None, ""
    best = max(classifications, key=lambda item: float(item.probability))
    class_id = int(best.label)
    return class_id, str(class_id)
