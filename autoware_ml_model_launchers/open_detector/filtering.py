from __future__ import annotations

import json
from typing import Dict, Iterable, List, Mapping, Sequence, Set

from .types import Detection


DEFAULT_DRIVING_CLASS_FILTER = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
]

DEFAULT_DRIVING_LABEL_MAP = {
    "person": "PEDESTRIAN",
    "pedestrian": "PEDESTRIAN",
    "car": "CAR",
    "bus": "BUS",
    "truck": "TRUCK",
    "bicycle": "BICYCLE",
    "motorcycle": "MOTORCYCLE",
    "motorbike": "MOTORCYCLE",
    "traffic light": "TRAFFIC_LIGHT",
    "stop sign": "TRAFFIC_SIGN",
}


def normalize_label(label: str) -> str:
    return " ".join(str(label).strip().lower().replace("_", "-").split()).replace("-", " ")


def parse_string_list(value: object) -> List[str]:
    """
    Parse ROS/CLI style list values.

    Accepts a Python list/tuple, JSON list string, comma-separated string, or empty value.
    """
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            parsed = json.loads(s)
            if not isinstance(parsed, list):
                raise ValueError("Expected a JSON list")
            return [str(v).strip() for v in parsed if str(v).strip()]
        return [v.strip() for v in s.split(",") if v.strip()]
    if isinstance(value, Sequence):
        return [str(v).strip() for v in value if str(v).strip()]
    raise TypeError(f"Unsupported list value type: {type(value).__name__}")


def parse_label_map(value: object) -> Dict[str, str]:
    """
    Parse label mapping.

    Supported forms:
    - dict: {"person": "PEDESTRIAN"}
    - JSON dict string
    - list/string entries: "person=PEDESTRIAN,car=CAR" or "person:PEDESTRIAN"
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {normalize_label(str(k)): str(v).strip() for k, v in value.items() if str(k).strip()}
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        if s.startswith("{"):
            parsed = json.loads(s)
            if not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object for label_map")
            return parse_label_map(parsed)
        entries = parse_string_list(s)
    else:
        entries = parse_string_list(value)

    out: Dict[str, str] = {}
    for entry in entries:
        if "=" in entry:
            raw, mapped = entry.split("=", 1)
        elif ":" in entry:
            raw, mapped = entry.split(":", 1)
        else:
            raise ValueError(f"Invalid label map entry {entry!r}; use raw=mapped")
        raw = normalize_label(raw)
        mapped = mapped.strip()
        if raw and mapped:
            out[raw] = mapped
    return out


def parse_class_filter(value: object) -> Set[str]:
    return {normalize_label(v) for v in parse_string_list(value)}


def apply_filter_and_mapping(
    detections: Iterable[Detection],
    class_filter: Set[str],
    label_map: Mapping[str, str],
    min_score: float = 0.0,
    max_det: int = 0,
) -> List[Detection]:
    """
    Filter by raw label, then map labels for output.

    Keeping the filter on raw labels avoids surprises when mapped labels are Autoware-like
    strings such as `PEDESTRIAN` while backends emit COCO strings such as `person`.
    """
    out: List[Detection] = []
    for det in detections:
        if not det.valid() or det.score < min_score:
            continue
        raw = normalize_label(det.label)
        if class_filter and raw not in class_filter:
            continue
        mapped = label_map.get(raw, det.label)
        out.append(det.with_label(mapped))

    out.sort(key=lambda d: d.score, reverse=True)
    if max_det and max_det > 0:
        out = out[:max_det]
    return out
