from __future__ import annotations


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = bbox_area((ix1, iy1, ix2, iy2))
    if inter <= 0.0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def xyxy_to_cxcywh(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> tuple[float, float, float, float]:
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    return x1 + 0.5 * w, y1 + 0.5 * h, w, h


def cxcywh_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> tuple[float, float, float, float]:
    return cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h
