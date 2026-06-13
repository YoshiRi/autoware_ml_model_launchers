#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import List

from .backend_loader import create_backend
from .drawing import draw_detections
from .filtering import (
    DEFAULT_DRIVING_CLASS_FILTER,
    DEFAULT_DRIVING_LABEL_MAP,
    apply_filter_and_mapping,
    parse_class_filter,
    parse_label_map,
)
from .image_io import read_image_bgr, write_image_bgr
from .types import BackendConfig, detections_to_dicts


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run one detector backend on a local image without ROS.")
    p.add_argument("--input", "-i", required=True, help="Input image path")
    p.add_argument("--output", "-o", default="", help="Annotated image output path")
    p.add_argument("--json", default="", help="Detection JSON output path")
    p.add_argument("--backend", default="ultralytics", choices=["ultralytics", "yolo", "dfine", "rfdetr", "fake"])
    p.add_argument("--model", default="", help="Model name/path. Empty means backend default.")
    p.add_argument("--device", default="", help="Backend device, e.g. cpu, 0, cuda:0")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--half", action="store_true")
    p.add_argument("--class-filter", default=",".join(DEFAULT_DRIVING_CLASS_FILTER), help="Comma list, JSON list, or empty for all")
    p.add_argument("--label-map", default="", help="Comma entries like person=PEDESTRIAN,car=CAR or JSON object")
    p.add_argument("--default-driving-label-map", action="store_true")
    p.add_argument("--repeat", type=int, default=1, help="Run inference N times for simple timing")
    p.add_argument("--warmup", type=int, default=0, help="Warmup runs excluded from timing")
    return p


def main(argv: List[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    image = read_image_bgr(args.input)

    label_map = parse_label_map(args.label_map)
    if args.default_driving_label_map:
        merged = dict(DEFAULT_DRIVING_LABEL_MAP)
        merged.update(label_map)
        label_map = merged
    class_filter = parse_class_filter(args.class_filter)

    config = BackendConfig(
        backend=args.backend,
        model=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        half=args.half,
    )
    backend = create_backend(config)
    backend.load()

    last = []
    timings_ms: List[float] = []
    total_runs = max(1, args.repeat) + max(0, args.warmup)
    for i in range(total_runs):
        t0 = time.perf_counter()
        raw = backend.infer(image)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if i >= args.warmup:
            timings_ms.append(elapsed_ms)
        last = raw

    height, width = image.shape[:2]
    processed = apply_filter_and_mapping(
        [d.clipped((height, width)) for d in last],
        class_filter=class_filter,
        label_map=label_map,
        min_score=0.0,
        max_det=args.max_det,
    )

    if args.output:
        drawn = draw_detections(image, processed)
        write_image_bgr(args.output, drawn)

    payload = {
        "input": str(args.input),
        "backend": args.backend,
        "model": args.model or "<backend-default>",
        "image_size_hw": [int(height), int(width)],
        "num_raw_detections": len(last),
        "num_output_detections": len(processed),
        "timing_ms": {
            "runs": len(timings_ms),
            "mean": statistics.mean(timings_ms) if timings_ms else None,
            "median": statistics.median(timings_ms) if timings_ms else None,
            "min": min(timings_ms) if timings_ms else None,
            "max": max(timings_ms) if timings_ms else None,
        },
        "detections": detections_to_dicts(processed),
    }

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
