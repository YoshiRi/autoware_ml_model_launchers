#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Dict, List

from .backend_loader import canonical_backend_name, create_backend
from .drawing import draw_detections
from .filtering import (
    DEFAULT_DRIVING_CLASS_FILTER,
    DEFAULT_DRIVING_LABEL_MAP,
    parse_class_filter,
    parse_label_map,
    parse_string_list,
)
from .image_io import read_image_bgr, write_image_bgr
from .make_test_image import make_image
from .runtime import OpenDetectorRuntime, TimingStats
from .types import BackendConfig, detections_to_dicts


ALL_BACKENDS = ["fake", "ultralytics", "yolo_world", "dfine", "rfdetr"]


def _parse_backend_list(value: str) -> List[str]:
    backends: List[str] = []
    for item in str(value or "").split(","):
        name = item.strip()
        if not name:
            continue
        canonical = canonical_backend_name(name)
        if canonical not in backends:
            backends.append(canonical)
    return backends


def _parse_model_overrides(entries: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --model entry {entry!r}; use backend=model")
        backend, model = entry.split("=", 1)
        backend = canonical_backend_name(backend)
        model = model.strip()
        if not model:
            raise ValueError(f"Invalid --model entry {entry!r}; model is empty")
        overrides[backend] = model
    return overrides


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run lightweight backend smoke checks and emit a machine-readable report."
    )
    p.add_argument("--input", "-i", default="", help="Input image path. Empty uses an in-memory synthetic image.")
    p.add_argument("--json", default="", help="Write report JSON to this path instead of stdout.")
    p.add_argument("--output-dir", default="", help="Optional directory for annotated images from successful backends.")
    p.add_argument(
        "--backends",
        default="fake",
        help="Comma list: fake,ultralytics,yolo_world,dfine,rfdetr,yolo,d-fine,rf-detr",
    )
    p.add_argument("--all", action="store_true", help="Check fake plus all real backend adapters.")
    p.add_argument(
        "--model",
        action="append",
        default=[],
        help="Backend-specific model override, e.g. ultralytics=/path/yolo.pt or dfine=repo/name",
    )
    p.add_argument("--device", default="", help="Backend device, e.g. cpu, 0, cuda:0")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.70)
    p.add_argument("--max-det", type=int, default=100)
    p.add_argument("--half", action="store_true")
    p.add_argument("--class-filter", default=",".join(DEFAULT_DRIVING_CLASS_FILTER))
    p.add_argument("--label-map", default="", help="Comma entries like person=PEDESTRIAN,car=CAR or JSON object")
    p.add_argument("--prompt-classes", default="", help="Open-vocabulary classes, e.g. car,traffic light")
    p.add_argument("--default-driving-label-map", action="store_true")
    p.add_argument("--repeat", type=int, default=1, help="Timed inference runs per backend")
    p.add_argument("--warmup", type=int, default=0, help="Warmup runs excluded from timing")
    p.add_argument("--fail-on-error", action="store_true", help="Exit non-zero when any selected backend is not ok.")
    p.add_argument("--traceback", action="store_true", help="Include Python traceback strings for backend errors.")
    return p


def _timing_payload(values_ms: List[float]) -> dict:
    return TimingStats.from_values(values_ms).to_dict()


def _run_backend(name: str, image, args, label_map: Dict[str, str], class_filter: set[str], model: str) -> dict:
    config = BackendConfig(
        backend=name,
        model=model,
        device=args.device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        half=args.half,
        extra={"classes": parse_string_list(args.prompt_classes)},
    )
    report = {
        "backend": name,
        "model": model or "<backend-default>",
        "status": "ok",
        "error_type": None,
        "error": None,
        "timing_ms": _timing_payload([]),
        "num_raw_detections": 0,
        "num_output_detections": 0,
        "labels": [],
        "detections": [],
    }

    try:
        backend = create_backend(config)
        runtime = OpenDetectorRuntime(
            backend,
            class_filter=class_filter,
            label_map=label_map,
            max_det=args.max_det,
        )

        last_result = None
        timings_ms: List[float] = []
        total_runs = max(1, args.repeat) + max(0, args.warmup)
        for index in range(total_runs):
            runtime_result = runtime.update(image)
            if index >= args.warmup:
                timings_ms.append(runtime_result.infer_ms)
            last_result = runtime_result
        if last_result is None:
            raise RuntimeError("Detector did not run")

        processed = last_result.detections
        report.update(
            {
                "timing_ms": _timing_payload(timings_ms),
                "num_raw_detections": len(last_result.raw_detections),
                "num_output_detections": len(processed),
                "labels": sorted({det.label for det in processed}),
                "detections": detections_to_dicts(processed),
            }
        )

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            write_image_bgr(output_dir / f"{name}.jpg", draw_detections(image, processed))
    except ImportError as exc:
        report.update({"status": "missing_dependency", "error_type": type(exc).__name__, "error": str(exc)})
        if args.traceback:
            report["traceback"] = traceback.format_exc()
    except Exception as exc:
        report.update({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})
        if args.traceback:
            report["traceback"] = traceback.format_exc()

    return report


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    selected_backends = list(ALL_BACKENDS) if args.all else _parse_backend_list(args.backends)
    model_overrides = _parse_model_overrides(args.model)

    image = read_image_bgr(args.input) if args.input else make_image()
    height, width = image.shape[:2]

    label_map = parse_label_map(args.label_map)
    if args.default_driving_label_map:
        merged = dict(DEFAULT_DRIVING_LABEL_MAP)
        merged.update(label_map)
        label_map = merged
    class_filter = parse_class_filter(args.class_filter)

    backend_results = [
        _run_backend(
            backend,
            image,
            args,
            label_map=label_map,
            class_filter=class_filter,
            model=model_overrides.get(backend, ""),
        )
        for backend in selected_backends
    ]
    summary: Dict[str, int] = {}
    for item in backend_results:
        summary[item["status"]] = summary.get(item["status"], 0) + 1

    payload = {
        "input": str(Path(args.input).resolve()) if args.input else "<synthetic>",
        "image_size_hw": [int(height), int(width)],
        "backends": backend_results,
        "summary": summary,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)

    if args.fail_on_error and any(item["status"] != "ok" for item in backend_results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
