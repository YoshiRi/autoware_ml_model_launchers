#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
from .runtime import OpenDetectorRuntime, TimingStats
from .types import BackendConfig, detections_to_dicts


DEFAULT_EXTENSIONS = ["jpg", "jpeg", "png", "bmp", "webp"]


@dataclass(frozen=True)
class BackendSpec:
    backend: str
    display_name: str
    model: str = ""
    prompt_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedBackendSpec:
    backend: BackendSpec
    device: str


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: dict, indent: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=indent, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def discover_images(
    input_path: Path,
    *,
    recursive: bool = False,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    limit: int = 0,
    sort: str = "name",
) -> List[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    normalized_extensions = {ext.lower().lstrip(".") for ext in extensions if ext}
    if input_path.is_file():
        if input_path.suffix.lower().lstrip(".") not in normalized_extensions:
            raise ValueError(f"Input file extension is not enabled: {input_path}")
        return [input_path.resolve()]

    globber = input_path.rglob if recursive else input_path.glob
    images = [
        path.resolve()
        for path in globber("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in normalized_extensions
    ]
    if sort == "name":
        images.sort(key=lambda path: str(path))
    elif sort == "mtime":
        images.sort(key=lambda path: (path.stat().st_mtime, str(path)))
    else:
        raise ValueError(f"Unsupported sort: {sort}")
    if limit > 0:
        images = images[:limit]
    return images


def parse_model_overrides(entries: Sequence[str]) -> Dict[str, str]:
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


def parse_prompt_sets(entries: Sequence[str]) -> Dict[str, tuple[str, ...]]:
    prompt_sets: Dict[str, tuple[str, ...]] = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid --prompt-set entry {entry!r}; use name=class1,class2")
        name, value = entry.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid --prompt-set entry {entry!r}; name is empty")
        classes = tuple(parse_string_list(value))
        if not classes:
            raise ValueError(f"Invalid --prompt-set entry {entry!r}; class list is empty")
        prompt_sets[name] = classes
    return prompt_sets


def build_backend_specs(
    backend_names: str,
    *,
    model_overrides: Dict[str, str],
    prompt_classes: Sequence[str],
    prompt_sets: Dict[str, tuple[str, ...]],
) -> List[BackendSpec]:
    specs: List[BackendSpec] = []
    seen = set()
    for item in backend_names.split(","):
        if not item.strip():
            continue
        backend = canonical_backend_name(item)
        if backend in seen and backend != "yolo_world":
            continue
        seen.add(backend)
        model = model_overrides.get(backend, "")
        if backend == "yolo_world" and prompt_sets:
            for set_name, classes in prompt_sets.items():
                specs.append(
                    BackendSpec(
                        backend=backend,
                        display_name=f"{backend}__{set_name}",
                        model=model,
                        prompt_classes=classes,
                    )
                )
        else:
            specs.append(
                BackendSpec(
                    backend=backend,
                    display_name=backend,
                    model=model,
                    prompt_classes=tuple(prompt_classes) if backend == "yolo_world" else (),
                )
            )
    if not specs:
        raise ValueError("No backends selected")
    return specs


def resolve_device(backend: str, requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    if backend == "fake":
        return ""
    if backend not in {"ultralytics", "yolo_world", "dfine"}:
        return ""

    try:
        import torch
    except ImportError:
        return "cpu"

    if backend in {"ultralytics", "yolo_world"}:
        mps = getattr(torch.backends, "mps", None)
        if platform.system() == "Darwin" and mps is not None and mps.is_available():
            return "mps"

    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _image_payload(path: Path, image) -> dict:
    height, width = image.shape[:2]
    return {
        "path": str(path),
        "name": path.name,
        "size_hw": [int(height), int(width)],
    }


def _backend_config_from_args(spec: BackendSpec, args, device: str) -> BackendConfig:
    return BackendConfig(
        backend=spec.backend,
        model=spec.model,
        device=device,
        imgsz=args.imgsz,
        conf_thres=args.conf,
        iou_thres=args.iou,
        max_det=args.max_det,
        half=args.half,
        extra={"classes": list(spec.prompt_classes)},
    )


def _run_runtime(runtime: OpenDetectorRuntime, image, args):
    last_result = None
    timings_ms: List[float] = []
    total_runs = max(1, args.repeat) + max(0, args.warmup)
    for index in range(total_runs):
        result = runtime.update(image)
        if index >= args.warmup:
            timings_ms.append(result.infer_ms)
        last_result = result
    if last_result is None:
        raise RuntimeError("Detector did not run")
    return last_result, TimingStats.from_values(timings_ms)


def _record_for_result(
    path: Path,
    image,
    spec: BackendSpec,
    args,
    device: str,
    result,
    timing: TimingStats,
) -> dict:
    return {
        "image": _image_payload(path, image),
        "backend": spec.backend,
        "display_name": spec.display_name,
        "model": spec.model or "<backend-default>",
        "device": device,
        "prompt_classes": list(spec.prompt_classes),
        "status": "ok",
        "timing_ms": timing.to_dict(),
        "num_raw_detections": len(result.raw_detections),
        "num_output_detections": len(result.detections),
        "labels": sorted({det.label for det in result.detections}),
        "detections": detections_to_dicts(result.detections),
    }


def _failure_record(
    spec: BackendSpec | None,
    path: Path | None,
    status: str,
    exc: Exception,
    trace: bool,
) -> dict:
    record = {
        "image": {"path": str(path), "name": path.name} if path is not None else None,
        "backend": spec.backend if spec is not None else None,
        "display_name": spec.display_name if spec is not None else None,
        "model": (spec.model or "<backend-default>") if spec is not None else None,
        "prompt_classes": list(spec.prompt_classes) if spec is not None else [],
        "status": status,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if trace:
        record["traceback"] = traceback.format_exc()
    return record


def _aggregate(records: Iterable[dict], failures: Iterable[dict]) -> dict:
    backend_summary: Dict[str, dict] = {}
    for record in records:
        name = record["display_name"]
        summary = backend_summary.setdefault(
            name,
            {
                "backend": record["backend"],
                "model": record["model"],
                "prompt_classes": record["prompt_classes"],
                "status_counts": {},
                "timings_ms": [],
                "raw_detection_counts": [],
                "output_detection_counts": [],
                "labels": {},
            },
        )
        status = record["status"]
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        if record["timing_ms"]["mean"] is not None:
            summary["timings_ms"].append(record["timing_ms"]["mean"])
        summary["raw_detection_counts"].append(record["num_raw_detections"])
        summary["output_detection_counts"].append(record["num_output_detections"])
        for det in record["detections"]:
            label = det["label"]
            summary["labels"][label] = summary["labels"].get(label, 0) + 1

    for failure in failures:
        name = failure.get("display_name") or "<image>"
        summary = backend_summary.setdefault(
            name,
            {
                "backend": failure.get("backend"),
                "model": failure.get("model"),
                "prompt_classes": failure.get("prompt_classes", []),
                "status_counts": {},
                "timings_ms": [],
                "raw_detection_counts": [],
                "output_detection_counts": [],
                "labels": {},
            },
        )
        status = failure.get("status", "error")
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1

    for summary in backend_summary.values():
        timing = TimingStats.from_values(summary.pop("timings_ms"))
        raw_counts = summary.pop("raw_detection_counts")
        output_counts = summary.pop("output_detection_counts")
        summary["timing_ms"] = timing.to_dict()
        summary["raw_detection_count"] = _count_stats(raw_counts)
        summary["output_detection_count"] = _count_stats(output_counts)

    return backend_summary


def _count_stats(values: Sequence[int]) -> dict:
    if not values:
        return {"runs": 0, "min": None, "max": None, "sum": 0}
    return {"runs": len(values), "min": min(values), "max": max(values), "sum": sum(values)}


def _write_report(path: Path, summary: dict, failures: List[dict]) -> None:
    lines = [
        "# Open Detector Benchmark Report",
        "",
        f"- Images: {summary['input_images']}",
        f"- Backend runs: {summary['backend_runs']}",
        f"- Failures: {len(failures)}",
        f"- Git commit: {summary.get('git_commit') or '<unknown>'}",
        "",
        "## Backends",
        "",
        "| Backend | Status | Mean ms | Detections | Labels |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for name, item in summary["backends"].items():
        statuses = ", ".join(f"{k}:{v}" for k, v in sorted(item["status_counts"].items()))
        mean = item["timing_ms"]["mean"]
        mean_text = f"{mean:.1f}" if mean is not None else ""
        detections = item["output_detection_count"]["sum"]
        labels = ", ".join(f"{k}:{v}" for k, v in sorted(item["labels"].items()))
        lines.append(f"| {name} | {statuses} | {mean_text} | {detections} | {labels} |")

    if failures:
        lines.extend(["", "## Failures", "", "| Backend | Image | Error |", "| --- | --- | --- |"])
        for failure in failures:
            image = failure["image"]["name"] if failure.get("image") else ""
            error = f"{failure['error_type']}: {failure['error']}".replace("|", "\\|")
            lines.append(f"| {failure.get('display_name') or ''} | {image} | {error} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(args) -> int:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detections_path = output_dir / "detections.jsonl"
    failures_path = output_dir / "failures.jsonl"
    for path in (detections_path, failures_path):
        if path.exists():
            path.unlink()
        path.touch()

    images = discover_images(
        Path(args.input),
        recursive=args.recursive,
        extensions=parse_string_list(args.extensions),
        limit=args.limit,
        sort=args.sort,
    )
    if not images:
        raise ValueError(f"No images found under input: {args.input}")
    if args.repeat < 1:
        raise ValueError("--repeat must be greater than or equal to 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be greater than or equal to 0")
    if args.limit < 0:
        raise ValueError("--limit must be greater than or equal to 0")
    model_overrides = parse_model_overrides(args.model)
    prompt_sets = parse_prompt_sets(args.prompt_set)
    prompt_classes = parse_string_list(args.prompt_classes)
    specs = build_backend_specs(
        args.backends,
        model_overrides=model_overrides,
        prompt_classes=prompt_classes,
        prompt_sets=prompt_sets,
    )
    resolved_specs = [
        ResolvedBackendSpec(spec, resolve_device(spec.backend, args.device))
        for spec in specs
    ]

    label_map = parse_label_map(args.label_map)
    if args.default_driving_label_map:
        merged = dict(DEFAULT_DRIVING_LABEL_MAP)
        merged.update(label_map)
        label_map = merged
    class_filter = parse_class_filter(args.class_filter)

    records: List[dict] = []
    failures: List[dict] = []

    for resolved in resolved_specs:
        spec = resolved.backend
        try:
            backend = create_backend(_backend_config_from_args(spec, args, resolved.device))
            runtime = OpenDetectorRuntime(
                backend,
                class_filter=class_filter,
                label_map=label_map,
                max_det=args.max_det,
            )
        except ImportError as exc:
            failure = _failure_record(spec, None, "missing_dependency", exc, args.traceback)
            failures.append(failure)
            _append_jsonl(failures_path, failure)
            if args.fail_fast:
                break
            continue
        except Exception as exc:
            failure = _failure_record(spec, None, "load_error", exc, args.traceback)
            failures.append(failure)
            _append_jsonl(failures_path, failure)
            if args.fail_fast:
                break
            continue

        for image_path in images:
            try:
                image = read_image_bgr(image_path)
            except Exception as exc:
                failure = _failure_record(spec, image_path, "image_error", exc, args.traceback)
                failures.append(failure)
                _append_jsonl(failures_path, failure)
                if args.fail_fast:
                    break
                continue

            try:
                result, timing = _run_runtime(runtime, image, args)
                record = _record_for_result(
                    image_path,
                    image,
                    spec,
                    args,
                    resolved.device,
                    result,
                    timing,
                )
                records.append(record)
                _append_jsonl(detections_path, record)
                if args.annotated:
                    annotated = output_dir / "annotated" / spec.display_name / image_path.name
                    write_image_bgr(annotated, draw_detections(image, result.detections))
            except Exception as exc:
                failure = _failure_record(spec, image_path, "inference_error", exc, args.traceback)
                failures.append(failure)
                _append_jsonl(failures_path, failure)
                if args.fail_fast:
                    break
        if args.fail_fast and failures:
            break

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": getattr(args, "command", sys.argv),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "input": str(Path(args.input).resolve()),
        "input_images": len(images),
        "backend_runs": len(records),
        "requested_device": args.device,
        "resolved_devices": {
            resolved.backend.display_name: resolved.device for resolved in resolved_specs
        },
        "output_dir": str(output_dir.resolve()),
        "detections_jsonl": str(detections_path.resolve()),
        "failures_jsonl": str(failures_path.resolve()),
        "backends": _aggregate(records, failures),
    }
    _write_json(output_dir / "summary.json", summary, args.json_indent)
    if args.report == "markdown":
        _write_report(output_dir / "report.md", summary, failures)
    return 1 if failures else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark open detector backends on local images."
    )
    parser.add_argument("--input", required=True, help="Input image file or directory")
    parser.add_argument("--output-dir", required=True, help="Benchmark output directory")
    parser.add_argument("--backends", default="fake", help="Comma list of detector backends")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Backend-specific model override",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Backend device, e.g. auto, cpu, mps, 0, cuda:0",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--class-filter", default=",".join(DEFAULT_DRIVING_CLASS_FILTER))
    parser.add_argument(
        "--label-map",
        default="",
        help="Comma entries like person=PEDESTRIAN,car=CAR or JSON object",
    )
    parser.add_argument("--default-driving-label-map", action="store_true")
    parser.add_argument("--prompt-classes", default="", help="Open-vocabulary classes")
    parser.add_argument(
        "--prompt-set",
        action="append",
        default=[],
        help="Named prompt set: name=class1,class2",
    )
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--extensions", default=",".join(DEFAULT_EXTENSIONS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sort", choices=["name", "mtime"], default="name")
    parser.add_argument("--annotated", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report", choices=["markdown", "none"], default="markdown")
    parser.add_argument("--json-indent", type=int, default=2)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--traceback", action="store_true")
    return parser


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.command = [sys.argv[0]] + (list(argv) if argv is not None else sys.argv[1:])
    try:
        return run_benchmark(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
