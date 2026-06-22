# Open Detector Benchmark Design

## Purpose

The benchmark tool should make open model detector experiments repeatable before they are wired into
ROS or Autoware launch files. It is a local, ROS-free harness for comparing model backends, prompt
sets, thresholds, latency, labels, and annotated outputs over one image or an image directory.

This is not intended to be a formal accuracy benchmark at first. The first version should answer:

- Does this backend load and run on this machine?
- What labels and boxes does it produce for the same image set?
- How sensitive is it to prompts and confidence thresholds?
- How slow is model load and per-image inference?
- What failed, and was the failure dependency, model download, input, or inference related?

The same output format should later feed a small web UI and support segmentation results.

## Non-Goals

- Do not require ROS, Autoware, rosbag playback, or an Autoware workspace.
- Do not introduce Autoware message dependencies into the benchmark core.
- Do not implement mAP or dataset ground-truth evaluation in the first version.
- Do not build a full web UI in the first version.
- Do not commit model weights, generated reports, annotated images, or benchmark datasets.

## Layering

The benchmark should sit above the existing open detector core:

```text
images on disk
  -> image_io.read_image_bgr
  -> backend_loader.create_backend
  -> OpenDetectorRuntime
  -> benchmark result writers
```

It should reuse:

- `BackendConfig`
- `create_backend`
- `OpenDetectorRuntime`
- `detections_to_dicts`
- `draw_detections`
- `read_image_bgr` / `write_image_bgr`
- existing parsing helpers for class filters, label maps, and prompt classes

The benchmark package must not import ROS modules.

## Initial CLI

Proposed module:

```bash
python -m autoware_ml_model_launchers.open_detector.tools_benchmark_images \
  --input /path/to/image_or_dir \
  --output-dir /tmp/open_detector_benchmark \
  --backends ultralytics,yolo_world,dfine \
  --model ultralytics=yolo11n.pt \
  --model yolo_world=yolov8s-world.pt \
  --prompt-classes 'car,bus,person,traffic light,cone' \
  --device auto \
  --class-filter '' \
  --conf 0.25 \
  --repeat 1 \
  --warmup 0
```

Required arguments:

- `--input`: image file or image directory.
- `--output-dir`: report output directory.

Core options:

- `--backends`: comma-separated backend list. Use existing aliases.
- `--model backend=value`: backend-specific model override; repeatable.
- `--device`: backend device. Use `auto` to choose MPS on supported macOS/Ultralytics paths,
  CUDA when available, then CPU.
- `--imgsz`
- `--conf`
- `--iou`
- `--max-det`
- `--half`
- `--class-filter`
- `--label-map`
- `--default-driving-label-map`
- `--prompt-classes`: shared prompt classes for open-vocabulary backends.
- `--prompt-set name=class1,class2,...`: named prompt set; repeatable, for prompt sweeps.
- `--repeat`
- `--warmup`
- `--fail-fast`
- `--traceback`

Input directory options:

- `--recursive`
- `--extensions jpg,jpeg,png,bmp,webp`
- `--limit N`
- `--sort name|mtime`

Output options:

- `--annotated`: write annotated images.
- `--no-annotated`: skip annotated images.
- `--report markdown|none`
- `--json-indent N`

## Output Directory

The first version should produce a stable directory layout:

```text
output-dir/
  summary.json
  detections.jsonl
  failures.jsonl
  report.md
  annotated/
    backend_name/
      image_name.jpg
```

`summary.json` is for tools and UI. `report.md` is for quick human review. `detections.jsonl`
keeps per-image records append-friendly and easy to inspect with command-line tools.

## Record Schema

Each `detections.jsonl` line should be one backend run for one image:

```json
{
  "image": {
    "path": "/abs/path/image.jpg",
    "name": "image.jpg",
    "size_hw": [1080, 1920]
  },
  "backend": "yolo_world",
  "model": "yolov8s-world.pt",
  "device": "cpu",
  "prompt_classes": ["car", "bus", "person"],
  "status": "ok",
  "timing_ms": {
    "runs": 1,
    "mean": 277.4,
    "median": 277.4,
    "min": 277.4,
    "max": 277.4
  },
  "num_raw_detections": 5,
  "num_output_detections": 5,
  "labels": ["bus", "person"],
  "detections": []
}
```

Failure records should have the same image/backend identity plus:

```json
{
  "status": "missing_dependency",
  "error_type": "ImportError",
  "error": "..."
}
```

Use the same status vocabulary as smoke checks where possible:

- `ok`
- `missing_dependency`
- `load_error`
- `image_error`
- `inference_error`
- `error`

## Summary Schema

`summary.json` should include:

- command arguments
- timestamp
- git commit, if available
- Python version and platform
- input image count
- backend count
- per-backend status counts
- per-backend timing aggregates
- per-backend detection count aggregates
- per-backend label histogram
- output paths
- requested and resolved devices

Example:

```json
{
  "images": 12,
  "backends": ["ultralytics", "yolo_world"],
  "status_counts": {
    "ultralytics": {"ok": 12},
    "yolo_world": {"ok": 12}
  },
  "labels": {
    "yolo_world": {"person": 20, "bus": 3}
  }
}
```

## Markdown Report

`report.md` should be concise and readable in GitHub:

- command
- environment summary
- backend table
- failures table
- label histogram table
- slowest image/backend runs
- annotated output location

Avoid embedding images in the first version. File links are enough.

## Backend Loading Strategy

Model loading is expensive. The benchmark should load each backend once per backend configuration
and reuse it across all images.

Pseudo-flow:

```text
parse image list
parse backend specs
for backend_spec in backend_specs:
  create backend once
  create runtime once
  for image in images:
    run warmup/repeat inference
    write record
```

For prompt sweeps, each prompt set is a distinct backend spec because YOLO-World class prompts are
part of backend state:

```text
yolo_world/default-prompts
yolo_world/traffic-workzone-prompts
yolo_world/vru-prompts
```

## Prompt Handling

Prompt classes should be represented explicitly in output records. For named prompt sets:

```bash
--prompt-set road=car,bus,truck,person
--prompt-set workzone=traffic cone,barrier,construction vehicle
```

The generated backend display names should be stable:

```text
yolo_world__road
yolo_world__workzone
```

The unnamed `--prompt-classes` option should map to:

```text
yolo_world
```

## Annotated Images

Use the existing `draw_detections` first. The first version should not try to create multi-backend
comparison grids. Keep one annotated image per backend per input:

```text
annotated/yolo_world__road/frame001.jpg
annotated/dfine/frame001.jpg
```

Later versions can add:

- side-by-side contact sheets
- overlay comparison
- false-positive review labels
- mask rendering for segmentation

## Error Handling

One failed backend should not prevent other backends from running unless `--fail-fast` is set.

Expected handling:

- Backend load failure: write one failure record per backend, skip that backend for all images.
- Image read failure: write one failure record per image/backend or one image-level failure in
  `failures.jsonl`.
- Inference failure: write one failure record and continue with the next image.
- If `--fail-fast` is set, exit non-zero on the first failure.

Exit codes:

- `0`: all selected runs completed, even if there were zero detections.
- `1`: one or more selected runs failed.
- `2`: invalid CLI arguments.

## Testing Plan

Unit tests should avoid real model downloads.

Initial tests:

- image discovery: file, directory, recursive, extension filter
- backend spec parsing: aliases, model overrides, prompt sets
- summary aggregation
- JSONL writing
- failure record writing
- fake backend end-to-end benchmark over generated images

Optional local smoke tests:

- Ultralytics CPU on synthetic image.
- YOLO-World CPU on Ultralytics `bus.jpg` with `bus,person` prompts.
- D-FINE and RF-DETR CPU where dependencies are installed.

## Implementation Steps

1. Add `tools_benchmark_images.py` with fake backend support only.
2. Add output writers for `summary.json`, `detections.jsonl`, `failures.jsonl`, and `report.md`.
3. Add image discovery and backend spec parsing tests.
4. Reuse real backends through the existing backend loader.
5. Add prompt set support for YOLO-World.
6. Add annotated image output.
7. Add README examples.
8. Consider a small read-only web report viewer once output schemas stabilize.

## Web UI Follow-Up

The web UI should read benchmark outputs first, not run models first. This keeps the initial UI
simple and makes benchmark runs reproducible.

Useful first UI:

- open an output directory
- browse images
- switch backend/prompt set
- inspect raw JSON for a selected detection
- compare latency and label histograms

After that, add a local service mode that runs benchmarks from the UI.

## Segmentation Follow-Up

Segmentation should extend the result schema rather than overloading detections:

- `detections` remains bbox/object detections.
- `masks` can be added for instance segmentation.
- `semantic_map` can be added for semantic segmentation.
- `annotated` output can render masks with alpha blending.

The benchmark record should allow these fields to coexist so mixed detector and segmentation
backends can be compared in one report.
