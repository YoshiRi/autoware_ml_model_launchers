# Open Detector Architecture Plan

This package keeps open model detector work separate from ROS and Autoware-specific wiring.
The near-term goal is to make model backends easy to add and test on macOS, while preserving the
existing ROS launch surface for Autoware validation.

## Boundaries

`open_detector` is split into three conceptual layers:

- Core runtime: backend loading, inference, filtering, label mapping, and plain Python result types.
- ROS adapter: ROS image decoding, `vision_msgs` conversion, debug image publishing, and node
  lifecycle.
- Autoware integration: launch defaults, topic conventions, and any Autoware-specific message
  adapters.

The core runtime must not import ROS, Autoware packages, or launch APIs. It accepts an OpenCV BGR
image as `numpy.ndarray` and returns backend-neutral detections. This is the layer used by macOS
smoke checks, CLI tools, and future web UI services.

The ROS adapter can import ROS packages, but should avoid Autoware-specific assumptions except for
parameters passed in from launch files. Autoware-specific refactors should stay in launch files or
future Autoware adapter modules.

## Current Core API

`OpenDetectorRuntime` owns the common per-frame path:

```text
image_bgr -> backend.infer -> clip -> filter -> label map -> DetectorRuntimeResult
```

Backend constructors load dependencies and model weights before inference. This keeps dependency and
model failures at startup rather than on the first frame.

## Backend Roadmap

Add backends behind the existing backend interface before adding ROS launch surface:

1. Add a backend module under `open_detector/backends/`.
2. Keep package imports local to backend `load()` methods.
3. Return only backend-neutral `Detection` or a future task-specific result type.
4. Add a dependency-free adapter test with mocked model output.
5. Add a CLI smoke command and document runtime dependencies.
6. Add ROS launch files only after the pure Python path is stable.

Likely backend groups:

- 2D object detection: Ultralytics, D-FINE, RF-DETR, YOLO-World.
- Open-vocabulary detection: YOLO-World-style text prompt inputs.
- Segmentation: instance masks and semantic masks, likely requiring result types beyond `Detection`.
- Multitask perception: shared image preprocessing with multiple output heads.

## Segmentation Direction

Segmentation should not be forced into bounding boxes. Add explicit result types before adding
segmentation backends:

- `MaskDetection`: bbox plus binary or polygon mask.
- `SemanticSegmentation`: label map, palette metadata, and optional confidence map.
- `OpenDetectorRuntime` can either become a generic `OpenVisionRuntime`, or segmentation can get a
  parallel runtime with the same dependency boundaries.

The ROS adapter should convert these results to standard ROS messages where possible. Autoware-
specific conversions should remain outside the core runtime.

## Web UI Direction

A future web UI should call the core runtime through a small service layer, not through ROS. The
initial useful UI surface is:

- Upload or select an image.
- Choose backend and model.
- Set device, confidence threshold, class filter, and open-vocabulary prompts.
- Render detections, masks, timing, labels, and raw JSON.
- Export a reproducible CLI command.

This keeps the UI usable on macOS and avoids requiring a running Autoware workspace for model
exploration.

## Submodule Direction

If open model code grows beyond launch validation, split it into a separate Python package or Git
submodule with this shape:

```text
open_model_runtime/
  open_detector/core
  open_detector/backends
  open_detector/cli

autoware_ml_model_launchers/
  ROS nodes
  launch files
  Autoware defaults
```

Do this only after the backend interface and result types stabilize. Until then, keeping the code in
this repository makes refactoring cheaper.

## Near-Term Steps

1. Keep `OpenDetectorRuntime` as the only shared detector execution path.
2. Move ROS message conversion helpers out of `ros_node.py` when the next ROS-facing change is
   needed.
3. Add backend capability metadata for task type, default model, supported devices, and required
   dependency extras.
4. Add YOLO-World as an open-vocabulary detection backend using the pure Python CLI first.
5. Add segmentation result types before introducing segmentation launchers.
6. Prototype a local web UI against the core runtime after backend metadata exists.
