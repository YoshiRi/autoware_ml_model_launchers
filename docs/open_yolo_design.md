# Open YOLO integration design

## Status

The comparison node supports Ultralytics YOLO, Ultralytics YOLO-World, Hugging Face D-FINE, and
Roboflow RF-DETR through a backend-neutral ROS interface. Model weights and Python
machine-learning runtimes are intentionally not stored in this repository.

## Goals

- Provide several generic 2D detectors for comparison with Autoware YOLOX.
- Keep cloning and building the ROS package straightforward.
- Use standard ROS messages rather than Autoware-specific detection messages.
- Prevent model weights and large Python/CUDA packages from growing the Git repository.
- Make the implementation easy to extract when the collection of experiments becomes larger.

## Current structure

- `autoware_ml_model_launchers/open_detector/`: shared types, ROS I/O, and detector backends
- `autoware_ml_model_launchers/compressed_yolo_node.py`: compatibility entry point
- `launch/open_detector.launch.xml`: backend-neutral launcher
- `launch/open_yolo.launch.xml`: backward-compatible Ultralytics launcher
- `launch/open_dfine.launch.xml`: D-FINE launcher
- `launch/open_rfdetr.launch.xml`: RF-DETR launcher
- `requirements-open-*.txt`: backend-specific pip dependencies
- `/opt/autoware/mlmodels` or another external path: production-managed model storage

The node publishes `vision_msgs/msg/Detection2DArray`. Its output is not a drop-in replacement for
Autoware YOLOX because the message schema and class taxonomy differ.

The Ultralytics launcher's default inference size is 960 to match the existing Autoware YOLOX
comparison configuration. YOLO-World adds open-vocabulary class prompts while still returning
bounding boxes. D-FINE and RF-DETR own their preprocessing. The convenience launchers default to
smaller model variants that are practical on a 6 GB GPU. Explicit local model paths are preferred
for repeatable and offline runs.

## Dependency policy

ROS, OpenCV, and NumPy dependencies are declared in `package.xml`. Heavy backend dependencies are
split into separate requirements files and imported only by the selected backend:

- Ultralytics and YOLO-World: `requirements-open-yolo.txt`
- D-FINE: `requirements-open-detector-dfine.txt`
- RF-DETR: `requirements-open-detector-rfdetr.txt`

Virtual environments should use `--system-site-packages` so they can import ROS Python packages.
Separate environments per backend are preferred if PyTorch, Transformers, or RF-DETR dependency
constraints conflict.

`create_backend()` is the detector construction boundary. It imports only the selected backend,
loads that backend's model, and returns a detector ready for inference. The ROS node therefore fails
during initialization when dependencies, model files, downloads, or GPU setup are invalid rather
than waiting for the first image.

Do not commit any of the following:

- model weights such as `.pt`, `.onnx`, or TensorRT engine files
- virtual environments
- pip download caches
- generated datasets, bags, or debug images

The source code in this repository remains Apache-2.0. Backend libraries and model weights retain
their own licenses. In particular, Ultralytics 8.4.65 reports AGPL-3.0 in its package metadata.
D-FINE model cards and RF-DETR documentation currently identify Apache-2.0, but the exact selected
model and installed package must still be reviewed before redistribution or product use.

## When to split the implementation

Keep the adapters in this repository while the shared source remains small, optional dependencies
remain backend-specific, and CI can run without installing real model runtimes. Extract them when one or
more of these conditions apply:

- multiple independent detectors, datasets, or evaluation frameworks are added
- its release cadence differs from the Autoware launcher package
- it requires its own CI matrix, container image, or maintainers
- optional dependencies make installation or testing of this repository materially harder
- the detector becomes useful outside the Autoware workspace

## Extraction mechanism

When extraction is required, create a standalone repository and reference it through a
`*.repos` file for `vcs import`. Prefer this over a Git submodule because Autoware workspaces
already use repository manifests and because submodules introduce a separate initialization and
revision-update workflow.

The extracted repository should preserve the ROS executable and parameter interface. This package
can then keep only `open_yolo.launch.xml`, the environment check, and the manifest entry.
