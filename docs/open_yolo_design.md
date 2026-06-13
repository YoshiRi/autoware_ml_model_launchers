# Open YOLO integration design

## Status

The generic Ultralytics comparison node is maintained in
`autoware_ml_model_launchers` alongside its launch file. Model weights and the Python machine
learning runtime are intentionally not stored in this repository.

## Goals

- Provide a generic 2D detector for comparison with Autoware YOLOX.
- Keep cloning and building the ROS package straightforward.
- Use standard ROS messages rather than Autoware-specific detection messages.
- Prevent model weights and large Python/CUDA packages from growing the Git repository.
- Make the implementation easy to extract when the collection of experiments becomes larger.

## Current structure

- `autoware_ml_model_launchers/compressed_yolo_node.py`: compressed image inference node
- `launch/open_yolo.launch.xml`: camera topics and model runtime parameters
- `requirements-open-yolo.txt`: optional pip dependencies
- `/opt/autoware/mlmodels` or another external path: production-managed model storage

The node publishes `vision_msgs/msg/Detection2DArray`. Its output is not a drop-in replacement for
Autoware YOLOX because the message schema and class taxonomy differ.

The launcher's default inference size is 960 to match the existing Autoware YOLOX comparison
configuration. The default `yolo26s.pt` weight can be downloaded by Ultralytics, but an explicit
local path is preferred for repeatable and offline runs.

## Dependency policy

ROS dependencies are declared in `package.xml`. Ultralytics, OpenCV, NumPy, PyTorch, and their CUDA
runtime are optional Python dependencies installed in a virtual environment. The virtual
environment should use `--system-site-packages` so it can import ROS Python packages.

Do not commit any of the following:

- model weights such as `.pt`, `.onnx`, or TensorRT engine files
- virtual environments
- pip download caches
- generated datasets, bags, or debug images

Ultralytics 8.4.65 reports an AGPL-3.0 license in its package metadata. The source code in this
repository remains Apache-2.0, but redistribution and product use of the combined runtime must be
reviewed against the applicable Ultralytics and model licenses.

## When to split the implementation

Keep the node in this repository while it remains a small comparison utility sharing the same
camera validation workflow. Extract it when one or more of these conditions apply:

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
