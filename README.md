# autoware_ml_model_launchers

Launchers and small helper nodes for validating Autoware ML models with recorded camera topics.

## Development

Development uses `dev` as the integration branch. Create `feature/*`, `fix/*`, `refactor/*`, or
`docs/*` branches from the latest `dev` and open pull requests back into `dev`. Direct pushes to
`main` are not part of the development workflow. See [`AGENTS.md`](AGENTS.md) for the complete
branch policy and local validation commands.

## Setup

Clone this repository into the `src` directory of an existing Autoware workspace:

```bash
cd "${AUTOWARE_WS}/src"
git clone git@github.com:YoshiRi/autoware_ml_model_launchers.git \
  tools/autoware_ml_model_launchers
```

Build the package from the workspace root with a symlink install, then source the workspace:

```bash
cd "${AUTOWARE_WS}"
colcon build --symlink-install --packages-select autoware_ml_model_launchers
source install/setup.bash
```

Set `AUTOWARE_WS` to the root of your Autoware workspace. The workspace dependencies must already
be installed and built.

## Quick start

This command runs generic YOLOX detection, ByteTrack, traffic light detection, and traffic light
classification for `camera5`:

```bash
ros2 launch autoware_ml_model_launchers all_single_camera_detection.launch.xml namespace:=camera5
```

The launcher expects compressed images on:

```text
/sensing/camera/camera5/image_raw/compressed
```

Change `camera5` to match the camera namespace in the live system or rosbag. `namespace` is kept as
a short compatibility argument; `camera_namespace:=camera5` is the explicit equivalent.

> [!IMPORTANT]
> This is a model-validation pipeline, not the standard Autoware perception pipeline. In
> particular, traffic light ROIs are generated directly from YOLOX detections by
> `tlr_yolox_roi_adapter`; they are not associated with map traffic light IDs. Do not use its
> output as a drop-in replacement for the normal Autoware traffic light recognition pipeline.

## Prerequisites

- ROS 2 and the Autoware workspace are sourced.
- An NVIDIA GPU is available.
- The camera source or rosbag publishes the compressed image topic shown above.
- The required ML models are installed under `/opt/autoware/mlmodels`.

Internal users should provision the runtime and ML models with **Inteverse** before using this
package.

After building and sourcing this package, run the environment check:

```bash
ros2 run autoware_ml_model_launchers check_environment --camera camera5
```

It checks the GPU, required ROS packages, model files, and input topic. A missing input topic is
reported as a warning because the check may be run before rosbag playback starts. To use another
pipeline or model root:

```bash
ros2 run autoware_ml_model_launchers check_environment \
  --pipeline ptv3 \
  --data-path /path/to/mlmodels
```

Available pipelines are `camera`, `open_detector`, `open_yolo`, `bevfusion`, `ptv3`,
`centerpoint`, and `streampetr`.

The default model tree must contain:

```text
/opt/autoware/mlmodels/
|-- yolox/
|   |-- yolox-sPlus-T4-960x960-debris.onnx
|   |-- label.txt
|   `-- semseg_color_map.csv
|-- traffic_light_detector/
|   |-- yolox_s_car_ped_tl_detector_960_960_batch_1.onnx
|   `-- car_ped_tl_detector_labels.txt
`-- traffic_light_classifier/
    |-- traffic_light_classifier_mobilenetv2_batch_6.onnx
    |-- ped_traffic_light_classifier_mobilenetv2_batch_6.onnx
    |-- lamp_labels.txt
    |-- lamp_labels_ped.txt
    `-- lamp_recognizer_ml.param.yaml
```

## Main launcher

`all_single_camera_detection.launch.xml` starts all image-processing examples against one camera
stream.

Useful switches:

- `use_decompress:=true`: decompress the input image once for all ML nodes
- `enable_yolox:=true`: run generic YOLOX object detection
- `enable_bytetrack:=true`: run ByteTrack from the generic YOLOX ROIs
- `enable_tlr:=true`: run traffic light detection
- `enable_tlr_classification:=true`: run traffic light classification and category merge
- `data_path:=/opt/autoware/mlmodels`: set the model root directory
- `build_only:=true`: build TensorRT engines and exit the TensorRT-backed nodes

Example without ByteTrack:

```bash
ros2 launch autoware_ml_model_launchers all_single_camera_detection.launch.xml \
  camera_namespace:=camera5 \
  enable_bytetrack:=false
```

Default outputs:

- YOLOX ROIs: `/perception/object_recognition/detection/camera5/rois`
- ByteTrack ROIs: `/perception/object_recognition/detection/camera5/tracked/rois`
- TLR YOLOX objects: `/perception/traffic_light_recognition/camera5/detection/yolox/objects`
- TLR ROIs: `/perception/traffic_light_recognition/camera5/detection/rois`
- TLR signals: `/perception/traffic_light_recognition/camera5/classification/traffic_signals`
- TLR debug image: `/perception/traffic_light_recognition/camera5/detection/yolox/debug/image`

## Visualization with Lichtblick

Visualization currently assumes **Lichtblick**. Connect Lichtblick to the ROS graph and add image
and raw-message panels with these settings:

| Panel | Topic |
| --- | --- |
| Image (camera input) | `/sensing/camera/camera5/image_raw` |
| Image (TLR debug) | `/perception/traffic_light_recognition/camera5/detection/yolox/debug/image` |
| Image (Open YOLO debug) | `/perception/object_recognition/detection/open_yolo/camera5/debug/image` |
| Raw messages (YOLOX ROIs) | `/perception/object_recognition/detection/camera5/rois` |
| Raw messages (Open YOLO) | `/perception/object_recognition/detection/open_yolo/camera5/detections` |
| Raw messages (tracked ROIs) | `/perception/object_recognition/detection/camera5/tracked/rois` |
| Raw messages (traffic signals) | `/perception/traffic_light_recognition/camera5/classification/traffic_signals` |

When viewing the compressed source directly, select
`/sensing/camera/camera5/image_raw/compressed`. Replace `camera5` in every topic when another
camera namespace is used. Save the configured workspace as a Lichtblick layout for reuse.

## Individual launchers

Run generic YOLOX and ByteTrack only:

```bash
ros2 launch autoware_ml_model_launchers yolox_camera.launch.xml camera_namespace:=camera5
```

Run traffic light detection and classification only:

```bash
ros2 launch autoware_ml_model_launchers tlr_detect_and_classifier.launch.xml \
  camera_namespace:=camera5
```

`node_namespace` defaults to `perception/<camera_namespace>`. `camera_namespace` controls the
`/sensing/camera/<camera_namespace>` input path and per-camera output topic names.

## Open detector comparison

The included backend-switchable detector compares Autoware YOLOX with several open 2D detection
models while keeping one ROS interface:

| Launcher | Backend | Default model |
| --- | --- | --- |
| `open_yolo.launch.xml` | Ultralytics | `yolo26s.pt`, input size 960 |
| `open_dfine.launch.xml` | Hugging Face D-FINE | `ustc-community/dfine-small-obj2coco` |
| `open_rfdetr.launch.xml` | Roboflow RF-DETR | `small` |
| `open_detector.launch.xml` | Select with `backend:=...` | Backend default |

All backends consume the camera stream and publish `vision_msgs/msg/Detection2DArray`. Their class
taxonomies and preprocessing differ from Autoware YOLOX, so outputs are for comparison rather than
drop-in replacement.

The ROS adapters are included in this repository. Heavy Python runtimes and model weights remain
external and are loaded only for the selected backend. Review
[`docs/open_yolo_design.md`](docs/open_yolo_design.md) before redistribution or product use.

Install common ROS dependencies:

```bash
sudo apt install ros-${ROS_DISTRO}-vision-msgs python3-opencv python3-venv
```

Create an isolated environment and install only the backend being tested:

```bash
python3 -m venv ~/venvs/open_detector --system-site-packages
source ~/venvs/open_detector/bin/activate
python3 -m pip install --upgrade pip

# Ultralytics
python3 -m pip install -r \
  "${AUTOWARE_WS}/src/tools/autoware_ml_model_launchers/requirements-open-yolo.txt"

# D-FINE
python3 -m pip install -r \
  "${AUTOWARE_WS}/src/tools/autoware_ml_model_launchers/requirements-open-detector-dfine.txt"

# RF-DETR
python3 -m pip install -r \
  "${AUTOWARE_WS}/src/tools/autoware_ml_model_launchers/requirements-open-detector-rfdetr.txt"
```

Separate virtual environments per backend are recommended when dependency versions conflict.
For the CUDA 13.0 GPU setup tested with D-FINE, including the required PyTorch nightly index and
Pillow fix, see
[`docs/open_dfine_troubleshooting.md`](docs/open_dfine_troubleshooting.md).

Build from the Autoware workspace root:

```bash
cd "${AUTOWARE_WS}"
source /opt/ros/${ROS_DISTRO}/setup.bash
colcon build --symlink-install --packages-select autoware_ml_model_launchers
source install/setup.bash
```

Activate the selected virtual environment before checking or launching:

```bash
source ~/venvs/open_detector/bin/activate

ros2 run autoware_ml_model_launchers check_environment \
  --pipeline open_detector \
  --backend dfine \
  --camera camera5
```

Launch each backend:

```bash
ros2 launch autoware_ml_model_launchers open_yolo.launch.xml camera_namespace:=camera5
ros2 launch autoware_ml_model_launchers open_dfine.launch.xml camera_namespace:=camera5
ros2 launch autoware_ml_model_launchers open_rfdetr.launch.xml camera_namespace:=camera5
```

The first launch may download model weights. For offline or repeatable runs, pre-download the
weight and pass a local path where the backend supports it:

```bash
ros2 launch autoware_ml_model_launchers open_yolo.launch.xml \
  camera_namespace:=camera5 \
  model:=/path/to/yolo26s.pt

ros2 launch autoware_ml_model_launchers open_dfine.launch.xml \
  camera_namespace:=camera5 \
  model:=/path/to/local/huggingface/model
```

Use the generic launcher for aliases or less common variants:

```bash
ros2 launch autoware_ml_model_launchers open_detector.launch.xml \
  camera_namespace:=camera5 \
  detector_name:=open_dfine_large \
  backend:=dfine \
  model:=ustc-community/dfine-xlarge-obj2coco \
  device:=cuda:0
```

The Ultralytics backend defaults to `imgsz:=960` to match the Autoware YOLOX input size. D-FINE
and RF-DETR own their input preprocessing and ignore this setting.

Default outputs use a backend-specific detector name:

- Detections: `/perception/object_recognition/detection/<detector>/camera5/detections`
- Debug image: `/perception/object_recognition/detection/<detector>/camera5/debug/image`
- Inference time: `/perception/object_recognition/detection/<detector>/camera5/latency/infer_ms`

For backward compatibility, `open_yolo.launch.xml` retains
`/perception/object_recognition/detection/open_yolo/camera5/latency/predict_ms`.

Run dependency-free plumbing and adapter checks:

```bash
ros2 run autoware_ml_model_launchers open_detector_smoke \
  --backends fake \
  --fail-on-error
```

`infer_ms` measures each backend adapter's complete inference call, which can include preprocessing
and postprocessing. It must not be compared directly with an inference-only TensorRT metric.

## LiDAR model launchers

The following launchers use `/sensing/lidar/concatenated/pointcloud` by default. Override
`input/pointcloud` when the rosbag uses another topic.

### BEVFusion LiDAR

```bash
ros2 run autoware_ml_model_launchers check_environment --pipeline bevfusion
ros2 launch autoware_ml_model_launchers bevfusion_lidar.launch.xml
```

Output:

```text
/perception/object_recognition/detection/bevfusion/objects
```

The default model directory is `/opt/autoware/mlmodels/bevfusion_lidar`. To build the TensorRT
engine and exit:

```bash
ros2 launch autoware_ml_model_launchers bevfusion_lidar.launch.xml build_only:=true
```

### PTv3

```bash
ros2 run autoware_ml_model_launchers check_environment --pipeline ptv3
ros2 launch autoware_ml_model_launchers ptv3.launch.xml
```

Outputs:

- Segmentation: `/perception/segmentation/ptv3/segmentation`
- Colored visualization cloud: `/perception/segmentation/ptv3/visualization`
- Filtered cloud: `/perception/segmentation/ptv3/filtered`

The default model directory is `/opt/autoware/mlmodels/ptv3`.

### CenterPoint

```bash
ros2 run autoware_ml_model_launchers check_environment --pipeline centerpoint
ros2 launch autoware_ml_model_launchers lidar_centerpoint.launch.xml
```

The default model is `centerpoint_tiny` under
`/opt/autoware/mlmodels/lidar_centerpoint`. The model was not installed in the environment used
to develop these launchers, so only launch-file expansion was tested. Run the environment check
before attempting inference.

## StreamPETR for X2

`streampetr_x2.launch.xml` captures the known X2 camera mapping. It is intentionally vehicle
specific because the input order must match the order used during training:

| Model input | Direction | X2 camera |
| --- | --- | --- |
| camera0 | front | camera8 |
| camera1 | front-left | camera6 |
| camera2 | back-left | camera10 |
| camera3 | front-right | camera7 |
| camera4 | back-right | camera9 |

Check the model and expected topics, then launch:

```bash
ros2 run autoware_ml_model_launchers check_environment --pipeline streampetr
ros2 launch autoware_ml_model_launchers streampetr_x2.launch.xml
```

The launcher defaults to compressed, distorted images and models under
`/opt/autoware/mlmodels/streampetr`. It also requires matching `CameraInfo`, camera-to-base TF,
and map-to-base TF. All five images must arrive within `max_camera_time_diff:=0.2`.

The first launch builds three TensorRT engines and may take several minutes. Build them before
starting rosbag playback:

```bash
ros2 launch autoware_ml_model_launchers streampetr_x2.launch.xml build_only:=true
```

TensorRT engine building may not respond immediately to `Ctrl-C`; allow it to finish when possible.

The camera namespace arguments can be changed, but changing only topic names does not guarantee
that another vehicle has the camera order or calibration expected by the model.

Output:

```text
/perception/object_recognition/detection/streampetr/objects
```

The X2 deployment may additionally pass this output through
`autoware_object_sorter` to `/perception/object_recognition/detection/camera_only/objects`.
That step is not included here because its parameter file belongs to the vehicle-specific launch
configuration.

## Model launcher status

| Launcher | Model files found | Launch expansion | Inference |
| --- | --- | --- | --- |
| BEVFusion LiDAR | Yes | Tested | Model and engine load tested; rosbag inference not tested |
| PTv3 | Yes | Tested | Model and engine load tested; rosbag inference not tested |
| CenterPoint | No | Tested | Not tested |
| StreamPETR X2 | Yes | Tested | Parameters and ONNX load tested; full engine build not completed |
| Open YOLO | Auto-download or explicit path | Tested | GPU inference and ROS topic round-trip tested |
| Open D-FINE | Hugging Face or local path | Tested | Adapter and fake ROS plumbing tested |
| Open RF-DETR | Auto-download or variant | Tested | Adapter and fake ROS plumbing tested |
