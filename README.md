# autoware_ml_model_launchers

Launchers and small helper nodes for validating Autoware ML models with recorded camera topics.

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
model root:

```bash
ros2 run autoware_ml_model_launchers check_environment \
  --camera camera5 \
  --data-path /path/to/mlmodels
```

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
| Raw messages (YOLOX ROIs) | `/perception/object_recognition/detection/camera5/rois` |
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
