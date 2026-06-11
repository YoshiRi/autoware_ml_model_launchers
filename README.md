# autoware_ml_model_launchers

Launchers and small helper nodes for validating Autoware ML models with recorded camera topics.

## YOLOX object detection

```bash
ros2 launch autoware_ml_model_launchers yolox_camera.launch.xml \
  camera_namespace:=camera5
```

Default topics:

- input: `/sensing/camera/camera5/image_raw/compressed`
- YOLOX ROI output: `/perception/object_recognition/detection/camera5/rois`
- ByteTrack output: `/perception/object_recognition/detection/camera5/tracked/rois`

## Traffic light detection and classification

```bash
ros2 launch autoware_ml_model_launchers tlr_detect_and_classifier.launch.xml \
  camera_namespace:=camera5
```

Default topics:

- input: `/sensing/camera/camera5/image_raw/compressed`
- YOLOX objects: `/perception/traffic_light_recognition/camera5/detection/yolox/objects`
- classifier ROIs: `/perception/traffic_light_recognition/camera5/detection/rois`
- final traffic signals: `/perception/traffic_light_recognition/camera5/classification/traffic_signals`

`node_namespace` defaults to `perception/<camera_namespace>`, while `camera_namespace` controls
the `/sensing/camera/<camera_namespace>` input path and per-camera output topic names.
