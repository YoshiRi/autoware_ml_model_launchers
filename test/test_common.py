import json

import pytest

from autoware_ml_model_launchers.open_detector.backend_loader import canonical_backend_name, create_backend
from autoware_ml_model_launchers.open_detector.cli_detect_image import main as detect_image_main
from autoware_ml_model_launchers.open_detector.filtering import (
    apply_filter_and_mapping,
    parse_class_filter,
    parse_label_map,
)
from autoware_ml_model_launchers.open_detector.image_io import read_image_bgr, write_image_bgr
from autoware_ml_model_launchers.open_detector.make_test_image import make_image
from autoware_ml_model_launchers.open_detector.smoke_backends import main as smoke_backends_main
from autoware_ml_model_launchers.open_detector.types import BackendConfig, Detection


def test_parse_class_filter():
    assert parse_class_filter("person, car") == {"person", "car"}
    assert parse_class_filter('["traffic light", "truck"]') == {"traffic light", "truck"}


def test_parse_label_map():
    assert parse_label_map("person=PEDESTRIAN,car=CAR") == {"person": "PEDESTRIAN", "car": "CAR"}
    assert parse_label_map('{"traffic light":"TRAFFIC_LIGHT"}') == {"traffic light": "TRAFFIC_LIGHT"}


def test_filter_and_mapping():
    detections = [
        Detection(0, 0, 10, 20, 0.9, "person", 0),
        Detection(0, 0, 10, 20, 0.8, "dog", 16),
    ]
    out = apply_filter_and_mapping(
        detections,
        class_filter={"person"},
        label_map={"person": "PEDESTRIAN"},
    )
    assert len(out) == 1
    assert out[0].label == "PEDESTRIAN"


def test_backend_alias_and_fake():
    assert canonical_backend_name("yolo") == "ultralytics"
    assert canonical_backend_name("yolo-world") == "yolo_world"
    backend = create_backend(BackendConfig(backend="fake"))
    assert backend.loaded is True


@pytest.mark.parametrize(
    "name",
    ["ultralytics", "yolo", "yolo_world", "yolo-world", "dfine", "d-fine", "rfdetr", "rf-detr", "fake"],
)
def test_backend_factory_can_skip_loading_for_adapter_tests(name):
    backend = create_backend(BackendConfig(backend=name), load=False)
    assert backend.loaded is False


def test_unloaded_backend_rejects_inference():
    backend = create_backend(BackendConfig(backend="fake"), load=False)
    with pytest.raises(RuntimeError, match="model is not loaded"):
        backend.infer(make_image())


def test_fake_cli_outputs_json_and_debug_image(tmp_path):
    input_path = tmp_path / "input.jpg"
    output_path = tmp_path / "annotated.jpg"
    json_path = tmp_path / "detections.json"
    write_image_bgr(input_path, make_image(width=320, height=180))

    detect_image_main(
        [
            "--backend",
            "fake",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--json",
            str(json_path),
            "--default-driving-label-map",
        ]
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["backend"] == "fake"
    assert payload["image_size_hw"] == [180, 320]
    assert payload["num_output_detections"] == 2
    assert {det["label"] for det in payload["detections"]} == {"CAR", "PEDESTRIAN"}
    assert read_image_bgr(output_path).shape[:2] == (180, 320)


def test_smoke_backends_fake_report(tmp_path):
    report_path = tmp_path / "smoke.json"
    output_dir = tmp_path / "annotated"

    exit_code = smoke_backends_main(
        [
            "--backends",
            "fake",
            "--json",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--default-driving-label-map",
            "--fail-on-error",
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["summary"] == {"ok": 1}
    assert payload["backends"][0]["backend"] == "fake"
    assert payload["backends"][0]["num_output_detections"] == 2
    assert (output_dir / "fake.jpg").exists()


def test_ros_detection_message_conversion_when_ros_available():
    ros_node = pytest.importorskip("autoware_ml_model_launchers.open_detector.ros_node")
    std_msgs = pytest.importorskip("std_msgs.msg")

    header = std_msgs.Header()
    header.frame_id = "camera"
    msg = ros_node.OpenDetectorNode._to_detection_array(
        [Detection(10, 20, 30, 60, 0.75, "CAR", 2)],
        header,
    )

    assert msg.header.frame_id == "camera"
    assert len(msg.detections) == 1
    assert msg.detections[0].bbox.center.position.x == 20
    assert msg.detections[0].bbox.center.position.y == 40
    assert msg.detections[0].bbox.size_x == 20
    assert msg.detections[0].bbox.size_y == 40
    assert msg.detections[0].results[0].hypothesis.class_id == "CAR"
