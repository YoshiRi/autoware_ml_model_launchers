import json

from autoware_ml_model_launchers.open_detector.image_io import read_image_bgr, write_image_bgr
from autoware_ml_model_launchers.open_detector.make_test_image import make_image
from autoware_ml_model_launchers.open_detector.tools_benchmark_images import (
    build_backend_specs,
    discover_images,
    main as benchmark_main,
    parse_model_overrides,
    parse_prompt_sets,
    resolve_device,
)


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_discover_images_filters_extensions_and_recursion(tmp_path):
    image_dir = tmp_path / "images"
    nested_dir = image_dir / "nested"
    write_image_bgr(image_dir / "a.jpg", make_image(width=64, height=48))
    write_image_bgr(image_dir / "b.png", make_image(width=64, height=48))
    write_image_bgr(nested_dir / "c.jpg", make_image(width=64, height=48))
    (image_dir / "notes.txt").write_text("ignore me", encoding="utf-8")

    non_recursive = discover_images(image_dir, extensions=["jpg"])
    recursive = discover_images(image_dir, recursive=True, extensions=["jpg"])

    assert [path.name for path in non_recursive] == ["a.jpg"]
    assert [path.name for path in recursive] == ["a.jpg", "c.jpg"]


def test_parse_model_overrides_canonicalizes_backend_aliases():
    assert parse_model_overrides(["yolo=yolo11n.pt", "yolo-world=yolov8s-world.pt"]) == {
        "ultralytics": "yolo11n.pt",
        "yolo_world": "yolov8s-world.pt",
    }


def test_yolo_world_prompt_sets_expand_backend_specs():
    specs = build_backend_specs(
        "fake,yolo-world",
        model_overrides={"yolo_world": "yolov8s-world.pt"},
        prompt_classes=["fallback"],
        prompt_sets=parse_prompt_sets(["traffic=car,person", "site=cone,barrier"]),
    )

    assert [spec.display_name for spec in specs] == [
        "fake",
        "yolo_world__traffic",
        "yolo_world__site",
    ]
    assert specs[1].prompt_classes == ("car", "person")
    assert specs[2].model == "yolov8s-world.pt"


def test_auto_device_keeps_fake_backend_dependency_free():
    assert resolve_device("fake", "auto") == ""
    assert resolve_device("ultralytics", "cpu") == "cpu"


def test_benchmark_images_fake_backend_writes_report_and_artifacts(tmp_path):
    image_dir = tmp_path / "images"
    output_dir = tmp_path / "benchmark"
    write_image_bgr(image_dir / "a.jpg", make_image(width=320, height=180))
    write_image_bgr(image_dir / "b.png", make_image(width=400, height=240))

    exit_code = benchmark_main(
        [
            "--input",
            str(image_dir),
            "--output-dir",
            str(output_dir),
            "--backends",
            "fake",
            "--default-driving-label-map",
            "--repeat",
            "2",
        ]
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    detections = _read_jsonl(output_dir / "detections.jsonl")

    assert exit_code == 0
    assert summary["input_images"] == 2
    assert summary["backend_runs"] == 2
    assert summary["requested_device"] == "auto"
    assert summary["resolved_devices"] == {"fake": ""}
    assert summary["backends"]["fake"]["status_counts"] == {"ok": 2}
    assert summary["backends"]["fake"]["output_detection_count"]["sum"] == 4
    assert summary["backends"]["fake"]["labels"] == {"CAR": 2, "PEDESTRIAN": 2}
    assert [record["num_output_detections"] for record in detections] == [2, 2]
    assert (output_dir / "failures.jsonl").read_text(encoding="utf-8") == ""
    assert (output_dir / "report.md").exists()
    assert read_image_bgr(output_dir / "annotated" / "fake" / "a.jpg").shape[:2] == (180, 320)
    assert read_image_bgr(output_dir / "annotated" / "fake" / "b.png").shape[:2] == (240, 400)


def test_benchmark_images_returns_cli_error_for_missing_input(tmp_path):
    exit_code = benchmark_main(
        [
            "--input",
            str(tmp_path / "missing"),
            "--output-dir",
            str(tmp_path / "benchmark"),
        ]
    )

    assert exit_code == 2
