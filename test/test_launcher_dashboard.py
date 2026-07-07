import subprocess

import pytest

from autoware_ml_model_launchers.launcher_dashboard import process_manager
from autoware_ml_model_launchers.launcher_dashboard import bag_player
from autoware_ml_model_launchers.launcher_dashboard.bag_player import (
    BagPlayerManager,
    _select_player_services,
    browse_bag_path,
    build_bag_play_command,
)
from autoware_ml_model_launchers.launcher_dashboard.process_manager import ProcessManager
from autoware_ml_model_launchers.launcher_dashboard.registry import (
    RegistryError,
    build_launch_command,
    default_registry_path,
    load_registry,
)


def test_default_registry_builds_yolox_command():
    registry = load_registry(default_registry_path())
    command = build_launch_command(
        registry.get("yolox_camera"),
        {"camera_namespace": "camera6", "use_decompress": False},
    )
    default_command = build_launch_command(
        registry.get("yolox_camera"),
        {"camera_namespace": "camera6"},
    )

    assert command[:4] == [
        "ros2",
        "launch",
        "autoware_ml_model_launchers",
        "yolox_camera.launch.xml",
    ]
    assert "camera_namespace:=camera6" in command
    assert "use_decompress:=false" in command
    assert "use_decompress:=true" in default_command
    assert "use_sim_time:=true" in default_command
    assert not any(item.startswith("node_namespace:=") for item in default_command)
    assert not any(item.startswith("input/image:=") for item in default_command)
    assert not any(item.startswith("output/objects:=") for item in default_command)


def test_multi_yolox_registry_exposes_cameras_and_groups():
    registry = load_registry(default_registry_path())
    payload = registry.to_json()["multi_yolox"]

    assert payload["cameras"][:5] == ["camera0", "camera1", "camera2", "camera3", "camera4"]
    assert payload["args"]["use_decompress"]["group"] == "basic"
    assert payload["args"]["use_decompress"]["default"] is True
    assert payload["args"]["use_sim_time"]["default"] is True
    assert payload["args"]["enable_bytetrack"]["group"] == "addon"


def test_tlr_validation_registry_builds_command():
    registry = load_registry(default_registry_path())
    payload = registry.to_json()["tlr_validation"]
    removed_arg = "detector_param" + "_suffix"
    command = build_launch_command(
        registry.get(payload["launcher_id"]),
        {
            "camera_namespace": "camera4",
            "use_decompress": True,
            "use_sim_time": True,
            "enable_classification": True,
            "detector_ml_package_name": "traffic_light_detector",
        },
    )

    assert payload["cameras"][:5] == ["camera0", "camera1", "camera2", "camera3", "camera4"]
    assert payload["args"]["enable_classification"]["group"] == "mode"
    assert payload["args"]["detector_ml_package_name"]["group"] == "model"
    assert removed_arg not in payload["args"]
    assert command[:4] == [
        "ros2",
        "launch",
        "autoware_ml_model_launchers",
        "tlr_detect_and_classifier.launch.xml",
    ]
    assert "camera_namespace:=camera4" in command
    assert "use_decompress:=true" in command
    assert "use_sim_time:=true" in command
    assert "enable_classification:=true" in command
    assert "detector_ml_package_name:=traffic_light_detector" in command
    assert not any(item.startswith(f"{removed_arg}:=") for item in command)
    assert not any(item.startswith("yolox_model_path:=") for item in command)
    assert not any(item.startswith("yolox_label_path:=") for item in command)


def test_model_comparison_registry_exposes_variants_and_isolation():
    registry = load_registry(default_registry_path())
    payload = registry.to_json()["model_comparison"]
    removed_arg = "detector_param" + "_suffix"
    yolox = registry.get("yolox_camera")
    tlr = registry.get("tlr_detect_and_classifier")

    assert payload["cameras"][:5] == ["camera0", "camera1", "camera2", "camera3", "camera4"]
    assert payload["variants"][0]["id"] == "yolox_960"
    assert payload["variants"][1]["args"]["model_name"] == "yolox-sPlus-T4-640x640-debris.onnx"
    assert payload["variants"][2]["args"]["detector_ml_package_name"] == "traffic_light_detector"
    assert all(removed_arg not in variant["args"] for variant in payload["variants"])
    assert payload["args"]["use_sim_time"]["default"] is True
    assert yolox.isolation is not None
    assert tlr.isolation is not None

    yolox_args = yolox.isolation.build_args(
        {
            "run_id": "run_a",
            "variant_id": "yolox_base",
            "camera_namespace": "camera5",
        }
    )
    tlr_args = tlr.isolation.build_args(
        {
            "run_id": "run_a",
            "variant_id": "tlr_detector",
            "camera_namespace": "camera5",
        }
    )

    assert yolox_args["node_namespace"] == "evaluation/run_a/yolox_base/camera5"
    assert (
        yolox_args["output/objects"]
        == "/evaluation/run_a/yolox_base/camera5/object_recognition/rois"
    )
    assert (
        tlr_args["output/rois"]
        == "/evaluation/run_a/tlr_detector/camera5/tlr/rois"
    )


def test_registry_rejects_unknown_launcher_arg():
    registry = load_registry(default_registry_path())

    with pytest.raises(RegistryError, match="unknown args"):
        build_launch_command(registry.get("yolox_camera"), {"not_an_arg": "value"})


def test_open_yolo_registry_uses_launch_defaults_for_wiring():
    registry = load_registry(default_registry_path())
    command = build_launch_command(
        registry.get("open_yolo"),
        {"camera_namespace": "camera6", "model": "yolo11n.pt"},
    )

    assert "camera_namespace:=camera6" in command
    assert "model:=yolo11n.pt" in command
    assert not any(item.startswith("detector_name:=") for item in command)
    assert not any(item.startswith("input/image:=") for item in command)
    assert not any(item.startswith("output/detections:=") for item in command)


def test_process_manager_start_uses_command_list(monkeypatch, tmp_path):
    registry = load_registry(default_registry_path())
    captured = {}

    class FakeProcess:
        pid = 12345
        returncode = None

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(process_manager.subprocess, "Popen", FakeProcess)

    manager = ProcessManager(registry, log_dir=tmp_path)
    process = manager.start(
        "yolox_camera",
        {"camera_namespace": "camera7", "enable_bytetrack_visualizer": False},
        label_suffix="camera7",
    )

    assert process["pid"] == 12345
    assert process["running"] is True
    assert captured["command"][0:3] == ["ros2", "launch", "autoware_ml_model_launchers"]
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["start_new_session"] is True
    assert "camera_namespace:=camera7" in captured["command"]
    assert "enable_bytetrack_visualizer:=false" in captured["command"]
    assert manager.tail_log(process["id"]).startswith("$ ros2 launch ")


def test_process_manager_close_removes_only_exited_processes(monkeypatch, tmp_path):
    registry = load_registry(default_registry_path())

    class FakeProcess:
        pid = 12345
        returncode = None

        def __init__(self, command, **kwargs):
            pass

        def poll(self):
            return self.returncode

    monkeypatch.setattr(process_manager.subprocess, "Popen", FakeProcess)

    manager = ProcessManager(registry, log_dir=tmp_path)
    first = manager.start("yolox_camera", {"camera_namespace": "camera5"})
    second = manager.start("yolox_camera", {"camera_namespace": "camera6"})

    with pytest.raises(ValueError, match="running processes"):
        manager.close(first["id"])

    manager._processes[first["id"]].process.returncode = 0
    closed = manager.close_all()

    assert [process["id"] for process in closed] == [first["id"]]
    assert [process["id"] for process in manager.list_processes()] == [second["id"]]


def test_process_manager_tail_log_caps_requested_lines(monkeypatch, tmp_path):
    registry = load_registry(default_registry_path())

    class FakeProcess:
        pid = 12345
        returncode = None

        def __init__(self, command, **kwargs):
            pass

        def poll(self):
            return self.returncode

    monkeypatch.setattr(process_manager.subprocess, "Popen", FakeProcess)

    manager = ProcessManager(registry, log_dir=tmp_path)
    process = manager.start("yolox_camera", {"camera_namespace": "camera5"})
    log_path = tmp_path / f"{process['id']}_yolox_camera.log"
    log_path.write_text("".join(f"line-{index}\n" for index in range(600)))

    tail = manager.tail_log(process["id"], lines=1000).splitlines()

    assert len(tail) == 500
    assert tail[0] == "line-100"
    assert tail[-1] == "line-599"


def test_process_manager_previews_isolated_comparison():
    registry = load_registry(default_registry_path())
    manager = ProcessManager(registry)

    preview = manager.preview_comparison(
        {
            "run_id": "eval run",
            "camera_namespace": "camera4",
            "common_args": {"use_sim_time": True, "use_decompress": True},
            "variants": [
                {
                    "id": "yolox/base",
                    "label": "YOLOX Base",
                    "launcher_id": "yolox_camera",
                    "args": {"model_name": "model_a.onnx", "enable_bytetrack": False},
                }
            ],
        }
    )

    process = preview["processes"][0]

    assert preview["run_id"] == "eval_run"
    assert preview["group_id"] == "comparison:eval_run"
    assert process["variant_id"] == "yolox_base"
    assert "model_name:=model_a.onnx" in process["command"]
    assert "camera_namespace:=camera4" in process["command"]
    assert "node_namespace:=evaluation/eval_run/yolox_base/camera4" in process["command"]
    assert (
        "output/objects:=/evaluation/eval_run/yolox_base/camera4/object_recognition/rois"
        in process["command"]
    )


def test_process_manager_starts_comparison_with_group_metadata(monkeypatch, tmp_path):
    registry = load_registry(default_registry_path())
    captured = []

    class FakeProcess:
        pid = 12345
        returncode = None

        def __init__(self, command, **kwargs):
            captured.append(command)

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(process_manager.subprocess, "Popen", FakeProcess)

    manager = ProcessManager(registry, log_dir=tmp_path)
    result = manager.start_comparison(
        {
            "run_id": "run_a",
            "camera_namespace": "camera5",
            "common_args": {"use_sim_time": True},
            "variants": [
                {
                    "id": "yolox_base",
                    "label": "YOLOX Base",
                    "launcher_id": "yolox_camera",
                    "args": {"enable_bytetrack": False},
                },
                {
                    "id": "tlr_detector",
                    "label": "TLR Detector",
                    "launcher_id": "tlr_detect_and_classifier",
                    "args": {"enable_classification": False},
                },
            ],
        }
    )

    assert len(result["processes"]) == 2
    assert len(captured) == 2
    assert {process["variant_id"] for process in result["processes"]} == {
        "yolox_base",
        "tlr_detector",
    }
    assert all(process["group_id"] == "comparison:run_a" for process in result["processes"])
    assert "output/rois:=/evaluation/run_a/tlr_detector/camera5/tlr/rois" in captured[1]
    assert "detector_ml_package_name:=traffic_light_detector" in captured[1]
    assert not any(item.startswith("yolox_model_path:=") for item in captured[1])

    stopped = manager.stop_group("comparison:run_a")

    assert len(stopped) == 2


def test_bag_play_command_keeps_args_as_list(tmp_path):
    bag_path = tmp_path / "sample_bag"
    command = build_bag_play_command(
        bag_path,
        rate=2.5,
        loop=True,
        clock=True,
        start_paused=True,
    )

    assert command[:4] == ["ros2", "bag", "play", str(bag_path)]
    assert "2.5" in command
    assert "--loop" in command
    assert "--clock" in command
    assert "--start-paused" in command
    assert command.index(str(bag_path)) < command.index("--clock")


def test_bag_player_manager_start(monkeypatch, tmp_path):
    captured = {}
    bag_path = tmp_path / "sample_bag"
    bag_path.mkdir()

    class FakeProcess:
        pid = 23456
        returncode = None

        def __init__(self, command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return self.returncode

    monkeypatch.setattr(bag_player.subprocess, "Popen", FakeProcess)

    manager = BagPlayerManager(log_dir=tmp_path)
    player = manager.start(str(bag_path), rate="1.5", clock=True)

    assert player["pid"] == 23456
    assert player["running"] is True
    assert captured["command"][:3] == ["ros2", "bag", "play"]
    assert "--disable-keyboard-controls" in captured["command"]
    assert "--clock" in captured["command"]
    assert captured["kwargs"]["stderr"] is subprocess.STDOUT
    assert captured["kwargs"]["start_new_session"] is True
    assert manager.tail_log().startswith("$ ros2 bag play ")


def test_bag_player_tail_log_caps_requested_lines(monkeypatch, tmp_path):
    bag_path = tmp_path / "sample_bag"
    bag_path.mkdir()

    class FakeProcess:
        pid = 23456
        returncode = None

        def __init__(self, command, **kwargs):
            pass

        def poll(self):
            return self.returncode

    monkeypatch.setattr(bag_player.subprocess, "Popen", FakeProcess)

    manager = BagPlayerManager(log_dir=tmp_path)
    player = manager.start(str(bag_path))
    log_path = tmp_path / f"{player['id']}_bag_play.log"
    log_path.write_text("".join(f"bag-line-{index}\n" for index in range(600)))

    tail = manager.tail_log(lines=1000).splitlines()

    assert len(tail) == 500
    assert tail[0] == "bag-line-100"
    assert tail[-1] == "bag-line-599"


def test_select_player_services_prefers_rosbag_player_controls():
    selected = _select_player_services(
        [
            ("/recorder/pause", "rosbag2_interfaces/srv/Pause"),
            ("/rosbag2_player/pause", "rosbag2_interfaces/srv/Pause"),
            ("/rosbag2_player/resume", "rosbag2_interfaces/srv/Resume"),
            ("/rosbag2_player/set_rate", "rosbag2_interfaces/srv/SetRate"),
            ("/rosbag2_player/get_rate", "rosbag2_interfaces/srv/GetRate"),
        ]
    )

    assert selected == {
        "pause": "/rosbag2_player/pause",
        "resume": "/rosbag2_player/resume",
        "set_rate": "/rosbag2_player/set_rate",
        "get_rate": "/rosbag2_player/get_rate",
    }


def test_bag_player_set_rate_uses_detected_service(monkeypatch, tmp_path):
    bag_path = tmp_path / "sample_bag"
    bag_path.mkdir()
    calls = []

    class FakeProcess:
        pid = 34567
        returncode = None

        def __init__(self, command, **kwargs):
            pass

        def poll(self):
            return self.returncode

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["ros2", "service", "list", "-t"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "/rosbag2_player/pause [rosbag2_interfaces/srv/Pause]\n"
                    "/rosbag2_player/resume [rosbag2_interfaces/srv/Resume]\n"
                    "/rosbag2_player/set_rate [rosbag2_interfaces/srv/SetRate]\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="response\n", stderr="")

    monkeypatch.setattr(bag_player.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(bag_player.subprocess, "run", fake_run)

    manager = BagPlayerManager(log_dir=tmp_path)
    manager.start(str(bag_path))
    player = manager.set_rate(2.0)

    assert player["requested_rate"] == 2.0
    assert [
        "ros2",
        "service",
        "call",
        "/rosbag2_player/set_rate",
        "rosbag2_interfaces/srv/SetRate",
        "{rate: 2.0}",
    ] in calls


def test_bag_player_controls_external_rosbag_services(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command == ["ros2", "service", "list", "-t"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "/external_player/pause [rosbag2_interfaces/srv/Pause]\n"
                    "/external_player/resume [rosbag2_interfaces/srv/Resume]\n"
                    "/external_player/set_rate [rosbag2_interfaces/srv/SetRate]\n"
                    "/external_player/get_rate [rosbag2_interfaces/srv/GetRate]\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="response\n", stderr="")

    monkeypatch.setattr(bag_player.subprocess, "run", fake_run)

    manager = BagPlayerManager(log_dir=tmp_path)
    status = manager.status(force_service_refresh=True)
    paused = manager.pause()

    assert status["running"] is False
    assert status["external_player"] is True
    assert status["controls"]["pause"] is True
    assert paused["service_names"]["pause"] == "/external_player/pause"
    assert [
        "ros2",
        "service",
        "call",
        "/external_player/pause",
        "rosbag2_interfaces/srv/Pause",
        "{}",
    ] in calls


def test_browse_bag_path_marks_rosbag_candidates(tmp_path):
    bag_dir = tmp_path / "sample_bag"
    bag_dir.mkdir()
    (bag_dir / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    mcap_file = tmp_path / "single_file.mcap"
    mcap_file.touch()
    text_file = tmp_path / "notes.txt"
    text_file.touch()

    result = browse_bag_path(str(tmp_path))
    entries = {entry["name"]: entry for entry in result["entries"]}

    assert entries["sample_bag"]["kind"] == "directory"
    assert entries["sample_bag"]["playable"] is True
    assert entries["single_file.mcap"]["kind"] == "file"
    assert entries["single_file.mcap"]["playable"] is True
    assert entries["notes.txt"]["playable"] is False
