import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest

from autoware_ml_model_launchers.launcher_dashboard.registry import (
    RegistryError,
    default_registry_path,
    load_registry,
)
from autoware_ml_model_launchers.recording.bag_recorder import build_bag_record_command
from autoware_ml_model_launchers.recording.manager import RecordingManager
from autoware_ml_model_launchers.recording.spec import (
    RecordingConfig,
    RecordingError,
    build_manifest,
    infer_sink,
    resolve_recordings,
    topic_slug,
)
from autoware_ml_model_launchers.recording.video_writer import (
    build_ffmpeg_command,
    select_encoder,
)


LAUNCH_DIR = Path(__file__).parents[1] / "launch"


@pytest.fixture
def registry():
    return load_registry(default_registry_path())


@pytest.fixture
def config(registry, tmp_path):
    base = RecordingConfig.from_registry(registry)
    return RecordingConfig(
        output_root=tmp_path,
        session_layout=base.session_layout,
        file_layout=base.file_layout,
        video=base.video,
        bag=base.bag,
    )


def comparison_process(**overrides):
    process = {
        "id": "p1",
        "launcher_id": "tlr_detect_and_classifier",
        "group_id": "comparison:run1",
        "run_id": "run1",
        "variant_id": "tlr_960",
        "command": [
            "ros2",
            "launch",
            "autoware_ml_model_launchers",
            "tlr_detect_and_classifier.launch.xml",
            "camera_namespace:=camera5",
        ],
        "outputs": {
            "output/debug/image": "/evaluation/run1/tlr_960/camera5/tlr/debug/image",
            "output/rois": "/evaluation/run1/tlr_960/camera5/tlr/rois",
            "output/traffic_signals": "/evaluation/run1/tlr_960/camera5/tlr/traffic_signals",
        },
    }
    process.update(overrides)
    return process


# -- registry contract ---------------------------------------------------


def test_registry_record_contract_declares_sinks(registry):
    spec = registry.get("tlr_detect_and_classifier")
    assert spec.record is not None
    assert spec.record.sink_for("output/debug/image") == "video"
    assert spec.record.sink_for("output/traffic_signals") == "bag"
    assert spec.record.sink_for("camera_namespace") is None


def test_record_args_must_exist_in_the_launcher_args(registry):
    for spec in registry.launchers.values():
        if spec.record is None:
            continue
        unknown = sorted(set(spec.record.arg_names()) - set(spec.args))
        assert not unknown, f"{spec.launcher_id} records unknown args: {unknown}"


def test_record_topic_templates_match_launch_file_defaults(registry):
    """The registry duplicates launch defaults, so guard against drift."""
    for spec in registry.launchers.values():
        if spec.record is None or not spec.record.topic_templates:
            continue
        defaults = _launch_arg_defaults(LAUNCH_DIR / spec.file)
        for arg_name, template in spec.record.topic_templates.items():
            assert arg_name in defaults, f"{spec.file} has no arg {arg_name}"
            expected = _template_from_launch_default(defaults[arg_name])
            assert template == expected, f"{spec.file}:{arg_name} drifted from the launch default"


def test_record_spec_rejects_arg_in_both_sinks():
    from autoware_ml_model_launchers.launcher_dashboard.registry import RecordSpec

    with pytest.raises(RegistryError):
        RecordSpec.from_mapping({"video_args": ["a"], "bag_args": ["a"]})


# -- topic resolution ----------------------------------------------------


def test_resolve_uses_isolated_outputs_of_a_comparison_group(config, registry):
    recordings, skipped = resolve_recordings(
        {"from_group_id": "comparison:run1"},
        config,
        "session1",
        registry=registry,
        processes=[comparison_process()],
    )

    by_topic = {item.topic: item for item in recordings}
    debug = by_topic["/evaluation/run1/tlr_960/camera5/tlr/debug/image"]
    assert debug.sink == "video"
    assert debug.stem == "run1/tlr_960_camera5_output_debug_image"
    assert by_topic["/evaluation/run1/tlr_960/camera5/tlr/rois"].sink == "bag"
    # detected_objects is declared but not isolated by this process, and the
    # camera is known, so it falls back to the launch default topic.
    assert any(
        item.topic.endswith("/detection/yolox/objects") for item in recordings
    ), "unisolated args should fall back to the launch default topic"
    assert skipped == []


def test_resolve_falls_back_to_launch_defaults_for_plain_runs(config, registry):
    process = comparison_process(
        group_id=None, run_id=None, variant_id=None, outputs={}
    )
    recordings, skipped = resolve_recordings(
        {"from_process_ids": ["p1"]},
        config,
        "session1",
        registry=registry,
        processes=[process],
    )

    by_topic = {item.topic: item.stem for item in recordings}
    debug_topic = (
        "/perception/traffic_light_recognition/camera5/detection/yolox/debug/image"
    )
    assert debug_topic in by_topic
    # No run or variant: the empty path segments must not leak into the stem.
    assert by_topic[debug_topic] == "tlr_detect_and_classifier_camera5_output_debug_image"
    assert skipped == []


def test_resolve_reports_unresolvable_topics(config, registry):
    process = comparison_process(
        launcher_id="yolox_camera",
        outputs={},
        command=["ros2", "launch", "autoware_ml_model_launchers", "yolox_camera.launch.xml"],
    )
    recordings, skipped = resolve_recordings(
        {"from_process_ids": ["p1"]},
        config,
        "session1",
        registry=registry,
        processes=[process],
    )

    assert recordings == []
    assert {item["arg_name"] for item in skipped} == {
        "output/mask",
        "output/objects",
        "output/tracked_objects",
    }


def test_resolve_explicit_topics_and_sink_inference(config):
    recordings, _ = resolve_recordings(
        {
            "topics": [
                {"topic": "/tlr/debug/image", "name": "tlr_debug"},
                "/tlr/traffic_signals",
            ]
        },
        config,
        "session1",
    )

    assert [(item.topic, item.sink, item.stem) for item in recordings] == [
        ("/tlr/debug/image", "video", "tlr_debug"),
        ("/tlr/traffic_signals", "bag", "tlr_traffic_signals"),
    ]


def test_resolve_filters_by_requested_sink(config, registry):
    recordings, _ = resolve_recordings(
        {"from_group_id": "comparison:run1", "include_sinks": ["video"]},
        config,
        "session1",
        registry=registry,
        processes=[comparison_process()],
    )

    assert {item.sink for item in recordings} == {"video"}


def test_resolve_rejects_a_request_without_matching_processes(config, registry):
    with pytest.raises(RecordingError):
        resolve_recordings(
            {"from_group_id": "comparison:missing"},
            config,
            "session1",
            registry=registry,
            processes=[comparison_process()],
        )


def test_infer_sink_and_topic_slug():
    assert infer_sink("/perception/camera5/debug/image") == "video"
    assert infer_sink("/tensorrt_yolox/out/image/compressed") == "video"
    assert infer_sink("/perception/traffic_signals") == "bag"
    assert topic_slug("/perception/camera5/debug/image") == "perception_camera5_debug_image"


# -- planning ------------------------------------------------------------


def test_preview_builds_one_recorder_per_sink(config, registry):
    manager = RecordingManager(config=config, registry=registry)
    plan = manager.preview(
        {"from_group_id": "comparison:run1", "session_id": "s1"},
        processes=[comparison_process()],
    )

    by_sink = {recorder["sink"]: recorder for recorder in plan["recorders"]}
    assert set(by_sink) == {"video", "bag"}

    video_command = by_sink["video"]["command"]
    assert video_command[1:3] == ["-m", "autoware_ml_model_launchers.recording.video_recorder"]
    assert "/evaluation/run1/tlr_960/camera5/tlr/debug/image" in video_command
    assert "run1/tlr_960_camera5_output_debug_image" in video_command
    assert by_sink["video"]["outputs"][0]["file"].endswith(
        "s1/run1/tlr_960_camera5_output_debug_image.mp4"
    )

    bag_command = by_sink["bag"]["command"]
    assert bag_command[:3] == ["ros2", "bag", "record"]
    assert "--storage" in bag_command
    assert bag_command[bag_command.index("-o") + 1].endswith("s1/run1_bag")


def test_preview_does_not_create_files(config, registry, tmp_path):
    manager = RecordingManager(config=config, registry=registry)
    manager.preview({"from_group_id": "comparison:run1"}, processes=[comparison_process()])
    assert list(tmp_path.iterdir()) == []


def test_bag_record_command_needs_topics():
    with pytest.raises(ValueError):
        build_bag_record_command(Path("/tmp/out"), [])
    command = build_bag_record_command(Path("/tmp/out"), ["/a", "/b"], storage=None)
    assert command == ["ros2", "bag", "record", "-o", "/tmp/out", "/a", "/b"]


# -- manifest ------------------------------------------------------------


def test_manifest_round_trips_run_metadata(tmp_path):
    manifest = build_manifest(
        "s1",
        tmp_path,
        [{"sink": "video", "topic": "/a/image", "file": "run1/a.mp4", "frames": 12}],
        processes=[comparison_process()],
        bag={"path": "/data/a.mcap", "rate": 0.2},
    )
    text = json.dumps(manifest)

    assert manifest["schema"] == "launcher_recording/1"
    assert manifest["processes"][0]["variant_id"] == "tlr_960"
    assert manifest["recordings"][0]["frames"] == 12
    assert json.loads(text)["bag"]["rate"] == 0.2


def test_finalize_writes_a_manifest_for_an_empty_session(config, registry, tmp_path):
    manager = RecordingManager(config=config, registry=registry)
    manager.session_id("s1")
    result = manager.finalize()

    manifest_path = Path(result["manifest_path"])
    assert manifest_path == tmp_path / "s1" / "manifest.json"
    assert json.loads(manifest_path.read_text())["session_id"] == "s1"
    with pytest.raises(RecordingError):
        manager.finalize()  # the session was cleared


# -- video encoder -------------------------------------------------------


def test_select_encoder():
    assert select_encoder("opencv") == "opencv"
    assert select_encoder("auto") in {"ffmpeg", "opencv"}
    with pytest.raises(ValueError):
        select_encoder("nvenc")


def test_ffmpeg_command_uses_raw_bgr_input_and_browser_playable_output():
    command = build_ffmpeg_command(Path("/tmp/a.mp4"), 1920, 1080, 9.5, 20)
    assert "rawvideo" in command
    assert command[command.index("-s") + 1] == "1920x1080"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt", command.index("-c:v")) + 1] == "yuv420p"
    assert command[-1] == "/tmp/a.mp4"


# -- bag playlist --------------------------------------------------------


def test_resolve_bag_paths_accepts_a_single_bag_a_list_and_an_at_file(tmp_path):
    from autoware_ml_model_launchers.launcher_dashboard.bag_player import resolve_bag_paths

    listing = tmp_path / "play_order.txt"
    listing.write_text("\n".join(["# comment", "", "/data/a.db3", "/data/b.db3"]))

    assert resolve_bag_paths("/data/a.mcap") == [Path("/data/a.mcap")]
    assert resolve_bag_paths(None, ["/data/a", "/data/b"]) == [Path("/data/a"), Path("/data/b")]
    assert resolve_bag_paths(f"@{listing}") == [Path("/data/a.db3"), Path("/data/b.db3")]
    with pytest.raises(ValueError):
        resolve_bag_paths(None, None)
    with pytest.raises(ValueError):
        resolve_bag_paths(f"@{tmp_path / 'missing.txt'}")


def test_playlist_reports_running_between_two_bags(tmp_path):
    from autoware_ml_model_launchers.launcher_dashboard.bag_player import BagPlayerManager

    manager = BagPlayerManager(tmp_path)
    # Simulate the gap after the first bag exited and before the next launches.
    manager._playlist = [tmp_path / "a", tmp_path / "b"]
    manager._playlist_index = 0
    assert manager.is_running() is True
    assert manager.playlist_status() == {"index": 0, "total": 2, "current": str(tmp_path / "a")}

    manager._playlist_index = 1
    assert manager.is_running() is False
    assert manager.playlist_status()["index"] == 1

    # A finished playlist must not walk off the end of the list.
    manager._playlist_index = 2
    assert manager.playlist_status() == {"index": 1, "total": 2, "current": str(tmp_path / "b")}
    assert manager.is_running() is False


# -- session runner ------------------------------------------------------


def test_load_sessions_merges_defaults(tmp_path):
    from autoware_ml_model_launchers.recording.session import load_sessions

    config = tmp_path / "sessions.yaml"
    config.write_text(
        "defaults:\n"
        "  rate: 0.2\n"
        "  camera_namespace: camera6\n"
        "sessions:\n"
        "  - id: a\n"
        "    bags: /data/a.mcap\n"
        "    launcher: {launcher_id: yolox_camera}\n"
        "  - id: b\n"
        "    bags: /data/b.mcap\n"
        "    rate: 1.0\n"
        "    comparison: {run_id: r}\n"
    )

    sessions = load_sessions(config)
    assert [session["rate"] for session in sessions] == [0.2, 1.0]
    assert sessions[0]["camera_namespace"] == "camera6"
    assert sessions[0]["startup_timeout"] == 900.0  # from the built-in defaults


def test_load_sessions_rejects_incomplete_entries(tmp_path):
    from autoware_ml_model_launchers.recording.session import SessionError, load_sessions

    config = tmp_path / "sessions.yaml"
    config.write_text("sessions:\n  - id: a\n    bags: /data/a.mcap\n")
    with pytest.raises(SessionError):
        load_sessions(config)

    config.write_text("sessions: []\n")
    with pytest.raises(SessionError):
        load_sessions(config)


def test_session_clips_expand_pairs_and_inherit_playback_defaults():
    from autoware_ml_model_launchers.recording.session import session_clips

    session = {
        "id": "batch",
        "rate": 0.2,
        "clock": True,
        "loop": False,
        "clips": [
            {"name": "scene_a", "bag": "/data/a.mcap"},
            {"name": "scene_b", "bag": "/data/b.mcap", "rate": 1.0},
            {"bag": "/data/c.mcap"},
        ],
    }

    clips = session_clips(session)
    assert [clip["name"] for clip in clips] == ["scene_a", "scene_b", "clip_3"]
    assert [clip["rate"] for clip in clips] == [0.2, 1.0, 0.2]
    assert all(clip["clock"] for clip in clips)


def test_session_without_clips_is_one_unnamed_clip():
    from autoware_ml_model_launchers.recording.session import session_clips

    clips = session_clips(
        {"id": "single", "bags": "/data/a.mcap", "rate": 1.0, "clock": True, "loop": False}
    )
    assert clips == [
        {"name": "", "bags": "/data/a.mcap", "rate": 1.0, "clock": True, "loop": False}
    ]


def test_session_clips_reject_duplicates_and_missing_bags():
    from autoware_ml_model_launchers.recording.session import SessionError, session_clips

    base = {"id": "batch", "rate": 1.0, "clock": True, "loop": False}
    with pytest.raises(SessionError):
        session_clips({**base, "clips": [{"name": "a", "bag": "/x"}, {"name": "a", "bag": "/y"}]})
    with pytest.raises(SessionError):
        session_clips({**base, "clips": [{"name": "a"}]})


def test_clip_gives_every_output_its_own_directory(config, registry):
    """Two clips of one batch must not resolve to the same file."""
    stems = {}
    for clip in ("scene_a", "scene_b"):
        recordings, _ = resolve_recordings(
            {"from_group_id": "comparison:run1", "clip": clip},
            config,
            "batch",
            registry=registry,
            processes=[comparison_process()],
        )
        stems[clip] = {item.stem for item in recordings}

    assert all(stem.startswith("scene_a/") for stem in stems["scene_a"])
    assert all(stem.startswith("scene_b/") for stem in stems["scene_b"])
    assert not stems["scene_a"] & stems["scene_b"]


def test_clip_placed_by_the_layout_is_not_prefixed_again(registry, tmp_path):
    layout_config = RecordingConfig(
        output_root=tmp_path,
        session_layout="{session_id}",
        file_layout="{clip}_{variant_id}_{arg_name}",
        video=RecordingConfig().video,
        bag=RecordingConfig().bag,
    )
    recordings, _ = resolve_recordings(
        {"from_group_id": "comparison:run1", "clip": "scene_a"},
        layout_config,
        "batch",
        registry=registry,
        processes=[comparison_process()],
    )

    assert "scene_a_tlr_960_output_debug_image" in {item.stem for item in recordings}


def test_explicit_topic_names_are_scoped_to_the_clip(config):
    recordings, _ = resolve_recordings(
        {"topics": [{"topic": "/tlr/debug/image", "name": "tlr_debug"}], "clip": "scene_a"},
        config,
        "batch",
    )
    assert recordings[0].stem == "scene_a/tlr_debug"


def test_colliding_output_names_are_rejected_with_a_fix(registry, tmp_path):
    collapsing_config = RecordingConfig(
        output_root=tmp_path,
        session_layout="{session_id}",
        file_layout="{clip}",
        video=RecordingConfig().video,
        bag=RecordingConfig().bag,
    )
    manager = RecordingManager(config=collapsing_config, registry=registry)

    second_variant = comparison_process(
        id="p2",
        variant_id="tlr_1280",
        outputs={"output/debug/image": "/evaluation/run1/tlr_1280/camera5/tlr/debug/image"},
    )
    with pytest.raises(RecordingError, match="same output file"):
        manager.preview(
            {"from_group_id": "comparison:run1", "clip": "scene_a"},
            processes=[comparison_process(), second_variant],
        )


def test_manifest_carries_clips_for_a_batch_run(tmp_path):
    manifest = build_manifest(
        "batch",
        tmp_path,
        [{"sink": "video", "clip": "scene_a", "file": "scene_a/a.mp4", "frames": 10}],
        clips=[{"name": "scene_a", "bags": ["/data/a.mcap"], "rate": 0.2}],
    )

    assert manifest["clips"][0]["bags"] == ["/data/a.mcap"]
    assert manifest["recordings"][0]["clip"] == "scene_a"


def test_sample_session_config_plans_without_starting_anything(tmp_path, monkeypatch):
    from autoware_ml_model_launchers.recording import session as session_module

    sample = Path(__file__).parents[1] / "samples" / "record_session.example.yaml"
    sessions = session_module.load_sessions(sample)
    assert [item["id"] for item in sessions] == [
        "odaiba_tlr_phase2",
        "odaiba_yolox",
        "tlr_regression_batch",
    ]
    assert len(session_module.session_clips(sessions[2])) == 3

    # The playlist file of the sample only exists on the author's machine.
    monkeypatch.setattr(
        session_module, "resolve_bag_paths", lambda *args, **kwargs: [Path("/data/a.mcap")]
    )
    registry = load_registry(default_registry_path())
    for entry in sessions:
        entry["output_root"] = str(tmp_path)
        result = session_module._plan_session(entry, registry)
        assert result["ok"], result["detail"]
    assert list(tmp_path.iterdir()) == []


def _launch_arg_defaults(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    return {
        element.get("name"): element.get("default", "")
        for element in root.findall("arg")
        if element.get("name")
    }


def _template_from_launch_default(default: str) -> str:
    return re.sub(r"\$\(var ([^)]+)\)", r"{\1}", default)
