from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from autoware_ml_model_launchers.launcher_dashboard.registry import (
    build_launch_command,
    default_registry_path,
    load_registry,
)
from autoware_ml_model_launchers.screen_camera.capture import (
    CaptureError,
    Region,
    build_x11grab_command,
    normalize_display,
    parse_region,
    parse_xrandr_monitors,
    resolve_region,
    select_backend,
)
from autoware_ml_model_launchers.screen_camera.ros_node import (
    NUMERIC_PARAMETERS,
    PARAMETER_DEFAULTS,
    scaled_size,
)


XRANDR_TWO_MONITORS = """Monitors: 2
 0: +*eDP-1 2560/344x1600/215+0+0  eDP-1
 1: +HDMI-1-0 3840/597x2160/336+2560+0  HDMI-1-0
"""


def test_parse_region_accepts_x11_geometry():
    assert parse_region("1920x1080+2560+0") == Region(1920, 1080, 2560, 0)
    assert parse_region("640x480") == Region(640, 480, 0, 0)
    assert parse_region(" 800x600+10+20 ") == Region(800, 600, 10, 20)


@pytest.mark.parametrize("text", ["", "1920", "1920x", "x1080", "1920*1080", "0x100"])
def test_parse_region_rejects_junk(text):
    with pytest.raises(CaptureError):
        parse_region(text)


def test_parse_xrandr_monitors_reads_physical_layout():
    monitors = parse_xrandr_monitors(XRANDR_TWO_MONITORS)
    assert monitors == [Region(2560, 1600, 0, 0), Region(3840, 2160, 2560, 0)]


def test_resolve_region_prefers_an_explicit_region(monkeypatch):
    monkeypatch.setattr(
        "autoware_ml_model_launchers.screen_camera.capture.list_monitors",
        lambda display=None: [],
    )
    assert resolve_region(region="320x240+5+6") == Region(320, 240, 5, 6)


def test_resolve_region_selects_a_monitor_or_the_whole_screen(monkeypatch):
    monitors = parse_xrandr_monitors(XRANDR_TWO_MONITORS)
    monkeypatch.setattr(
        "autoware_ml_model_launchers.screen_camera.capture.list_monitors",
        lambda display=None: monitors,
    )

    assert resolve_region(monitor=1) == monitors[0]
    assert resolve_region(monitor=2) == monitors[1]
    # monitor 0 is the union of both screens, not just the first one
    assert resolve_region(monitor=0) == Region(6400, 2160, 0, 0)
    with pytest.raises(CaptureError):
        resolve_region(monitor=3)


def test_normalize_display_always_names_a_screen(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    assert normalize_display(":1") == ":1.0"
    assert normalize_display(":0.0") == ":0.0"
    assert normalize_display(None) == ":0.0"
    monkeypatch.setenv("DISPLAY", ":2")
    assert normalize_display() == ":2.0"


def test_x11grab_command_captures_the_region_as_raw_bgr():
    command = build_x11grab_command(Region(1920, 1080, 2560, 0), 12.5, display=":0")

    assert command[command.index("-f") + 1] == "x11grab"
    assert command[command.index("-video_size") + 1] == "1920x1080"
    assert command[command.index("-i") + 1] == ":0.0+2560,0"
    assert command[command.index("-framerate") + 1] == "12.5"
    assert command[command.index("-draw_mouse") + 1] == "0"
    assert command[command.index("-pix_fmt") + 1] == "bgr24"
    assert command[-1] == "-"


def test_x11grab_command_can_draw_the_cursor():
    command = build_x11grab_command(Region(640, 480), 10.0, display=":0", show_cursor=True)
    assert command[command.index("-draw_mouse") + 1] == "1"


def test_select_backend():
    assert select_backend("ffmpeg") == "ffmpeg"
    assert select_backend("auto") in {"mss", "ffmpeg"}
    with pytest.raises(CaptureError):
        select_backend("wayland")


def test_scaled_size_keeps_the_aspect_ratio_when_one_side_is_given():
    assert scaled_size(1920, 1080, 0, 0) == (1920, 1080)
    assert scaled_size(1920, 1080, 960, 0) == (960, 540)
    assert scaled_size(1920, 1080, 0, 540) == (960, 540)
    assert scaled_size(1920, 1080, 640, 640) == (640, 640)


def test_launch_file_publishes_the_compressed_topic_of_the_camera_namespace():
    launch_path = Path(__file__).parents[1] / "launch" / "screen_camera.launch.xml"
    root = ET.parse(launch_path).getroot()
    args = {item.get("name"): item.get("default") for item in root.findall("arg")}
    params = {
        item.get("name"): item.get("value")
        for node in root.findall("node")
        for item in node.findall("param")
    }

    assert args["output/image"] == "/sensing/camera/$(var camera_namespace)/image_raw"
    assert params["output_topic"] == "$(var output/image)/compressed"
    # A live screen has no /clock; defaulting to true would stall the node.
    assert args["use_sim_time"] == "false"


def test_launch_file_only_sets_parameters_the_node_declares():
    launch_path = Path(__file__).parents[1] / "launch" / "screen_camera.launch.xml"
    root = ET.parse(launch_path).getroot()
    params = {
        item.get("name")
        for node in root.findall("node")
        for item in node.findall("param")
    }

    # use_sim_time is declared by rclpy itself, not by the node.
    assert params - {"use_sim_time"} == set(PARAMETER_DEFAULTS)


def test_numeric_parameters_are_declared_dynamically():
    """fps:=4 must work, not only fps:=4.0; a static DOUBLE rejects the integer."""
    assert set(NUMERIC_PARAMETERS) <= set(PARAMETER_DEFAULTS)
    for name in NUMERIC_PARAMETERS:
        assert isinstance(PARAMETER_DEFAULTS[name], (int, float))
    for name, default in PARAMETER_DEFAULTS.items():
        if isinstance(default, bool) or isinstance(default, str):
            continue
        assert name in NUMERIC_PARAMETERS, f"{name} is numeric but statically typed"


def test_registry_entry_builds_a_screen_camera_command():
    registry = load_registry(default_registry_path())
    spec = registry.get("screen_camera")
    command = build_launch_command(spec, {"monitor": 2, "fps": 5.0})

    assert command[:4] == [
        "ros2",
        "launch",
        "autoware_ml_model_launchers",
        "screen_camera.launch.xml",
    ]
    assert "monitor:=2" in command
    assert "fps:=5.0" in command
    assert "use_sim_time:=false" in command
