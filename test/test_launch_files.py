from pathlib import Path

import pytest
from launch import LaunchContext, LaunchDescription, LaunchService
from launch.actions import EmitEvent, IncludeLaunchDescription, TimerAction
from launch.events import Shutdown
from launch.launch_description_sources import AnyLaunchDescriptionSource


@pytest.mark.parametrize(
    "launch_file",
    [
        "open_yolo.launch.xml",
        "open_dfine.launch.xml",
        "open_rfdetr.launch.xml",
    ],
)
def test_open_detector_launcher_parses(launch_file):
    launch_path = Path(__file__).parents[1] / "launch" / launch_file
    source = AnyLaunchDescriptionSource(str(launch_path))
    assert source.get_launch_description(LaunchContext()) is not None


def test_open_detector_node_loads_fake_model_on_start():
    launch_path = Path(__file__).parents[1] / "launch" / "open_detector.launch.xml"
    description = LaunchDescription(
        [
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(str(launch_path)),
                launch_arguments={
                    "camera_namespace": "test_camera",
                    "backend": "fake",
                    "detector_name": "open_fake",
                    "publish_debug_image": "false",
                }.items(),
            ),
            TimerAction(
                period=1.0,
                actions=[EmitEvent(event=Shutdown(reason="launch test complete"))],
            ),
        ]
    )

    service = LaunchService()
    service.include_launch_description(description)
    assert service.run() == 0
