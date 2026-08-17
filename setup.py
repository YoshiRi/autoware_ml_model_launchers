from glob import glob

from setuptools import find_packages, setup

package_name = "autoware_ml_model_launchers"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (
            f"share/{package_name}",
            [
                "package.xml",
                "README.md",
                "requirements-open-detector-common.txt",
                "requirements-open-detector-dfine.txt",
                "requirements-open-detector-rfdetr.txt",
                "requirements-open-tracker-bbox.txt",
                "requirements-open-yolo.txt",
                "requirements-screen-camera.txt",
            ],
        ),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/docs", glob("docs/*.md")),
        (f"lib/{package_name}", ["scripts/check_environment"]),
    ],
    package_data={
        package_name: [
            "launcher_dashboard/static/*.html",
            "param_snapshot/static/*.html",
        ],
    },
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="YoshiRi",
    maintainer_email="yoshiri@example.com",
    description="Launchers and small helper nodes for Autoware ML model validation.",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "compressed_yolo_node = autoware_ml_model_launchers.compressed_yolo_node:main",
            "open_detector_node = autoware_ml_model_launchers.open_detector.ros_node:main",
            "open_detector_image = autoware_ml_model_launchers.open_detector.cli_detect_image:main",
            "open_detector_smoke = autoware_ml_model_launchers.open_detector.smoke_backends:main",
            "launcher_dashboard_ui = autoware_ml_model_launchers.launcher_dashboard.ui_server:main",
            "record_topics = autoware_ml_model_launchers.recording.cli:record_topics_main",
            "record_session = autoware_ml_model_launchers.recording.cli:record_session_main",
            "record_video = autoware_ml_model_launchers.recording.video_recorder:main",
            "param_snapshot_compare = autoware_ml_model_launchers.param_snapshot.compare:main",
            "param_snapshot_ui = autoware_ml_model_launchers.param_snapshot.ui_server:main",
            "reusable_bbox_tracker_node = autoware_ml_model_launchers.open_tracker.reusable_bbox_tracker_node:main",
            "screen_camera_node = autoware_ml_model_launchers.screen_camera.ros_node:main",
            "tlr_yolox_roi_adapter = autoware_ml_model_launchers.tlr_yolox_roi_adapter:main",
        ],
    },
)
