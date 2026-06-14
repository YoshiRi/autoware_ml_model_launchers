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
            ],
        ),
        (f"share/{package_name}/launch", glob("launch/*.launch.xml")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"lib/{package_name}", ["scripts/check_environment"]),
    ],
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
            "reusable_bbox_tracker_node = autoware_ml_model_launchers.open_tracker.reusable_bbox_tracker_node:main",
            "tlr_yolox_roi_adapter = autoware_ml_model_launchers.tlr_yolox_roi_adapter:main",
        ],
    },
)
